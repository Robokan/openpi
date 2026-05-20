# FlashRT JAX inference on DGX Spark — runbook

End-to-end checklist for running the existing `pi05_openarm_ngc_lora_v4`
Orbax checkpoint through FlashRT on DGX Spark, replacing the openpi JAX
server without touching the robot client.

Plan reference: `~/.cursor/plans/flashrt_jax_on_dgx_spark_f0288dd4.plan.md`

---

## Layout

```
~/sparkpack/
├── FlashRT/                         # FlashRT source + Spark patches
│   ├── CMakeLists.txt               # SM_121 patches (Phase 0)
│   ├── docker/Dockerfile.spark      # Grace+Blackwell build (Phase 0)
│   ├── flash_rt/frontends/jax/pi05_rtx.py   # LoRA merge in fp32 (Phase 2)
│   ├── flash_rt/frontends/torch/pi05_rtx.py # robot_action_dim arg (Phase 4)
│   ├── tests/test_lora_merge_jax_loader.py  # LoRA merge unit tests
│   └── scripts/
│       ├── spark_build_smoke.py             # Phase 0 gate
│       ├── spark_phase1_libero_smoke.py     # Phase 1 smoke (no LIBERO sim)
│       ├── spark_phase1_libero_run.sh       # Phase 1 full LIBERO eval
│       ├── spark_phase2_lora_load.py        # Phase 2 gate (LoRA merge)
│       └── spark_phase3_run_calib.py        # Phase 3 gate (FP8 calibrate)
└── openpi/
    ├── scripts/
    │   ├── docker/compose_ngc.yml           # adds flashrt_spark service
    │   ├── serve_policy_flashrt.py          # Phase 5 server
    │   ├── spark_phase3_prepare_calib.py    # Phase 3 part 1 (data prep)
    │   ├── spark_phase4_parity.py           # Phase 4 gate (vs JAX server)
    │   ├── spark_phase5_adapter_smoke.py    # Phase 5 wire test
    │   ├── spark_phase6_robot_compare.py    # Phase 6 record + compare
    │   └── spark_phase7_latency_breakdown.py # Phase 7 colocation decision
    └── src/openpi/serving/
        └── flashrt_policy_adapter.py        # FlashRT VLAModel → BasePolicy
```

---

## Phase 0 — Build FlashRT for SM_121 + aarch64

```bash
cd ~/sparkpack/openpi
docker compose -f scripts/docker/compose_ngc.yml build flashrt_spark
```

This applies the CMakeLists patches that treat SM_121 like SM_120 in the
consumer-Blackwell code paths and emits both `sm_120` and `sm_121` SASS
for FA2. First build is 10–20 minutes (CUTLASS template codegen).

**Gate**:

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm flashrt_spark \
    python3 /workspace/FlashRT/scripts/spark_build_smoke.py
```

Exits 0 only if `get_gpu_sm_version() == 121`, `supports_nvfp4() == True`,
and a small fa2 bf16 kernel returns finite output.

## Phase 1 — pi05_libero sanity

Download `pi05_libero` Orbax (~12 GB) to `/openpi_assets/`. Then:

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm flashrt_spark \
    python3 /workspace/FlashRT/scripts/spark_phase1_libero_smoke.py \
        --checkpoint /openpi_assets/pi05_libero
```

Confirms the FlashRT JAX frontend loads + runs end-to-end with no LoRA
involvement. The full LIBERO eval (separate, requires libero/MuJoCo
installed) is launched via `spark_phase1_libero_run.sh`.

## Phase 2 — LoRA Orbax load

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm flashrt_spark \
    python3 /workspace/FlashRT/scripts/spark_phase2_lora_load.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4
```

Verifies the new `_maybe_merge_lora` in
`flash_rt/frontends/jax/pi05_rtx.py` detects + fuses every LoRA pair
into base weights in fp32, before the existing bf16 truncation step.
Default scaling = 1.0 (matches openpi r=16/α=16 and r=32/α=32 LoRA
configs); override via `FLASHRT_LORA_SCALING` env var.

Unit tests are at `FlashRT/tests/test_lora_merge_jax_loader.py`:

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm flashrt_spark \
    pytest /workspace/FlashRT/tests/test_lora_merge_jax_loader.py -v
```

## Phase 3 — FP8 calibration on OpenArm data

```bash
# Inside the openpi training container (has lerobot installed):
docker compose -f scripts/docker/compose_ngc.yml run --rm openpi_server_ngc \
    python /openpi/scripts/spark_phase3_prepare_calib.py \
        --repo-id local/openarm-teleop-16dof \
        --num-samples 80 \
        --output /openpi_assets/calib_openarm_80.npz

# Then inside the flashrt_spark container:
docker compose -f scripts/docker/compose_ngc.yml run --rm flashrt_spark \
    python3 /workspace/FlashRT/scripts/spark_phase3_run_calib.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \
        --calib-data /openpi_assets/calib_openarm_80.npz \
        --percentile 99.9
```

Writes the FP8 cache to `~/.flash_rt/calibration/<hash>_Se<N>.json`.

## Phase 4 — Parity vs the openpi JAX server

```bash
# Shell A: start the JAX server (existing setup, port 8001):
docker compose -f scripts/docker/compose_ngc.yml run --rm \
    -p 8001:8001 openpi_serve \
    python scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_openarm_ngc_lora \
        --policy.dir=/openpi_assets/pi05_openarm_ngc_lora_v4 \
        --port=8001

# Shell B: run the parity harness from the flashrt container:
docker compose -f scripts/docker/compose_ngc.yml run --rm flashrt_spark \
    python3 /openpi/scripts/spark_phase4_parity.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \
        --calib-data /openpi_assets/calib_openarm_80.npz \
        --jax-server localhost:8001 \
        --robot-action-dim 16
```

Acceptance: post-unnorm cosine ≥ 0.999, ratio ∈ [0.995, 1.005] on every
sample. The `robot_action_dim=16` arg routes through the new
`FLASHRT_ROBOT_ACTION_DIM` knob and the three patched sites in
`pi05_rtx.py`.

## Phase 5 — Serve FlashRT over the openpi websocket protocol

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm \
    -p 8001:8001 flashrt_spark \
    python3 /openpi/scripts/serve_policy_flashrt.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \
        --robot-action-dim 16 \
        --calib-data /openpi_assets/calib_openarm_80.npz \
        --default-prompt "pick up the red block" \
        --port 8001
```

Wire test (in another shell):

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm openpi_server_ngc \
    python /openpi/scripts/spark_phase5_adapter_smoke.py --port 8001
```

This confirms the `WebsocketClientPolicy` (which `AsyncActionChunkBroker`
wraps internally) gets back exactly the same response shape the JAX
server produces — `actions`, `policy_timing.infer_ms`,
`_rtc_chunk_model_space`.

## Phase 6 — Real robot comparison

Run the same robot client against the JAX server, then against the
FlashRT server, recording outcomes via:

```bash
python3 scripts/spark_phase6_robot_compare.py append \
    --log jax_baseline.json --server jax --task pick_red_block \
    --success true --episode-steps 87
```

When both logs have enough episodes (≥ 20 per task is the rule of thumb):

```bash
python3 scripts/spark_phase6_robot_compare.py compare \
    --baseline jax_baseline.json --candidate flashrt_spark.json
```

Acceptance: overall ratio ≥ 0.95.

## Phase 7 — Colocation decision

With FlashRT serving:

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm \
    openpi_server_ngc \
    python /openpi/scripts/spark_phase7_latency_breakdown.py --port 8001
```

The script prints a per-component table and emits a recommendation:

- If JPEG decode > 10% of inference → switch the robot client to send
  raw uint8 arrays (`obs["image"] = np.uint8(...)`). The server already
  passes raw arrays through unchanged; only the client side changes.
- If client-side overhead (decode + msgpack + ws) > 50% of inference →
  investigate colocation. Otherwise the websocket boundary is free.

---

## Quick reference — environment vars

| var | effect |
|---|---|
| `FLASHRT_LORA_SCALING` | LoRA scaling factor passed to `_maybe_merge_lora` (default 1.0) |
| `FLASHRT_ROBOT_ACTION_DIM` | Robot DOF slice (default 7 = LIBERO; OpenArm = 16) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX preallocation fraction (0.6 in compose_ngc.yml) |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | false for inference (less aggressive than training) |
| `FVK_PI05_RTX_FORCE_BF16` | Bypass FP8 for debugging (1 = bf16 only) |

## Quick reference — failure modes

| symptom | likely phase | fix |
|---|---|---|
| `import flash_rt` fails | 0 | rebuild image; check sm_121 was applied |
| `supports_nvfp4()` returns False | 0 | CMakeLists patch not applied; rebuild |
| inference produces NaNs | 1, 2, 4 | check LoRA merge ran, scales not saturated |
| post-unnorm cosine 0.92 | 2 | LoRA pre-merged in bf16 — verify Phase 2 patch |
| post-unnorm cosine ~ 0.98 | 3 | calibration set too small or unstratified |
| wrong action dim in response | 4 | pass `--robot-action-dim 16` (OpenArm) |
| robot misses every grasp | 4 → 6 | run parity again with `--robot-action-dim 16` |
| latency > 200 ms | 7 | check JPEG decode share; switch to raw arrays |
