# π0.5 PyTorch vs JAX Parity Debug Log

**Goal:** Get the PyTorch policy server (port 8002) to produce actions
indistinguishable from the JAX policy server (port 8001) for the
`pi05_openarm_ngc_lora_v4 / chocolate_bars_pi05` checkpoint, so we can
deploy PT on DGX Spark / AGX Thor with FP4/FP8 quantization.

**Symptom:** Robot drifts arms upward when using PT server. JAX server works
correctly. Single-step model parity is excellent (`cos=0.995`) but
end-to-end actions after 10 diffusion steps + Unnormalize show consistent
**+0.135 rad bias on joint 3 (shoulder pitch)**.

---

## Setup

| Component | Value |
|-----------|-------|
| Robot | OpenArm 16-DOF dual-arm |
| Policy | `pi05_openarm_ngc_lora_v4` config |
| Checkpoint | `chocolate_bars_pi05` step 29999 (LoRA-trained) |
| JAX ckpt | `/app/checkpoints/.../chocolate_bars_pi05/29999/` |
| PT ckpt | `/app/checkpoints/.../chocolate_bars_pi05_pytorch/` (converted) |
| JAX server | `openpi-jax-server` container, port 8001 ✅ |
| PT server | `openpi-pt-server` container, port 8002 ❌ |
| Hardware | DGX Spark (NVIDIA GB10, ARM64) |
| Model arch | PaliGemma 2B + gemma_300m_lora expert (18 layers, AdaRMS) |
| Diffusion | 10 Euler steps, dt=-0.1, t=1.0→0.0 |
| Action | 50 horizon × 32 dim (16 dim used by robot, padded to 32) |

Run JAX server:
```bash
docker run -d --name openpi-jax-server --gpus all --network host \
  openpi-jax-server:latest \
  python scripts/serve_policy.py --port 8001 policy:checkpoint \
  --policy.config pi05_openarm_ngc_lora_v4 \
  --policy.dir checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999
```

Run PT server (currently `openpi-thor:latest` for CUDA):
```bash
docker run -d --name openpi-pt-server --gpus all --network host \
  -v /home/evaughan/sparkpack/openpi:/app \
  -v /home/evaughan/.cache/openpi:/openpi_assets \
  -v /home/evaughan/.cache/huggingface:/root/.cache/huggingface \
  -w /app \
  -e PYTHONPATH=/app/src:/app/packages/openpi-client/src \
  openpi-thor:latest \
  python scripts/serve_policy.py --port 8002 policy:checkpoint \
  --policy.config pi05_openarm_ngc_lora_v4 \
  --policy.dir checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch
```

---

## Bug summary (CURRENT)

Running `diag_full_loop.py` with **identical fixed noise (seed=42)** and
**identical real observation**:

| Layer | cos(JAX, PT) | norm(PT)/norm(JAX) | max\|diff\| |
|-------|------|--------|------|
| `embed_prefix` image tokens | 0.9999 | ~1.0 | small |
| `embed_prefix` lang tokens (after fix) | 1.0 | 1.0 | 0 |
| `action_in_proj` output | 0.999999 | 1.0000 | 0.01 |
| `adarms_cond` (time MLP) | 0.999998 | 1.0001 | 0.001 |
| paligemma after 18 layers (prefix) | 0.97 | ~1.0 | – |
| `suffix_out` after 18 expert layers | 0.991 | 1.001 | 0.75 |
| `v_t` single step (action_out_proj) | 0.995 | 0.993 | 0.80 |
| Final actions (10 Euler steps, raw) | 0.9958 | **0.917** (PT 8% smaller) | 0.16 |
| Post-Unnormalize actions | 0.963 | **1.087** (PT 8.7% LARGER in real space) | 0.18 |
| **Joint 3 (shoulder pitch) diff** | – | – | **+0.135 rad** |

The per-step v_t parity is excellent (cos=0.995, ratio=0.993). The 0.7%
per-step magnitude bias compounds over 10 diffusion steps to ~8%
chunk-level bias. After Unnormalize (which multiplies by std), high-std
joints get amplified disproportionately, and the sign flips because the
state values are positive (joint 3 state = 2.008 rad).

**Net effect:** PT consistently overshoots upward on shoulder pitch joints
(3 and 11) by ~0.13 rad per chunk, causing the arms to slowly drift up.

---

## Fixes ALREADY applied (do not redo)

These are committed; their effects on parity are below.

1. **LoRA merge fix** (commit pre-history)
   - `examples/convert_jax_model_to_pytorch.py`: `_merge_lora_into_base()`
     correctly handles both `attn/X/lora_a + .../w` and FFN-style
     `gating_einsum_lora_a + gating_einsum`. Without this the converted PT
     checkpoint was the un-fine-tuned base model.
   - Verified: cos=1.0 on every weight tensor (`diag_weight_match.py`).

2. **`load_pytorch` `strict=False, assign=True`**
   - `src/openpi/models/model.py`: handles bf16 safetensors loaded into
     fp32-initialized model parameters without dtype-mismatch errors.

3. **Re-tie `embed_tokens.weight = lm_head.weight`** AFTER load
   - `src/openpi/models/model.py`: `assign=True` breaks HuggingFace's
     constructor-time weight tying. The safetensors file only contains
     `lm_head.weight` (default safetensors behavior for tied weights). After
     `assign=True`, `embed_tokens.weight` was left at random init, making
     ALL language token embeddings random.
   - Effect: lang token cos went from **0.0016 → 1.0** at `embed_prefix`.
     Massive improvement: motion delta cos went from ~0 → 0.48-0.74.

4. **`fp32_attention` helper** for attention logits + softmax
   - `src/openpi/models_pytorch/gemma_pytorch.py`: matches JAX's
     `jnp.einsum(..., preferred_element_type=jnp.float32)` semantics.
   - Effect on PaliGemma prefix forward: lang token cos went from
     **0.65 → 0.97**, lang norm ratio 1.279 → 1.011.
   - **However**: did not fix end-to-end action parity (still 8% bias).

5. **`make_att_2d_masks` ordering matches JAX `make_attn_mask`**
   - Verified both produce identical masks on the same inputs.

6. **KV cache** ported from TurboPi
   - `src/openpi/models_pytorch/pi0_pytorch.py:compute_prefix_kv_cache` +
     `denoise_step_with_cache`. ~7× inference speedup vs joint forward.
   - Verified: joint-forward vs KV-cache give same result.

7. **`time_emb` precision**: kept as fp32 (JAX semantics) — `posemb_sincos`
   in JAX returns fp32.

8. **GPU container**: switched from CPU-only `openpi-jax-server` image
   to `openpi-thor:latest` with `PYTHONPATH=/app/src:...` override so the
   container picks up the live source from the mounted workspace.

---

## Things ruled out (with evidence)

For each, the diagnostic that proved it's not the bug:

### Weights
- All weights match cos=1.0 between JAX and PT (`diag_weight_match.py`).
  Includes AdaRMS dense kernel/bias, action_in_proj, action_out_proj,
  time_mlp_in/out, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj, input_layernorm, post_attention_layernorm, final_norm.

### Input pipeline
- Image preprocessing: cos=0.9999 on SigLIP output.
- Tokenization: identical tokens (verified character-by-character).
- `embed_prefix`: cos=1.0 on language tokens (after weight-tying fix),
  cos=0.9999 on image tokens.
- `_input_transform`: state values identical between JAX and PT through
  the entire input pipeline.

### Normalization stats
- `norm_stats.json` md5 matches between JAX and PT checkpoint dirs.
  Confirmed via `md5sum`.

### Output transforms
- `_output_transform` is the same code path for both JAX and PT
  (constructed from the same `data_config`).
- Unnormalize uses the same norm_stats.

### Precision
- bf16 vs fp32 throughout: gives **identical** end-to-end actions
  (tested with `OPENPI_PYTORCH_PRECISION=float32`). The 8% bias is
  preserved either way.
- TF32 vs strict fp32 matmul (`OPENPI_PT_MATMUL_PRECISION=highest`):
  gives **identical** actions to default `"high"`. TF32 is not the bug.
- `fp32_attention` enabled/disabled (`OPENPI_PT_FP32_ATTN`): improves
  per-layer prefix parity but does not change end-to-end action bias.

### Diffusion loop
- JAX and PT loops are byte-equivalent:
  - dt = -1.0 / num_steps = -0.1
  - while time >= -dt/2: x_t = x_t + dt * v_t; time += dt
  - both start with x_t = noise, time = 1.0
- Initial noise: when `noise` arg is passed to both, they use the same
  values.
- num_steps defaults to 10 on both sides.

### KV cache vs joint forward
- `OPENPI_PT_NO_KVCACHE=1` (slow joint-forward path through
  PaliGemmaWithExpert.forward) produces same result as KV-cache path
  (cos=0.9999 between the two PT paths).

### Time embedding
- `posemb_sincos` (JAX) vs `create_sinusoidal_pos_embedding` (PT):
  produce identical `time_emb` vectors at fp32. Verified via direct
  comparison.

### AdaRMS / final_norm
- Dense kernel/bias for AdaRMS match cos=1.0.
- The PT path correctly applies `gemma_expert.norm(x, cond=adarms_cond)`
  after the 18 layers.

### Attention mask
- `make_att_2d_masks(pad_masks, att_masks)` (PT) and `make_attn_mask`
  (JAX) produce identical bool masks for the same inputs (verified by
  direct equality check on bool tensors).
- Suffix self-attention is full bidirectional (all True). Cross-attention
  to prefix masks padding correctly.

### Output dimensionality
- Both produce (1, 50, 32) action tensors. Robot uses only first 16 dims.
- Slicing [:, :, :16] vs full [:, :, :] shows same cos and ratio.

### gemma_expert lm_head
- `gemma_expert.model.embed_tokens` is set to `None` (expert doesn't
  embed tokens; gets pre-embedded suffix input).
- `gemma_expert.lm_head` is loaded but never called (we use
  `action_out_proj` instead).
- The "Could not re-tie gemma_expert embed_tokens to lm_head" warning
  during load is BENIGN — embed_tokens is intentionally None.

### Vision tower
- SigLIP vision tower: image token cos=0.9999 vs JAX. Not the bug.

---

## TurboPi comparison (`/home/evaughan/sparkpack/TurboPi/openpi`)

TurboPi achieves 100% LIBERO accuracy with their PT impl. Compared
their code to ours:

| Item | TurboPi | Ours |
|------|---------|------|
| Attention in `compute_prefix_kv_cache` | SDPA, bf16, with mask | fp32_attention with mask |
| Attention in `denoise_step_with_cache` | SDPA, bf16, **NO mask, NO scale** | fp32_attention with mask |
| LoRA merging in converter | **NONE** (libero ckpt has no LoRA) | Full LoRA merge |
| `time_emb` dtype | cast to model_dtype (bf16) | kept as fp32 |
| `embed_suffix` action cast | `noisy_actions.to(model_dtype)` | direct |
| Final norm with adarms | identical | identical |
| Per-layer adarms application | `cond=adarms_cond` | `cond=adarms_cond` |

**Key observations:**
- TurboPi's success on LIBERO does **not** prove their impl is correct —
  LIBERO might be tolerant to bf16 drift / padding-unmasked attention.
- TurboPi has NO LoRA handling. Their working result is for a
  no-LoRA checkpoint. Our bug could be specific to LoRA-merged inference.

---

## Diagnostic scripts (use these to test)

All in `scripts/`. Run inside `openpi-pt-server` container:
```bash
docker exec openpi-pt-server bash -lc \
  "cd /app && PYTHONPATH=/app/src:/app/packages/openpi-client/src \
   python scripts/<diag_name>.py"
```

| Script | Purpose | Current result |
|--------|---------|----------------|
| `compare_jax_pytorch.py` | Compare server outputs on real obs (random noise) | cos=0.04-0.7, ratio=1.5-2.6 (noisy) |
| `compare_servers_live.py` | Same but via running servers (HTTP) | varies |
| `diag_weight_match.py` | Per-tensor weight comparison | cos=1.0 everywhere ✅ |
| `diag_embed_prefix.py` | `embed_prefix` token-by-token | image cos=0.9999, lang cos=1.0 ✅ |
| `diag_per_layer.py` | Per-layer hidden state diff in PaliGemma prefix | layer 0–17 norms+cos |
| `diag_model_prefix.py` | Final prefix output via `compute_prefix_kv_cache` | cos=0.97 (lang fixed) |
| `diag_one_step.py` | Single denoise step v_t comparison | cos=0.995, ratio=0.99 |
| **`diag_suffix.py`** | Single denoise step: action_in_proj → suffix_out → v_t | **see breakdown above** |
| **`diag_full_loop.py`** | Full 10-step diffusion + post-Unnormalize, fixed noise | **cos=0.996 raw, 0.963 post-xfm, +0.135 on joint 3** |

---

## Env vars for debugging

| Env var | Default | Effect |
|---------|---------|--------|
| `OPENPI_PYTORCH_PRECISION` | `bfloat16` | `float32` runs everything in fp32 |
| `OPENPI_PT_NO_KVCACHE` | `0` | `1` uses slow joint-forward path |
| `OPENPI_PT_FP32_ATTN` | (on) | `0` would disable fp32 attention if we add the flag |
| `OPENPI_PT_MATMUL_PRECISION` | `high` | `highest` for strict fp32, `medium` for bf16 |

---

## Still-open investigation paths

These haven't been ruled out yet:

1. **Per-layer suffix hidden state divergence**
   - We've done per-layer for the PREFIX. Haven't done it for the SUFFIX
     (gemma_expert layers). The cos=0.991 on suffix_out means each of the
     18 expert layers loses ~0.05% cosine + ~0.04% norm.
   - Need: monkey-patch JAX `Block.__call__` to capture per-layer
     gemma_expert outputs, then compare to PT layer-by-layer.
   - Initial attempt with `jax.disable_jit()` failed due to
     `TracerArrayConversionError`. Need a different approach (e.g.,
     `jax.experimental.callback` or rebuild the model without
     `nn.scan`).

2. **AdaRMS Dense modulation dtype**
   - JAX: `nn.Dense(...)` with `dtype=x.dtype` (= bf16) → modulation in bf16
   - PT: `nn.Linear` with weight kept in fp32 → modulation in fp32
   - PT modulation is MORE precise. Probably HELPS PT, not hurts. But
     worth verifying by forcing bf16 modulation in PT (one-line cast).

3. **LoRA merge math precision**
   - JAX: `out = w @ x + scaling * (lora_a @ lora_b) @ x` (3 matmuls, bf16)
   - PT: `out = (w + scaling * lora_a @ lora_b) @ x` (1 matmul, bf16)
   - Different rounding. JAX has MORE rounding error per LoRA application.
     PT has less. But this should mean PT is MORE accurate, not less.
   - To rule out: implement runtime LoRA on PT side (store lora_a/lora_b
     separately, apply JAX-style at every forward). Slow but proves it.

4. **HuggingFace transformers Gemma layer details**
   - We use `transformers_replace/models/gemma/modeling_gemma.py` which
     overrides the standard HF code. Audit this for any deviation from
     JAX semantics, especially in:
     - `_gated_residual`
     - `apply_rotary_pos_emb`
     - `eager_attention_forward`
     - `GemmaMLP.forward`

5. **RoPE precision**
   - JAX `_apply_rope` uses fp32 cos/sin.
   - PT `rotary_emb` precision: needs verification.
   - max_wavelength = 10000 on both. Confirmed.

6. **Suffix attention mask**
   - JAX masks padding tokens in prefix when suffix attends to them.
   - TurboPi does NOT mask (works on LIBERO).
   - We match JAX (mask padding). But the JAX TRAINING-time code might
     differ from sample-time. Worth double-checking JAX training mask.

7. **Numerical differences in suffix self-attention vs prefix forward**
   - In sample_actions, suffix tokens have positions
     `prefix_len + cumsum(suffix_mask) - 1`. Positions for suffix go
     768+200=968 through 1017.
   - Cross-check RoPE values at these high positions between JAX and PT.

8. **action_out_proj bias values**
   - PT bias values are small (~0.01–0.02 range). Verified non-zero.
   - Need: directly compare `m.action_out_proj.bias` values between JAX
     `_model.action_out_proj.bias` and PT counterpart (cos + max diff).

9. **gemma_expert specifically vs paligemma**
   - The paligemma prefix forward has good parity (cos=0.97 lang, 0.97
     image). The gemma_expert suffix forward also has cos=0.991.
   - Question: is the BIAS in gemma_expert systematically different from
     paligemma? E.g., do certain Q/K/V projections diverge more?

10. **Try Triton-based JAX-equivalent matmul kernels**
    - JAX (XLA) might use slightly different matmul algorithms than
      PyTorch (cuBLAS / CUTLASS). Same precision, different reduction
      order. Could account for the 0.05% per-layer drift.
    - Probably impossible to eliminate without a custom kernel.

11. **Compile vs eager**
    - Current code: `self.sample_actions = torch.compile(..., mode='max-autotune')`.
    - Comment in `pi0_pytorch.py` notes that EAGER gives cos=-0.96 (!)
      on pi05_libero and compile fixes it to 0.85. That's a HUGE swing
      from compile. Suggests compile is silently doing some
      precision/algorithmic substitution.
    - Worth testing: run with `OPENPI_PT_DISABLE_COMPILE=1` (if we add
      this) and see what eager gives on chocolate_bars.

---

## Action items (next session)

In priority order:

- [ ] Per-layer **suffix** (gemma_expert) hidden state diff (analog of
      `diag_per_layer.py` but for the expert). Will pinpoint which of
      the 18 expert layers introduces the most drift.
- [ ] Verify `action_out_proj.bias` matches between JAX and PT.
- [ ] Try forcing bf16 modulation in PT AdaRMS to match JAX exactly.
- [ ] Audit `transformers_replace/models/gemma/modeling_gemma.py` for
      any deviation from JAX `Block.__call__` (line by line).
- [ ] Test eager (no torch.compile) on chocolate_bars to see if compile
      is masking a bigger bug.
- [ ] Implement runtime LoRA on PT side (parallel base + lora paths) to
      rule out LoRA merge precision as the source.
- [ ] If all else fails: a **calibration hack** — apply a per-joint
      scale factor learned from JAX vs PT comparison on a held-out
      validation set. Pragmatic last resort to get the robot working
      while we keep debugging.

---

## Commit history (relevant)

```
197a88d pi0_pytorch: make matmul precision env-var configurable (default 'high')
b89bada diag: add suffix + full-loop comparisons that isolate residual PT bias
621d5c1 [previous: fp32 attention fix for prefix lang tokens]
[older: weight-tying re-tie fix in model.py]
[older: LoRA merge in convert_jax_model_to_pytorch.py]
[older: KV cache port from TurboPi]
```
