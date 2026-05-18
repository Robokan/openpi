#!/usr/bin/env python3
"""Compare JAX vs PT inference WITHOUT LoRA contribution.

Rationale:
  The PT model has a 0.7%/step v_t bias compounding to 8% chunk-level
  magnitude bias vs JAX (with the LoRA-merged checkpoint). The user
  asks: if FP4 quantization works (per NVIDIA Jetson AI Lab), then bf16
  precision can't cause an 8% bias. The bug must be structural.

  Possibilities:
    1. The base model code (PaliGemma + Gemma300M) is fine in PT, and
       the bug is in the LoRA conversion (PT merged LoRA differs from
       JAX runtime LoRA at this rounding scale).
    2. The base model code itself has a structural bug; LoRA is innocent.

  This script tests it: zero out the LoRA contribution on BOTH sides
  and compare. If the bias persists, base model has the bug.

JAX side: zero lora_a / lora_b at runtime so `result + (x@0@0)*scaling = result`.
PT  side: use the already-existing chocolate_bars_pi05_pytorch_BROKEN_NO_LORA
          checkpoint, which was produced WITHOUT the LoRA merge fix.

Fixed noise (seed=42) so any divergence is due to the model, not RNG.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT_NO_LORA = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_NO_LORA_v2"
CKPT_PT_LORA = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    nj = float(np.linalg.norm(a)); npt = float(np.linalg.norm(b))
    print(f"  [{name:30s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}  max|d|={float(np.max(np.abs(a-b))):.4f}")


def _load_obs():
    import subprocess, cv2
    import pyarrow.parquet as pq
    ds = Path("/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parquet = sorted(ds.glob("data/chunk-*/episode_*.parquet"))[0]
    table = pq.read_table(parquet)
    row = 100
    state = np.asarray(table.column("observation.state")[row].as_py(), dtype=np.float32)
    fidx = int(table.column("frame_index")[row].as_py())
    tidx = int(table.column("task_index")[row].as_py())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (ds / "meta" / "tasks.jsonl").read_text().splitlines()}
    prompt = tasks.get(tidx, "do the task")
    chunk, ep = parquet.parent.name, parquet.stem
    images = {}
    for cam_in, cam_out in (("ego", "cam_high"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist")):
        vp = ds / "videos" / chunk / f"observation.images.{cam_in}" / f"{ep}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            name = tmp.name
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp), "-vf", f"select=eq(n\\,{fidx})", "-vframes", "1", "-f", "image2", name], capture_output=True, timeout=15, check=True)
        frame = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def _zero_jax_lora(jax_model):
    """Walk the JAX model and zero out every lora_a / lora_b parameter."""
    import jax, jax.numpy as jnp
    # nnx-bridged Flax modules expose params via .variables
    n_zeroed = 0
    pg = jax_model.PaliGemma
    # PaliGemma.llm has scan-stored block params under .layers.variables
    state = pg.llm.layers.variables
    if "params" in state:
        params_tree = state["params"]
    else:
        params_tree = state

    def maybe_zero(path, leaf):
        nonlocal n_zeroed
        # path is a tuple of keys
        pstr = "/".join(str(p) for p in path)
        if "lora_a" in pstr or "lora_b" in pstr:
            n_zeroed += 1
            return jnp.zeros_like(leaf)
        return leaf

    new_params = jax.tree_util.tree_map_with_path(
        lambda kp, leaf: maybe_zero([k.key if hasattr(k, "key") else str(k) for k in kp], leaf),
        params_tree,
    )
    # Try a direct dict walk instead — flax variables are typically dict-like
    def walk(obj, path=()):
        nonlocal n_zeroed
        if hasattr(obj, "keys"):
            for k in list(obj.keys()):
                v = obj[k]
                full = path + (str(k),)
                if "lora_a" in str(k) or "lora_b" in str(k):
                    if hasattr(v, "value"):
                        v.value = jnp.zeros_like(v.value)
                    else:
                        obj[k] = jnp.zeros_like(v)
                    n_zeroed += 1
                else:
                    walk(v, full)

    walk(state)
    return n_zeroed


def _zero_jax_lora_v2(jax_model):
    """Alternative approach: walk all nnx variables of the model and zero
    out tensors whose path contains 'lora_a' or 'lora_b'."""
    from flax import nnx
    import jax.numpy as jnp

    n_zeroed = 0
    # nnx.Variable instances have a `.value` attribute
    for path, leaf in nnx.iter_graph(jax_model):
        pstr = "/".join(str(p) for p in path) if isinstance(path, tuple) else str(path)
        if isinstance(leaf, nnx.Variable):
            if "lora_a" in pstr or "lora_b" in pstr:
                leaf.value = jnp.zeros_like(leaf.value)
                n_zeroed += 1
    return n_zeroed


def main():
    raw_obs = _load_obs()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import jax, jax.numpy as jnp, torch

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")

    print("=" * 100)
    print("Step 1: Load JAX policy + zero LoRA")
    print("=" * 100)
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    n_zeroed = 0
    for fn in (_zero_jax_lora_v2, _zero_jax_lora):
        try:
            n_zeroed = fn(jax_policy._model)
            print(f"  {fn.__name__}: zeroed {n_zeroed} LoRA tensors")
            if n_zeroed > 0:
                break
        except Exception as e:  # noqa: BLE001
            print(f"  {fn.__name__} failed: {e}")
    if n_zeroed == 0:
        print("  WARNING: No LoRA params zeroed! Result will include LoRA.")

    print()
    print("=" * 100)
    print("Step 2: Load PT no-LoRA policy")
    print("=" * 100)
    if not Path(CKPT_PT_NO_LORA).exists():
        print(f"  ERROR: {CKPT_PT_NO_LORA} does not exist!")
        return
    pt_policy_no_lora = _pc.create_trained_policy(cfg, CKPT_PT_NO_LORA)
    print(f"  Loaded PT no-LoRA from {CKPT_PT_NO_LORA}")

    print()
    print("=" * 100)
    print("Step 3: Compare via fixed-noise full diffusion loop")
    print("=" * 100)
    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = jax_policy._model.action_horizon
    adim = jax_policy._model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    print(f"  Fixed noise norm={np.linalg.norm(noise_np):.3f}")

    # JAX
    print()
    print("  --- JAX (no-LoRA via zeroed lora_a/lora_b) ---")
    noise_jax = jnp.asarray(noise_np)
    rng = jax.random.PRNGKey(0)
    jax_actions = jax_policy._model.sample_actions(rng, jax_obs, noise=noise_jax, num_steps=10)
    jax_actions_np = np.asarray(jax_actions.astype(jnp.float32))
    print(f"  JAX no-lora actions: |norm|={np.linalg.norm(jax_actions_np):.3f}")
    print(f"    first action [:16]: {jax_actions_np[0, 0, :16]}")

    # PT no-LoRA
    print()
    print("  --- PT (no-LoRA from BROKEN_NO_LORA checkpoint) ---")
    pt_model = pt_policy_no_lora._model
    pt_model.eval()
    # The NO_LORA_v2 checkpoint was produced with the same converter as the
    # working LoRA-merged checkpoint, so projections are fp32 (matching what
    # the running PT code expects). Pass fp32 noise as usual.
    _w_dtype = pt_model.action_in_proj.weight.dtype
    print(f"  action_in_proj.weight.dtype = {_w_dtype}")
    print(f"  time_mlp_in.weight.dtype = {pt_model.time_mlp_in.weight.dtype if hasattr(pt_model, 'time_mlp_in') else 'N/A'}")
    noise_pt = torch.from_numpy(noise_np).to("cuda")
    with torch.no_grad():
        pt_actions = pt_model.sample_actions("cuda", pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_np = pt_actions.detach().float().cpu().numpy()
    print(f"  PT no-lora actions:  |norm|={np.linalg.norm(pt_actions_np):.3f}")
    print(f"    first action [:16]: {pt_actions_np[0, 0, :16]}")

    print()
    print("=" * 100)
    print("COMPARISON: JAX (no-LoRA) vs PT (no-LoRA)")
    print("=" * 100)
    _stats("actions (raw, no LoRA)", jax_actions_np, pt_actions_np)
    _stats("actions[:, :, :16]", jax_actions_np[:, :, :16], pt_actions_np[:, :, :16])
    _stats("first action [0, 0]", jax_actions_np[:, 0, :16], pt_actions_np[:, 0, :16])
    print(f"  diff[:16]: {pt_actions_np[0, 0, :16] - jax_actions_np[0, 0, :16]}")

    # Apply output transforms for direct comparison with diag_full_loop
    print()
    print("--- After unnormalize ---")
    jax_output = {"state": np.asarray(inputs["state"]), "actions": jax_actions_np[0]}
    pt_output = {"state": np.asarray(inputs["state"]), "actions": pt_actions_np[0]}
    jax_final = jax_policy._output_transform(jax_output)
    pt_final = pt_policy_no_lora._output_transform(pt_output)
    _stats("post-unnormalize", np.asarray(jax_final["actions"]), np.asarray(pt_final["actions"]))
    print(f"  JAX post-xfm joint 3 (shoulder pitch) action[0]: {jax_final['actions'][0, 3]:.5f}")
    print(f"  PT  post-xfm joint 3 (shoulder pitch) action[0]: {pt_final['actions'][0, 3]:.5f}")
    print(f"  diff: {pt_final['actions'][0, 3] - jax_final['actions'][0, 3]:.5f}")


if __name__ == "__main__":
    main()
