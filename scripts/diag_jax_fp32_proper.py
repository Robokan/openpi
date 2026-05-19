#!/usr/bin/env python3
"""PROPERLY force JAX to fp32 by deep-walking the flax linen scan dict
structure (not just nnx.iter_graph which only sees outer leaves).

If JAX-fp32 vs PT-fp32 then produces ratio=1.0 -> bug is bf16 rounding
order (merged weights have different rounding than runtime base+lora).
If still ratio=0.918 -> bug is structural / something else.
"""
from __future__ import annotations

import os, sys, json, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

os.environ["OPENPI_PYTORCH_PRECISION"] = "float32"

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
    nj = float(np.linalg.norm(a)); npt = float(np.linalg.norm(b))
    print(f"  [{name:36s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}")


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


def _deep_cast_jax_to_fp32(obj, path=()):
    """Recursively walk a flax nnx-wrapped model and cast every bf16 array
    leaf to fp32. Handles dicts (flax linen scan output), nnx.Variable
    instances, and nested attributes."""
    import jax.numpy as jnp
    from flax import nnx

    n_cast = 0

    if isinstance(obj, nnx.Variable):
        if hasattr(obj, "value") and hasattr(obj.value, "dtype"):
            if obj.value.dtype == jnp.bfloat16:
                obj.value = obj.value.astype(jnp.float32)
                return 1
        return 0

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, nnx.Variable):
                if hasattr(v, "value") and hasattr(v.value, "dtype"):
                    if v.value.dtype == jnp.bfloat16:
                        v.value = v.value.astype(jnp.float32)
                        n_cast += 1
            else:
                n_cast += _deep_cast_jax_to_fp32(v, path + (k,))
        return n_cast

    if hasattr(obj, "__dict__"):
        for attr_name in list(vars(obj).keys()):
            try:
                attr = getattr(obj, attr_name)
            except Exception:
                continue
            n_cast += _deep_cast_jax_to_fp32(attr, path + (attr_name,))

    return n_cast


def main():
    raw_obs = _load_obs()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import jax, jax.numpy as jnp, torch

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")

    print("=" * 100)
    print("Loading JAX + PT in fp32")
    print("=" * 100)
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)
    jax_model = jax_policy._model
    pt_model = pt_policy._model
    pt_model.eval()

    # Deep-walk cast JAX params to fp32
    print("Deep-walk casting JAX bf16 -> fp32...")
    n_cast = _deep_cast_jax_to_fp32(jax_model)
    print(f"  Cast {n_cast} JAX leaves to fp32")

    # Verify by sampling a known weight
    pg = jax_model.PaliGemma
    sample_path = pg.llm.layers
    if isinstance(sample_path, dict) and "attn" in sample_path:
        attn = sample_path["attn"]
        if isinstance(attn, dict):
            # Print dtype of first attn weight we find
            for k, v in attn.items():
                if hasattr(v, "value") and hasattr(v.value, "dtype"):
                    print(f"  Sample attn.{k}.value.dtype = {v.value.dtype}")
                    break

    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to(pt_device), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = jax_model.action_horizon
    adim = jax_model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    noise_jax = jnp.asarray(noise_np)
    noise_pt = torch.from_numpy(noise_np).to(pt_device)
    rng = jax.random.PRNGKey(0)

    print()
    print("Running JAX (fp32, deep-cast) + PT (fp32) 10-step diffusion...")
    jax_actions = jax_model.sample_actions(rng, jax_obs, noise=noise_jax, num_steps=10)
    jax_actions_np = np.asarray(jax_actions.astype(jnp.float32))
    with torch.no_grad():
        pt_actions = pt_model.sample_actions(pt_device, pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_np = pt_actions.detach().float().cpu().numpy()

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    _stats("JAX fp32 vs PT fp32 (raw)", jax_actions_np, pt_actions_np)
    jax_final = jax_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": jax_actions_np[0]})
    pt_final = pt_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": pt_actions_np[0]})
    _stats("JAX fp32 vs PT fp32 (post-unnorm)", np.asarray(jax_final["actions"]), np.asarray(pt_final["actions"]))

    print()
    print("Conclusion:")
    print(f"  If ratio ~ 1.0 -> bug is bf16 rounding order (merge-then-matmul vs base+lora).")
    print(f"  If ratio ~ 0.918 -> bug is structural and persists even in pure fp32.")


if __name__ == "__main__":
    main()
