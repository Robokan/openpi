#!/usr/bin/env python3
"""Compare KV-cache fast path vs joint-forward slow path in PT.

If joint-forward gives JAX parity but compute_prefix_kv_cache doesn't,
the bug is in our custom KV cache code, NOT the model itself.
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

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
    print(f"  [{name:40s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}")


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


def main():
    raw_obs = _load_obs()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import jax, jax.numpy as jnp, torch

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)
    jax_model = jax_policy._model
    pt_model = pt_policy._model
    pt_model.eval()

    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = jax_model.action_horizon
    adim = jax_model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    noise_jax = jnp.asarray(noise_np)
    noise_pt = torch.from_numpy(noise_np).to("cuda")
    rng = jax.random.PRNGKey(0)

    print("=" * 100)
    print("JAX 10-step diffusion")
    print("=" * 100)
    jax_actions = jax_model.sample_actions(rng, jax_obs, noise=noise_jax, num_steps=10)
    jax_actions_np = np.asarray(jax_actions.astype(jnp.float32))
    print(f"  norm={np.linalg.norm(jax_actions_np):.3f}")

    print()
    print("=" * 100)
    print("PT KV-cache fast path (default)")
    print("=" * 100)
    os.environ["OPENPI_PT_NO_KVCACHE"] = "0"
    with torch.no_grad():
        pt_actions_fast = pt_model.sample_actions("cuda", pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_fast_np = pt_actions_fast.detach().float().cpu().numpy()
    print(f"  norm={np.linalg.norm(pt_actions_fast_np):.3f}")

    print()
    print("=" * 100)
    print("PT joint-forward slow path (OPENPI_PT_NO_KVCACHE=1)")
    print("=" * 100)
    os.environ["OPENPI_PT_NO_KVCACHE"] = "1"
    with torch.no_grad():
        pt_actions_slow = pt_model.sample_actions("cuda", pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_slow_np = pt_actions_slow.detach().float().cpu().numpy()
    print(f"  norm={np.linalg.norm(pt_actions_slow_np):.3f}")

    print()
    print("=" * 100)
    print("COMPARISONS")
    print("=" * 100)
    _stats("JAX vs PT KV-cache fast", jax_actions_np, pt_actions_fast_np)
    _stats("JAX vs PT joint-forward", jax_actions_np, pt_actions_slow_np)
    _stats("PT fast vs PT slow", pt_actions_fast_np, pt_actions_slow_np)


if __name__ == "__main__":
    main()
