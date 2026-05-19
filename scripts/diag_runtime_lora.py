#!/usr/bin/env python3
"""Compare JAX vs PT-with-runtime-LoRA inference end-to-end.

PT now loads `lora.safetensors` alongside `model.safetensors` and applies
LoRA at every forward pass via einsum (no pre-merge). This should match
JAX's two-matmul order exactly in bf16.

Success criterion: cos > 0.9999 and ratio in [0.999, 1.001].
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
        print(f"  [{name}] SHAPE MISMATCH: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
    nj = float(np.linalg.norm(a)); npt = float(np.linalg.norm(b))
    rel_err = float(np.linalg.norm(a - b) / (nj + 1e-30))
    print(f"  [{name:36s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}  rel_err={rel_err:.4e}")
    return cos, npt / (nj + 1e-12), rel_err


def _load_obs():
    import cv2
    import pyarrow.parquet as pq
    import av  # PyAV instead of ffmpeg subprocess
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
        with av.open(str(vp)) as container:
            stream = container.streams.video[0]
            frame_np = None
            for i, frame in enumerate(container.decode(stream)):
                if i == fidx:
                    frame_np = frame.to_ndarray(format="rgb24")
                    break
            if frame_np is None:
                raise RuntimeError(f"Could not extract frame {fidx} from {vp}")
        if frame_np.shape[:2] != (224, 224):
            frame_np = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame_np, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def main():
    raw_obs = _load_obs()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import jax
    import jax.numpy as jnp
    import torch

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")

    print("=" * 100)
    print(f"JAX checkpoint: {CKPT_JAX}")
    print(f"PT checkpoint:  {CKPT_PT}")
    print("=" * 100)

    print("Loading JAX policy...")
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    print("Loading PT policy (with runtime LoRA)...")
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)
    jax_model = jax_policy._model
    pt_model = pt_policy._model
    pt_model.eval()

    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to(device), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = jax_model.action_horizon
    adim = jax_model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    noise_jax = jnp.asarray(noise_np)
    # Cast noise to match action_in_proj dtype (bf16 by default)
    pt_dtype = pt_model.action_in_proj.weight.dtype
    noise_pt = torch.from_numpy(noise_np).to(device).to(pt_dtype)
    rng = jax.random.PRNGKey(0)

    print()
    print("Running JAX + PT (runtime LoRA) full 10-step diffusion...")
    jax_actions = jax_model.sample_actions(rng, jax_obs, noise=noise_jax, num_steps=10)
    jax_actions_np = np.asarray(jax_actions.astype(jnp.float32))
    with torch.no_grad():
        pt_actions = pt_model.sample_actions(device, pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_np = pt_actions.detach().float().cpu().numpy()

    print()
    print("=" * 100)
    print("RAW ACTIONS")
    print("=" * 100)
    _stats("raw JAX vs PT", jax_actions_np, pt_actions_np)

    jax_final = jax_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": jax_actions_np[0]})
    pt_final = pt_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": pt_actions_np[0]})
    print()
    print("POST-UNNORMALIZED ACTIONS (what the robot sees)")
    cos, ratio, rel = _stats("post-unnorm JAX vs PT", np.asarray(jax_final["actions"]), np.asarray(pt_final["actions"]))

    print()
    print("=" * 100)
    if cos is not None and cos > 0.9999 and 0.999 < ratio < 1.001:
        print("SUCCESS: PT runtime LoRA matches JAX!")
    elif cos is not None and cos > 0.999 and 0.99 < ratio < 1.01:
        print("CLOSE but not perfect parity - tiny remaining bf16 noise expected.")
    else:
        print("PARITY NOT ACHIEVED - investigate further.")
    print("=" * 100)


if __name__ == "__main__":
    main()
