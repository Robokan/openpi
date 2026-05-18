#!/usr/bin/env python3
"""Compare JAX vs PyTorch outputs of embed_prefix (before any LLM layer).

Identifies whether divergence is in:
  - Image encoding (vision tower)
  - Text tokenization
  - Token concatenation
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
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: jax={a.shape} pt={b.shape}")
        return
    cos = float((a.flatten() @ b.flatten()) / (np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-12))
    print(f"  [{name:35s}] shape={a.shape}  cos={cos:+.6f}  |jax|={np.linalg.norm(a):.3f}  |pt|={np.linalg.norm(b):.3f}  max|diff|={float(np.max(np.abs(a-b))):.4f}")


def _load_obs():
    import subprocess
    import cv2
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
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp),
             "-vf", f"select=eq(n\\,{fidx})", "-vframes", "1", "-f", "image2", name],
            capture_output=True, timeout=15, check=True,
        )
        frame = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def main() -> int:
    raw_obs = _load_obs()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    print("Loading both policies ...")
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)
    print()

    # Apply both policies' input transforms to get identical model-ready inputs
    import jax
    import jax.numpy as jnp
    import torch

    inputs = jax.tree.map(lambda x: x, raw_obs)
    inputs = jax_policy._input_transform(inputs)
    print("Input transform produced keys:", list(inputs.keys()))

    # Show tokenized_prompt to verify both backends get same tokens
    tok = inputs["tokenized_prompt"]
    tok_mask = inputs["tokenized_prompt_mask"]
    print(f"tokenized_prompt shape={tok.shape}  first 32 tokens: {tok[:32].tolist()}")
    print(f"non-pad count: {int(tok_mask.sum())}")
    print()

    # Build JAX Observation
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)

    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    # Re-preprocess to get same image-format inputs both backends will pass to vision tower
    jax_obs_p = _model.preprocess_observation(None, jax_obs, train=False)
    pt_model = pt_policy._model
    jax_model = jax_policy._model

    # JAX embed_prefix
    jax_prefix, jax_pad, jax_ar = jax_model.embed_prefix(jax_obs_p)
    jax_prefix_np = np.asarray(jax_prefix).astype(np.float32)
    print(f"JAX prefix: shape={jax_prefix_np.shape}  range=[{float(jax_prefix_np.min()):.3f}, {float(jax_prefix_np.max()):.3f}]")
    print(f"JAX prefix mask sum: {int(np.asarray(jax_pad).sum())}")
    print()

    # PT embed_prefix
    with torch.no_grad():
        pt_images, pt_img_masks, pt_lang_tokens, pt_lang_masks, pt_state = pt_model._preprocess_observation(pt_obs, train=False)
        pt_prefix, pt_pad, pt_ar = pt_model.embed_prefix(pt_images, pt_img_masks, pt_lang_tokens, pt_lang_masks)
    pt_prefix_np = pt_prefix.detach().float().cpu().numpy()
    print(f"PT  prefix: shape={pt_prefix_np.shape}  range=[{pt_prefix_np.min():.3f}, {pt_prefix_np.max():.3f}]")
    print(f"PT  prefix mask sum: {int(pt_pad.sum().item())}")
    print()

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    _stats("full prefix embeddings", jax_prefix_np, pt_prefix_np)

    # Split into image tokens (first N) vs language tokens (remainder)
    # 3 cameras x 256 patches = 768 image tokens
    n_img = 768
    _stats("image tokens (first 768)", jax_prefix_np[:, :n_img], pt_prefix_np[:, :n_img])
    _stats("lang tokens (>=768)     ", jax_prefix_np[:, n_img:], pt_prefix_np[:, n_img:])

    # Per camera (256 tokens each)
    for i, name in enumerate(["base", "left_wrist", "right_wrist"]):
        s, e = i * 256, (i + 1) * 256
        _stats(f"image tokens cam {name}", jax_prefix_np[:, s:e], pt_prefix_np[:, s:e])

    return 0


if __name__ == "__main__":
    sys.exit(main())
