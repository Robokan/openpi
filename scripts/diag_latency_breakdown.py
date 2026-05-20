#!/usr/bin/env python3
"""Per-stage latency breakdown of sample_actions.

Mirrors TurboPi's reported breakdown:
  Vision -> Embed Prefix -> Prefill (KV cache compute) -> 10x Denoise -> Total
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _load_obs():
    import cv2
    import pyarrow.parquet as pq
    import av
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
            frame_np = None
            for i, frame in enumerate(container.decode(container.streams.video[0])):
                if i == fidx:
                    frame_np = frame.to_ndarray(format="rgb24"); break
        if frame_np.shape[:2] != (224, 224):
            frame_np = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame_np, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def _t(stmt, n=5, warmup=1):
    """Run stmt() n times after warmup, return median ms."""
    import torch
    for _ in range(warmup):
        stmt()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    times = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = stmt()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.median(times)) * 1000.0, out


def main():
    print("=" * 100)
    print(f"Latency breakdown | OPENPI_PYTORCH_PRECISION={os.environ.get('OPENPI_PYTORCH_PRECISION','bfloat16')}"
          f"  OPENPI_PT_QUANT={os.environ.get('OPENPI_PT_QUANT','')}"
          f"  OPENPI_PT_DISABLE_COMPILE={os.environ.get('OPENPI_PT_DISABLE_COMPILE','0')}"
          f"  OPENPI_PT_FP32_ATTN={os.environ.get('OPENPI_PT_FP32_ATTN','1')}")
    print("=" * 100)

    raw_obs = _load_obs()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import torch
    import jax

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    t0 = time.time()
    policy = _pc.create_trained_policy(cfg, CKPT_PT)
    print(f"Loaded in {time.time()-t0:.1f}s")
    pt_model = policy._model
    pt_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = policy._input_transform(raw_obs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to(device), inputs)
    pt_dtype = pt_model.action_in_proj.weight.dtype
    pt_inputs = jax.tree.map(lambda t: t.to(pt_dtype) if t.is_floating_point() else t, pt_inputs)
    obs = _model.Observation.from_dict(pt_inputs)

    horizon = pt_model.config.action_horizon
    adim = pt_model.config.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    noise = torch.from_numpy(noise_np).to(device).to(pt_dtype)

    print("\nStage timings (median of 5 iters, after 1 warmup):")
    print("-" * 100)

    # 1) full sample_actions
    def _full():
        with torch.no_grad():
            return pt_model.sample_actions(device, obs, noise=noise, num_steps=10)
    t_total, _ = _t(_full)
    print(f"  [full sample_actions(num_steps=10)        ] {t_total:7.1f} ms")

    # 2) preprocess + vision + embed prefix (do it manually to time)
    def _preproc_embed():
        with torch.no_grad():
            images, img_masks, lang_tokens, lang_masks, state = pt_model._preprocess_observation(obs, train=False)
            prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            return prefix_embs, prefix_pad_masks, prefix_att_masks, state
    t_embed, (prefix_embs, prefix_pad_masks, prefix_att_masks, state) = _t(_preproc_embed)
    print(f"  [preproc + vision + embed_prefix          ] {t_embed:7.1f} ms")

    # 3) compute_prefix_kv_cache
    def _prefill():
        with torch.no_grad():
            return pt_model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
    t_prefill, kv = _t(_prefill)
    print(f"  [compute_prefix_kv_cache                  ] {t_prefill:7.1f} ms")

    # 4) one denoise_step_with_cache
    def _one_denoise():
        with torch.no_grad():
            tnow = torch.tensor(0.5, dtype=torch.float32, device=device).expand(1)
            x = noise
            return pt_model.denoise_step_with_cache(state, kv, prefix_pad_masks, x, tnow)
    t_one, _ = _t(_one_denoise)
    print(f"  [denoise_step_with_cache (1 step)         ] {t_one:7.1f} ms")
    print(f"  [denoise_step_with_cache x 10 (estimated) ] {t_one*10:7.1f} ms")

    print("-" * 100)
    print(f"  [sum-of-parts estimate                    ] {t_embed + t_prefill + t_one*10:7.1f} ms")
    print(f"  [unaccounted overhead                     ] {t_total - (t_embed + t_prefill + t_one*10):7.1f} ms")

    # Identify dominant component
    print(f"\nDominant components:")
    parts = [("embed+vision", t_embed), ("prefill", t_prefill), ("10x denoise", t_one*10)]
    parts.sort(key=lambda p: -p[1])
    for name, ms in parts:
        print(f"  {name:20s} {ms:7.1f} ms ({ms/t_total*100:.0f}% of total)")


if __name__ == "__main__":
    main()
