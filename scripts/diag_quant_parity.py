#!/usr/bin/env python3
"""Validate quantized PT model vs unquantized fp32 PT baseline.

First run (no env var) caches the fp32 baseline to disk. Subsequent runs
with `OPENPI_PT_QUANT={fp8_w,fp8_wa,nvfp4}` compare against the cached baseline.

Usage:
  python scripts/diag_quant_parity.py                        # fp32 baseline (cached)
  OPENPI_PT_QUANT=fp8_w  python scripts/diag_quant_parity.py # FP8 weight-only
  OPENPI_PT_QUANT=fp8_wa python scripts/diag_quant_parity.py # FP8 dynamic act+w
  OPENPI_PT_QUANT=nvfp4  python scripts/diag_quant_parity.py # NVFP4 W4A4
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

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"
BASELINE_PATH = "/tmp/diag_quant_fp32_baseline.npz"


def _stats(name: str, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: a={a.shape} b={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    ratio = nb / (na + 1e-30)
    rel_err = float(np.linalg.norm(a - b) / (na + 1e-30))
    print(f"  [{name:40s}] cos={cos:+.6f}  |a|={na:>9.3f}  |b|={nb:>9.3f}  ratio={ratio:.4f}  rel_err={rel_err:.4e}")


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
            if frame_np is None:
                raise RuntimeError(f"Could not extract frame {fidx} from {vp}")
        if frame_np.shape[:2] != (224, 224):
            frame_np = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame_np, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def main():
    quant_mode = os.environ.get("OPENPI_PT_QUANT", "").strip() or "fp32"
    fast_attn = os.environ.get("OPENPI_PT_FAST_ATTN", "1")
    is_baseline_run = quant_mode == "fp32" and fast_attn == "0" and "DIAG_BASELINE" in os.environ
    label_extras = []
    if fast_attn == "1":
        label_extras.append("fast_attn")
    label = "fp32 baseline" if is_baseline_run else f"quant={quant_mode}" + (" " + ",".join(label_extras) if label_extras else "")
    print("=" * 100)
    print(f"Loading PT policy: {label}")
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
    print(f"  loaded in {time.time()-t0:.1f}s")
    pt_model = policy._model
    pt_model.eval()

    inputs = policy._input_transform(raw_obs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to(device), inputs)
    pt_dtype = pt_model.action_in_proj.weight.dtype
    pt_inputs = jax.tree.map(lambda t: t.to(pt_dtype) if t.is_floating_point() else t, pt_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = pt_model.config.action_horizon
    adim = pt_model.config.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    noise_pt = torch.from_numpy(noise_np).to(device).to(pt_dtype)

    n_warmup = int(os.environ.get("DIAG_WARMUP", "1"))
    n_iters = int(os.environ.get("DIAG_ITERS", "3"))
    print(f"  running 10-step diffusion (warmup={n_warmup}, iters={n_iters}, action_in_proj dtype={pt_dtype})...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = pt_model.sample_actions(device, pt_obs, noise=noise_pt, num_steps=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        times = []
        for _ in range(n_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            actions = pt_model.sample_actions(device, pt_obs, noise=noise_pt, num_steps=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    dt = float(np.median(times))
    actions_np = actions.detach().float().cpu().numpy()
    print(f"  iters={[round(t*1000) for t in times]}ms  median={dt*1000:.0f}ms")

    un = policy._output_transform({"state": np.asarray(inputs["state"]), "actions": actions_np[0]})
    actions_un = np.asarray(un["actions"])

    if is_baseline_run:
        np.savez(BASELINE_PATH, raw=actions_np, post_unnorm=actions_un, latency_ms=dt * 1000)
        print(f"  cached fp32 baseline (fp32_attention) -> {BASELINE_PATH}")
        return

    if not os.path.exists(BASELINE_PATH):
        print(f"  !! no baseline at {BASELINE_PATH} — run without OPENPI_PT_QUANT first")
        return
    baseline = np.load(BASELINE_PATH)
    print()
    print("=" * 100)
    print(f"Parity: {quant_mode} vs fp32 baseline")
    print("=" * 100)
    _stats(f"{quant_mode} vs fp32 raw",        baseline["raw"],         actions_np)
    _stats(f"{quant_mode} vs fp32 post-unnorm", baseline["post_unnorm"], actions_un)
    print()
    print(f"Latency: fp32={float(baseline['latency_ms']):.0f}ms  {quant_mode}={dt*1000:.0f}ms  speedup={float(baseline['latency_ms'])/(dt*1000):.2f}x")


if __name__ == "__main__":
    main()
