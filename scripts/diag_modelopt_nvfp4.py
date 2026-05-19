#!/usr/bin/env python3
"""Test whether modelopt NVFP4 with AWQ-lite calibration fixes the
6.4% magnitude shrinkage seen with torchao's NVFP4InferenceConfig.

Runs the same forward as `diag_quant_parity.py` but uses modelopt's
`mtq.quantize(..., NVFP4_AWQ_LITE_CFG, forward_loop=...)` after loading
the fp32 model, then compares to the cached fp32 baseline.
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
BASELINE_PATH = "/tmp/diag_quant_fp32_baseline.npz"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    print(f"  [{name:40s}] cos={cos:+.6f}  ratio={nb/(na+1e-30):.4f}  rel_err={np.linalg.norm(a-b)/(na+1e-30):.4e}")


def _load_obs(row: int = 100):
    import cv2
    import pyarrow.parquet as pq
    import av
    ds = Path("/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parquet = sorted(ds.glob("data/chunk-*/episode_*.parquet"))[0]
    table = pq.read_table(parquet)
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


def main():
    print("=" * 100)
    print("modelopt NVFP4 AWQ-lite + calibration parity test")
    print("=" * 100)

    # IMPORTANT: do NOT set OPENPI_PT_QUANT so policy_config doesn't apply torchao.
    os.environ.pop("OPENPI_PT_QUANT", None)
    os.environ["OPENPI_PYTORCH_PRECISION"] = "bfloat16"

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import torch
    import jax
    import modelopt.torch.quantization as mtq

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    print("Loading PT policy (bf16, no quant)...")
    t0 = time.time()
    policy = _pc.create_trained_policy(cfg, CKPT_PT)
    print(f"  loaded in {time.time()-t0:.1f}s")
    pt_model = policy._model
    pt_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build N calibration samples spanning different frames.
    print("Loading calibration samples...")
    calib_obs_list = []
    for row in (50, 150, 250, 350, 450, 550, 650, 750):
        try:
            raw = _load_obs(row=row)
            inputs = policy._input_transform(raw)
            pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to(device), inputs)
            pt_dtype = pt_model.action_in_proj.weight.dtype
            pt_inputs = jax.tree.map(lambda t: t.to(pt_dtype) if t.is_floating_point() else t, pt_inputs)
            obs = _model.Observation.from_dict(pt_inputs)
            calib_obs_list.append((obs, pt_dtype, inputs))
        except Exception as e:
            print(f"  skip row {row}: {e}")
    print(f"  collected {len(calib_obs_list)} calibration samples")

    horizon = pt_model.config.action_horizon
    adim = pt_model.config.action_dim
    np.random.seed(42)
    base_noise = torch.from_numpy(np.random.randn(1, horizon, adim).astype(np.float32)).to(device)

    def forward_loop(mdl):
        with torch.no_grad():
            for obs, dt, _ in calib_obs_list:
                noise = base_noise.to(dt)
                _ = mdl.sample_actions(device, obs, noise=noise, num_steps=4)

    print("Quantizing with modelopt NVFP4_AWQ_LITE_CFG (default targets)...")
    cfg_q = mtq.NVFP4_AWQ_LITE_CFG
    # The default config already excludes lm_head, proj_out, MoE routers, etc.
    # Conv2d (SigLIP patch embed) is auto-disabled below.
    cfg_q["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    t0 = time.time()
    mtq.quantize(pt_model, cfg_q, forward_loop=forward_loop)
    print(f"  quantized + calibrated in {time.time()-t0:.1f}s")

    print("\nMeasuring inference on baseline sample (row=100)...")
    raw = _load_obs(row=100)
    inputs = policy._input_transform(raw)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to(device), inputs)
    pt_dtype = pt_model.action_in_proj.weight.dtype
    pt_inputs = jax.tree.map(lambda t: t.to(pt_dtype) if t.is_floating_point() else t, pt_inputs)
    obs = _model.Observation.from_dict(pt_inputs)
    noise = base_noise.to(pt_dtype)

    with torch.no_grad():
        for _ in range(1):
            _ = pt_model.sample_actions(device, obs, noise=noise, num_steps=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        times = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            actions = pt_model.sample_actions(device, obs, noise=noise, num_steps=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    dt = float(np.median(times))
    actions_np = actions.detach().float().cpu().numpy()
    un = policy._output_transform({"state": np.asarray(inputs["state"]), "actions": actions_np[0]})
    actions_un = np.asarray(un["actions"])

    if not os.path.exists(BASELINE_PATH):
        print(f"  !! no baseline at {BASELINE_PATH}")
        return
    baseline = np.load(BASELINE_PATH)

    print("\n" + "=" * 100)
    print("Parity: modelopt NVFP4 (AWQ-lite, calibrated) vs fp32 baseline")
    print("=" * 100)
    _stats("nvfp4_awq vs fp32 raw",        baseline["raw"],         actions_np)
    _stats("nvfp4_awq vs fp32 post-unnorm", baseline["post_unnorm"], actions_un)
    print(f"\nLatency: fp32={float(baseline['latency_ms']):.0f}ms  nvfp4_awq={dt*1000:.0f}ms  speedup={float(baseline['latency_ms'])/(dt*1000):.2f}x")


if __name__ == "__main__":
    main()
