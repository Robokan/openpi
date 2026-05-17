#!/usr/bin/env python3
"""Compare JAX and PyTorch policy outputs on the same real OpenArm observations.

Diagnoses whether the PyTorch path produces materially different actions
than the JAX path that we know works on the real robot.

Both models are loaded in the same process and called with:
  - the same observation dict (real frame from training data)
  - the same diffusion noise (numpy float32, generated from a fixed seed)
  - the same num_denoising_steps (10, the model default)

Usage (inside the openpi-thor container which has both JAX+CUDA and PyTorch+CUDA):
    python scripts/compare_jax_pytorch.py --n 5
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Patch load_pytorch: the LoRA-converted checkpoints have extra keys we need to
# load non-strictly. This is the same patch used by serve_policy_trt.py.
# ---------------------------------------------------------------------------
def _patch_load_pytorch():
    import safetensors.torch as _st
    from openpi.models import model as _model_mod
    from openpi.models_pytorch import pi0_pytorch as _pi0pt

    def _load(self, train_config, weight_path):
        model = _pi0pt.PI0Pytorch(config=train_config.model)
        sd = _st.load_file(weight_path)
        model.load_state_dict(sd, strict=False)
        return model

    for cls in vars(_model_mod).values():
        if isinstance(cls, type) and hasattr(cls, "load_pytorch"):
            cls.load_pytorch = _load


_patch_load_pytorch()

from openpi.policies import policy_config as _pc  # noqa: E402
from openpi.training import config as _config  # noqa: E402


# ---------------------------------------------------------------------------
# Real-data sample loading (parquet state/action + ffmpeg-extracted MP4 frames)
# ---------------------------------------------------------------------------
CAMERAS = ["ego", "left_wrist", "right_wrist"]
CAMERA_MAP = {
    "ego": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def _load_tasks(tasks_path: Path) -> dict:
    if not tasks_path.exists():
        return {}
    mp = {}
    with open(tasks_path) as f:
        for line in f:
            e = json.loads(line.strip())
            mp[e["task_index"]] = e["task"]
    return mp


def _ffmpeg_frame(video_path: Path, frame_idx: int):
    tname = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tname = tmp.name
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-vframes", "1", "-f", "image2", tname,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0 or not os.path.exists(tname):
            return None
        frame = cv2.imread(tname)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        if tname and os.path.exists(tname):
            try:
                os.unlink(tname)
            except OSError:
                pass


def load_real_samples(local_dir: Path, n: int) -> list:
    """Load `n` real {state, images, prompt} samples from the OpenArm training dataset."""
    tasks = _load_tasks(local_dir / "meta" / "tasks.jsonl")
    parquets = sorted(local_dir.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets found in {local_dir}/data/")
    print(f"Found {len(parquets)} episode files in {local_dir}")

    samples = []
    step = max(1, len(parquets) // max(n, 1))
    for pq_path in parquets[::step]:
        if len(samples) >= n:
            break
        table = pq.read_table(pq_path)
        if table.num_rows == 0:
            continue
        # Pick a frame in the middle of the episode (more interesting than the start).
        row = table.num_rows // 2
        state = np.array(table.column("observation.state")[row].as_py(), dtype=np.float32)
        fidx = int(table.column("frame_index")[row].as_py())
        tidx = int(table.column("task_index")[row].as_py())
        prompt = tasks.get(tidx, "pick up the object")

        chunk = pq_path.parent.name
        ep = pq_path.stem
        images = {}
        ok = True
        for cam in CAMERAS:
            vp = local_dir / "videos" / chunk / f"observation.images.{cam}" / f"{ep}.mp4"
            if not vp.exists():
                ok = False
                break
            fr = _ffmpeg_frame(vp, fidx)
            if fr is None:
                ok = False
                break
            if fr.shape[:2] != (224, 224):
                fr = cv2.resize(fr, (224, 224), interpolation=cv2.INTER_AREA)
            fr = np.transpose(fr, (2, 0, 1)).astype(np.uint8)
            images[CAMERA_MAP[cam]] = fr
        if ok and len(images) == 3:
            samples.append({"state": state, "images": images, "prompt": prompt})
            print(f"  [{len(samples):2d}] ep={ep} frame={fidx} prompt={prompt!r}")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_openarm_ngc_lora_v4")
    p.add_argument(
        "--jax-ckpt",
        default="/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999",
    )
    p.add_argument(
        "--pytorch-ckpt",
        default="/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch",
    )
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="", help="Optional .npz path to dump all actions for offline analysis")
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    A_H = cfg.model.action_horizon
    A_D = cfg.model.action_dim
    print(f"Config={args.config}  action_horizon={A_H}  action_dim={A_D}")

    local_dir = Path(cfg.data.base_config.local_dir)
    print(f"Dataset: {local_dir}")
    samples = load_real_samples(local_dir, args.n)
    if not samples:
        print("ERROR: no real samples loaded")
        return 1

    rng = np.random.default_rng(args.seed)
    noises = [rng.standard_normal((A_H, A_D)).astype(np.float32) for _ in samples]

    # ----------------- JAX -----------------
    print("\n=== Loading JAX policy ===")
    print(f"  ckpt: {args.jax_ckpt}")
    jax_policy = _pc.create_trained_policy(cfg, args.jax_ckpt)
    print("  -> JAX policy loaded; running inference (first call will JIT compile)")
    jax_actions = []
    for i, (s, ns) in enumerate(zip(samples, noises)):
        result = jax_policy.infer(s, noise=ns)
        a = np.asarray(result["actions"])
        jax_actions.append(a)
        ms = result.get("policy_timing", {}).get("infer_ms", -1)
        print(f"  JAX [{i}] shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] infer={ms:.1f}ms")

    del jax_policy
    try:
        import jax as _jax
        _jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    try:
        import torch as _torch
        _torch.cuda.empty_cache()
    except Exception:
        pass

    # ----------------- PyTorch -----------------
    print("\n=== Loading PyTorch policy ===")
    print(f"  ckpt: {args.pytorch_ckpt}")
    pt_policy = _pc.create_trained_policy(cfg, args.pytorch_ckpt)
    print("  -> PyTorch policy loaded; running inference")
    pt_actions = []
    for i, (s, ns) in enumerate(zip(samples, noises)):
        result = pt_policy.infer(s, noise=ns)
        a = np.asarray(result["actions"])
        pt_actions.append(a)
        ms = result.get("policy_timing", {}).get("infer_ms", -1)
        print(f"  PT  [{i}] shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] infer={ms:.1f}ms")

    # ----------------- Compare -----------------
    print("\n=== JAX vs PyTorch  ===")
    print(f"Both policies run on the SAME observation and SAME float32 noise.")
    print(f"Action tensor: shape={jax_actions[0].shape}  (action_horizon, action_dim)")
    print(f"Robot uses first {16} dims of each timestep (16-DOF OpenArm).")

    all_diffs = []
    all_first_diffs = []
    cos_overall = []
    for i, (aj, ap) in enumerate(zip(jax_actions, pt_actions)):
        d = np.abs(aj - ap)
        first = d[0]                                  # diff at the first action (the one actually sent)
        per_joint_worst = d.max(axis=0)              # worst diff over the 50-step horizon, per dim
        jf = aj.reshape(-1).astype(np.float64)
        pf = ap.reshape(-1).astype(np.float64)
        cos = float(np.dot(jf, pf) / (np.linalg.norm(jf) * np.linalg.norm(pf) + 1e-12))
        cos_overall.append(cos)
        all_diffs.append(d)
        all_first_diffs.append(first)

        print(f"\n--- Sample {i}   prompt={samples[i]['prompt'][:70]!r}")
        print(f"   max|diff|={d.max():.4f}   mean|diff|={d.mean():.4f}   cosine={cos:.6f}")
        print(f"   FIRST step abs diff (robot joints 0..15):")
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print("     " + np.array2string(first[:16]))
        print(f"   WORST-over-horizon abs diff (robot joints 0..15):")
        print("     " + np.array2string(per_joint_worst[:16]))
        print(f"   JAX first action (joints 0..15):")
        print("     " + np.array2string(aj[0, :16]))
        print(f"   PT  first action (joints 0..15):")
        print("     " + np.array2string(ap[0, :16]))
        print(f"   diff/JAX_range first step (relative, joints 0..15):")
        jrange = max(1e-6, float(np.abs(aj[0, :16]).max()))
        print("     " + np.array2string(first[:16] / jrange))

    # Aggregate
    diffs = np.concatenate(all_diffs, axis=0)
    first_diffs = np.stack(all_first_diffs, axis=0)
    print("\n=== AGGREGATE ===")
    print(f"  Samples: {len(samples)}  Total positions compared: {diffs.size}")
    print(f"  Cosine similarity (overall):   mean={np.mean(cos_overall):.6f}  min={np.min(cos_overall):.6f}")
    print(f"  Overall:   max|diff|={diffs.max():.4f}  mean={diffs.mean():.4f}  P99={np.percentile(diffs, 99):.4f}")
    print(f"  First step (joints 0..15):   max|diff|={first_diffs[:, :16].max():.4f}  mean={first_diffs[:, :16].mean():.4f}")
    print()
    print("  Interpretation:")
    print("    - If cosine ~= 1.0 and max|diff| < 0.05 rad: PyTorch matches JAX.")
    print("    - If cosine < 0.95 or max|diff| > 0.2 rad on the first step:")
    print("      the JAX->PyTorch conversion is degrading the model and the bug")
    print("      is in the conversion, not in TRT.")
    print()

    if args.save:
        np.savez(
            args.save,
            jax=np.stack(jax_actions, axis=0),
            pt=np.stack(pt_actions, axis=0),
            noises=np.stack(noises, axis=0),
            prompts=np.array([s["prompt"] for s in samples]),
        )
        print(f"Saved raw actions to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
