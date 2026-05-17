#!/usr/bin/env python3
"""Compare two running policy servers (JAX vs PyTorch) on the SAME observation
and report deltas relative to the current state — not just absolute action values.

The purpose is to diagnose cases where both servers produce similar-looking
absolute actions (high cosine sim) but very different motion commands relative
to the current state.

Usage:
    python scripts/compare_servers_live.py --jax-port 8001 --pt-port 8002 --n 3
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/app/packages/openpi-client/src")
from openpi_client import websocket_client_policy  # noqa: E402


CAMERAS = ["ego", "left_wrist", "right_wrist"]
CAMERA_MAP = {"ego": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}


def _load_tasks(p: Path):
    m = {}
    if p.exists():
        with open(p) as f:
            for line in f:
                e = json.loads(line.strip())
                m[e["task_index"]] = e["task"]
    return m


def _ffmpeg_frame(vp: Path, idx: int):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        name = tmp.name
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp),
               "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-f", "image2", name]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0:
            return None
        f = cv2.imread(name)
        return None if f is None else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    finally:
        if os.path.exists(name):
            try:
                os.unlink(name)
            except OSError:
                pass


def load_samples(local_dir: Path, n: int):
    tasks = _load_tasks(local_dir / "meta" / "tasks.jsonl")
    parquets = sorted(local_dir.glob("data/chunk-*/episode_*.parquet"))
    samples = []
    step = max(1, len(parquets) // max(n, 1))
    for pq_path in parquets[::step]:
        if len(samples) >= n:
            break
        t = pq.read_table(pq_path)
        if t.num_rows == 0:
            continue
        row = t.num_rows // 2
        state = np.array(t.column("observation.state")[row].as_py(), dtype=np.float32)
        # Also pull the GROUND-TRUTH action from the dataset for this frame so we have a reference
        gt_action = np.array(t.column("action")[row].as_py(), dtype=np.float32) if "action" in t.column_names else None
        fidx = int(t.column("frame_index")[row].as_py())
        tidx = int(t.column("task_index")[row].as_py())
        prompt = tasks.get(tidx, "pick up the object")
        chunk, ep = pq_path.parent.name, pq_path.stem
        imgs, ok = {}, True
        for cam in CAMERAS:
            vp = local_dir / "videos" / chunk / f"observation.images.{cam}" / f"{ep}.mp4"
            if not vp.exists():
                ok = False
                break
            f = _ffmpeg_frame(vp, fidx)
            if f is None:
                ok = False
                break
            if f.shape[:2] != (224, 224):
                f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
            imgs[CAMERA_MAP[cam]] = np.transpose(f, (2, 0, 1)).astype(np.uint8)
        if ok:
            samples.append({
                "obs": {"state": state, "images": imgs, "prompt": prompt},
                "gt_action": gt_action,
                "episode": ep,
            })
    return samples


def average_actions(client, obs, k):
    """Call infer k times and average to wash out per-call diffusion noise.
    Returns (mean_actions, last_full_action_chunk)."""
    chunks = []
    for _ in range(k):
        r = client.infer(obs)
        chunks.append(np.asarray(r["actions"]))
    chunks = np.stack(chunks, axis=0)  # (k, H, D)
    return chunks.mean(axis=0), chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--jax-port", type=int, default=8001)
    p.add_argument("--pt-port", type=int, default=8002)
    p.add_argument("--n", type=int, default=2, help="number of distinct observations")
    p.add_argument("--reps", type=int, default=5, help="diffusion samples averaged per obs per server")
    p.add_argument("--dataset", default="/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    args = p.parse_args()

    print(f"Loading {args.n} real samples from {args.dataset}")
    samples = load_samples(Path(args.dataset), args.n)
    if not samples:
        print("ERROR: no samples loaded")
        return 1

    print(f"Connecting to JAX server ws://{args.host}:{args.jax_port}")
    jax_cli = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.jax_port)
    print(f"Connecting to PyTorch server ws://{args.host}:{args.pt_port}")
    pt_cli = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.pt_port)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    for i, s in enumerate(samples):
        obs = s["obs"]
        state = obs["state"]
        print(f"\n========== Sample {i}  episode={s['episode']}  prompt={obs['prompt']!r}")
        print(f"  state (joints 0..15): {state[:16]}")
        if s["gt_action"] is not None:
            gt = s["gt_action"]
            print(f"  GT next action (from dataset): {gt[:16]}")
            print(f"  GT delta (gt - state):         {(gt[:16] - state[:16])}")

        # JAX
        t0 = time.monotonic()
        jax_mean, _ = average_actions(jax_cli, obs, args.reps)
        jax_t = (time.monotonic() - t0) / args.reps * 1000

        # PyTorch
        t0 = time.monotonic()
        pt_mean, _ = average_actions(pt_cli, obs, args.reps)
        pt_t = (time.monotonic() - t0) / args.reps * 1000

        # First-step actions
        jax_a = jax_mean[0]
        pt_a = pt_mean[0]
        jax_delta = jax_a[:16] - state[:16]
        pt_delta = pt_a[:16] - state[:16]

        print(f"\n  JAX first action  (avg of {args.reps}):  {jax_a[:16]}    [{jax_t:.0f}ms/call]")
        print(f"  PT  first action  (avg of {args.reps}):  {pt_a[:16]}    [{pt_t:.0f}ms/call]")
        print(f"\n  JAX delta from state:                  {jax_delta}")
        print(f"  PT  delta from state:                  {pt_delta}")
        print(f"\n  ||JAX_delta||={np.linalg.norm(jax_delta):.4f}   max|JAX_delta|={np.abs(jax_delta).max():.4f}")
        print(f"  ||PT_delta||= {np.linalg.norm(pt_delta):.4f}    max|PT_delta|= {np.abs(pt_delta).max():.4f}")
        ratio = np.linalg.norm(pt_delta) / max(1e-6, np.linalg.norm(jax_delta))
        print(f"  PT motion magnitude / JAX motion magnitude: {ratio:.3f}")
        if ratio < 0.5:
            print(f"  >>> PyTorch is producing a MUCH SMALLER motion command than JAX (the robot won't move)")
        elif ratio > 2.0:
            print(f"  >>> PyTorch is producing a MUCH LARGER motion command than JAX")

        # Also compare absolute-action cosine
        ja = jax_a[:16].astype(np.float64)
        pa = pt_a[:16].astype(np.float64)
        cos_abs = float(np.dot(ja, pa) / (np.linalg.norm(ja) * np.linalg.norm(pa) + 1e-12))
        jd = jax_delta.astype(np.float64)
        pd = pt_delta.astype(np.float64)
        cos_delta = float(np.dot(jd, pd) / (np.linalg.norm(jd) * np.linalg.norm(pd) + 1e-12))
        print(f"  cos(JAX_abs, PT_abs)   = {cos_abs:.6f}")
        print(f"  cos(JAX_delta, PT_delta) = {cos_delta:.6f}    <- direction of intended motion agreement")

        # Look further out in the horizon to see if motion ramps up
        for h in (10, 25, 49):
            jd_h = jax_mean[h, :16] - state[:16]
            pd_h = pt_mean[h, :16] - state[:16]
            print(f"  horizon[{h:2d}]:  ||JAX_delta||={np.linalg.norm(jd_h):.3f}  ||PT_delta||={np.linalg.norm(pd_h):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
