#!/usr/bin/env python3
"""Smoke test a running openpi WebSocket policy server.

Connects to the given port, sends one (or N) real OpenArm observations,
and prints back the action shape, first-step values, and inference time.

Usage (inside openpi-thor container):
    python scripts/smoke_test_server.py --port 8002 --n 3
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

# Make openpi_client importable
sys.path.insert(0, "/app/packages/openpi-client/src")
from openpi_client import websocket_client_policy  # noqa: E402


CAMERAS = ["ego", "left_wrist", "right_wrist"]
CAMERA_MAP = {"ego": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}


def _load_tasks(p: Path) -> dict:
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


def load_samples(local_dir: Path, n: int) -> list:
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
            samples.append({"state": state, "images": imgs, "prompt": prompt})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--dataset", default="/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    args = p.parse_args()

    print(f"Loading {args.n} real samples from {args.dataset}")
    samples = load_samples(Path(args.dataset), args.n)
    if not samples:
        print("ERROR: no samples loaded")
        return 1

    print(f"Connecting to ws://{args.host}:{args.port}")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"  server metadata: {client.get_server_metadata()}")

    times = []
    for i, s in enumerate(samples):
        t0 = time.monotonic()
        try:
            result = client.infer(s)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            continue
        dt = time.monotonic() - t0
        actions = np.asarray(result["actions"])
        times.append(dt * 1000)
        print(f"  [{i}] prompt={s['prompt'][:50]!r}")
        print(f"        action shape={actions.shape}  range=[{actions.min():.3f},{actions.max():.3f}]")
        print(f"        first action (joints 0..15): {np.array2string(actions[0, :16], precision=3, suppress_small=True)}")
        print(f"        roundtrip={dt*1000:.1f}ms")
        srv_t = result.get("policy_timing", {}).get("infer_ms", -1)
        print(f"        server-reported infer={srv_t:.1f}ms")
    if times:
        print(f"\nRoundtrip stats (ms): min={min(times):.1f} median={sorted(times)[len(times)//2]:.1f} max={max(times):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
