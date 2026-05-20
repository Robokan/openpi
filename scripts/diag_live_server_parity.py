#!/usr/bin/env python3
"""End-to-end parity + latency test against the live PT websocket server.

Loads the same OpenArm observation we used in `diag_quant_parity.py`,
sends it to ws://localhost:8002, measures round-trip latency, and compares
the returned actions to the cached fp32 baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

BASELINE_PATH = "/tmp/diag_quant_fp32_baseline.npz"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    af, bf = a.flatten(), b.flatten()
    if af.shape != bf.shape:
        print(f"  [{name}] SHAPE MISMATCH a={a.shape} b={b.shape}")
        return
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    print(f"  [{name:40s}] cos={cos:+.6f}  |a|={na:>9.3f}  |b|={nb:>9.3f}  ratio={nb/(na+1e-30):.4f}  rel_err={np.linalg.norm(a-b)/(na+1e-30):.4e}")


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
    for cam_in, cam_out in (("ego", "cam_high"),
                             ("left_wrist", "cam_left_wrist"),
                             ("right_wrist", "cam_right_wrist")):
        vp = ds / "videos" / chunk / f"observation.images.{cam_in}" / f"{ep}.mp4"
        with av.open(str(vp)) as container:
            frame_np = None
            for i, frame in enumerate(container.decode(container.streams.video[0])):
                if i == fidx:
                    frame_np = frame.to_ndarray(format="rgb24"); break
        if frame_np.shape[:2] != (224, 224):
            frame_np = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_AREA)
        # OpenArmInputs expects (C, H, W) per its convert_image rearrange.
        images[cam_out] = np.transpose(frame_np.astype(np.uint8), (2, 0, 1))
    return {
        "state": state,
        "images": images,
        "prompt": prompt,
    }


def main():
    from openpi_client import websocket_client_policy
    print("=" * 100)
    print("Live PT server (ws://localhost:8002) parity + latency test")
    print("=" * 100)
    obs = _load_obs()
    print(f"prompt = {obs['prompt']!r}")
    print(f"state shape = {obs['state'].shape}, image shape = {obs['images']['cam_high'].shape}")

    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8002)
    print(f"server_metadata = {client.get_server_metadata()}")

    print("\nWarmup call (first call triggers torch.compile autotune, may take 30-90s)...")
    t0 = time.time()
    _ = client.infer(obs)
    print(f"  warmup latency: {(time.time()-t0)*1000:.0f} ms")

    print("\nMeasuring (3 calls)...")
    times = []
    last_actions = None
    for _ in range(3):
        t0 = time.time()
        result = client.infer(obs)
        times.append(time.time() - t0)
        last_actions = result["actions"]
    actions = np.asarray(last_actions)
    print(f"  per-call (ms): {[round(t*1000) for t in times]}  median={np.median(times)*1000:.0f}")

    print(f"\nReturned actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"  range: [{actions.min():.3f}, {actions.max():.3f}], mean: {actions.mean():.3f}")

    # Compare to cached baseline (post-unnormalized, since the server returns unnormalized actions)
    if Path(BASELINE_PATH).exists():
        baseline = np.load(BASELINE_PATH)
        print(f"\n--- Parity vs cached fp32 baseline (which used fp32_attention) ---")
        # baseline['post_unnorm'] is (50,16). actions may be (50,16) or other.
        _stats("live server vs cached fp32 baseline", baseline["post_unnorm"], actions)
        print(f"\nLatency: cached fp32 (in-process)={float(baseline['latency_ms']):.0f}ms  live-server end-to-end={np.median(times)*1000:.0f}ms")
    else:
        print(f"\nNo cached baseline at {BASELINE_PATH}; just printing actions.")


if __name__ == "__main__":
    main()
