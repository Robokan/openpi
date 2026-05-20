#!/usr/bin/env python3
"""End-to-end RTC test against the live PT WebSocket server.

Sequence:
  1) Call server with a normal obs -> get chunk_0. Extract the model-space
     chunk that the new Policy returns in `_rtc_chunk_model_space`.
  2) Call server again with the same obs PLUS:
       _rtc_prev_chunk = chunk_0_model_space
       _rtc_inference_delay = 4
       _rtc_prefix_attention_horizon = 25
     The server should dispatch to realtime_sample_actions and return a chunk
     whose model-space prefix matches chunk_0_model_space at indices [0:4].

Run from the host (outside the container) once the openpi-pt-server is up:
    python scripts/diag_rtc_live.py
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/evaughan/sparkpack/openpi/packages/openpi-client/src")

from openpi_client import websocket_client_policy as _wcp  # noqa: E402

HOST = "127.0.0.1"
PORT = 8002
# Inside the container the HF cache is mapped to /root/.cache. From the host
# it lives under /home/evaughan/.cache. We pick whichever exists.
_DATA_CANDIDATES = [
    Path("/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4"),
    Path("/home/evaughan/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4"),
]
DATA = next(p for p in _DATA_CANDIDATES if p.exists())


def _load_obs(row: int = 100) -> dict:
    import av  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    parquet = sorted(DATA.glob("data/chunk-*/episode_*.parquet"))[0]
    table = pq.read_table(parquet)
    state = np.asarray(table.column("observation.state")[row].as_py(), dtype=np.float32)
    fidx = int(table.column("frame_index")[row].as_py())
    tidx = int(table.column("task_index")[row].as_py())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in (DATA / "meta" / "tasks.jsonl").read_text().splitlines()}
    prompt = tasks.get(tidx, "do the task")
    chunk, ep = parquet.parent.name, parquet.stem
    images = {}
    for cam_in, cam_out in (("ego", "cam_high"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist")):
        vp = DATA / "videos" / chunk / f"observation.images.{cam_in}" / f"{ep}.mp4"
        with av.open(str(vp)) as container:
            stream = container.streams.video[0]
            frame_np = None
            for i, frame in enumerate(container.decode(stream)):
                if i == fidx:
                    frame_np = frame.to_ndarray(format="rgb24")
                    break
            if frame_np is None:
                raise RuntimeError(f"frame {fidx} not in {vp}")
        if frame_np.shape[:2] != (224, 224):
            frame_np = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame_np, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def main():
    print(f"connecting to ws://{HOST}:{PORT} ...")
    client = _wcp.WebsocketClientPolicy(host=HOST, port=PORT)

    obs = _load_obs(row=100)

    print("\n[warmup] vanilla call to trigger torch.compile autotune ...")
    t_warm = _time.perf_counter()
    _ = client.infer(dict(obs))
    print(f"   warmup took {(_time.perf_counter() - t_warm)*1000:.0f} ms (autotune)")

    print("\n[bench-vanilla] 3 steady-state vanilla calls ...")
    for k in range(3):
        t0 = _time.perf_counter()
        rk = client.infer(dict(obs))
        print(f"   vanilla #{k+1}: rtt={(_time.perf_counter() - t0)*1000:.0f} ms"
              f"  server_infer={rk.get('server_timing', {}).get('infer_ms', 0):.0f} ms")

    print("\n[1/3] First call (vanilla, no RTC) ...")
    t0 = _time.perf_counter()
    r0 = client.infer(dict(obs))
    t1 = _time.perf_counter()
    print(f"   round-trip: {(t1 - t0) * 1000:.0f} ms"
          f"  server_infer={r0.get('server_timing', {}).get('infer_ms', 0):.0f} ms")
    actions_0 = np.asarray(r0["actions"], dtype=np.float32)
    model_0 = np.asarray(r0.get("_rtc_chunk_model_space"), dtype=np.float32) if r0.get("_rtc_chunk_model_space") is not None else None
    print(f"   actions shape={actions_0.shape}  |actions|={np.linalg.norm(actions_0):.2f}")
    if model_0 is None:
        print("   [FAIL] server did not return '_rtc_chunk_model_space' -- old code path?")
        sys.exit(1)
    print(f"   model-space chunk shape={model_0.shape}  |chunk|={np.linalg.norm(model_0):.2f}")

    print("\n[2/3] Second call with RTC (prev=model_0, d=4, pah=25, exp) ...")
    t0 = _time.perf_counter()
    obs_rtc = dict(obs)
    obs_rtc["_rtc_prev_chunk"] = model_0
    obs_rtc["_rtc_inference_delay"] = 4
    obs_rtc["_rtc_prefix_attention_horizon"] = 25
    obs_rtc["_rtc_schedule"] = "exp"
    obs_rtc["_rtc_max_guidance_weight"] = 5.0
    r1 = client.infer(obs_rtc)
    t1 = _time.perf_counter()
    print(f"   round-trip: {(t1 - t0) * 1000:.0f} ms"
          f"  server_infer={r1.get('server_timing', {}).get('infer_ms', 0):.0f} ms")

    print("\n[bench-rtc] 3 steady-state RTC calls ...")
    for k in range(3):
        t0 = _time.perf_counter()
        rk = client.infer(dict(obs_rtc))
        print(f"   rtc #{k+1}: rtt={(_time.perf_counter() - t0)*1000:.0f} ms"
              f"  server_infer={rk.get('server_timing', {}).get('infer_ms', 0):.0f} ms")
    actions_1 = np.asarray(r1["actions"], dtype=np.float32)
    model_1 = np.asarray(r1["_rtc_chunk_model_space"], dtype=np.float32)
    rtc_used = r1.get("_rtc_used", False)
    print(f"   _rtc_used={rtc_used}  _rtc_inference_delay={r1.get('_rtc_inference_delay')}")
    print(f"   actions shape={actions_1.shape}  |actions|={np.linalg.norm(actions_1):.2f}")

    if not rtc_used:
        print("   [FAIL] server did NOT route to RTC path despite prev chunk being set")
        sys.exit(2)

    print("\n[3/3] Validate RTC behavior in model space:")
    # Frozen prefix [0:4]: should ~ match model_0[0:4]
    frozen_diff = np.linalg.norm(model_1[:4] - model_0[:4])
    frozen_ref = np.linalg.norm(model_0[:4])
    frozen_rel = frozen_diff / (frozen_ref + 1e-9)
    print(f"   frozen prefix [0:4]:  rel_err vs prev = {frozen_rel:.4e}  (~0 = correct)")
    # Free tail [25:50]: free to differ (since noise is regenerated server-side)
    free_diff = np.linalg.norm(model_1[25:] - model_0[25:])
    print(f"   free tail   [25:50]: rel_err vs prev = {free_diff/(np.linalg.norm(model_0[25:])+1e-9):.4e}  (can be anything)")
    # Seam discontinuity at index 4 (last frozen vs first decay)
    seam = np.abs(model_1[4] - model_1[3]).max()
    print(f"   max seam delta (idx 3 -> 4): {seam:.4e}")

    print()
    if frozen_rel < 0.1:
        print(f"   [OK] frozen prefix rel_err = {frozen_rel:.2%} (< 10%) -- RTC inpainting working")
    else:
        print(f"   [WARN] frozen prefix rel_err = {frozen_rel:.2%} (>= 10%) -- weaker than expected")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
