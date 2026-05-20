#!/usr/bin/env python3
"""Exercise AsyncActionChunkBroker against the live PT WebSocket server.

Simulates a robot polling at 103 Hz (the OpenArm control rate) for N seconds.
For each call we record the action returned. Logs:
- which chunk each action came from (initial / chunk_1 / chunk_2 / ...)
- the seam between chunks
- whether RTC was used
- pacing (time between consecutive calls)

Run after starting the server with the new RTC-aware policy:
    docker exec openpi-pt-server bash -lc 'cd /app && \
        PYTHONPATH=/app/src:/app/packages/openpi-client/src \
        python scripts/diag_async_broker.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/packages/openpi-client/src")
sys.path.insert(0, "/app/src")
sys.path.insert(0, str(Path(__file__).parent))

from openpi_client import async_action_chunk_broker as _bk  # noqa: E402
from openpi_client import websocket_client_policy as _wcp  # noqa: E402
from diag_rtc_live import _load_obs  # noqa: E402

HOST = "127.0.0.1"
PORT = 8002
CONTROL_HZ = 103.0
RUN_SECONDS = 5.0  # ~515 steps at 103 Hz
EXEC_HORIZON = 25
INFERENCE_DELAY = 18  # ~= 175ms vanilla / 9.7ms control period


def _bench(broker, label, obs, n_steps, period):
    actions = []
    timestamps = []
    print(f"\n[{label}] running {n_steps} steps at {1/period:.0f}Hz (period={period*1000:.1f}ms) ...")
    t_start = time.perf_counter()
    next_tick = t_start
    for i in range(n_steps):
        now = time.perf_counter()
        sleep = next_tick - now
        if sleep > 0:
            time.sleep(sleep)
        t0 = time.perf_counter()
        result = broker.infer(dict(obs))
        latency_ms = (time.perf_counter() - t0) * 1000
        actions.append(np.asarray(result.get("actions"), dtype=np.float32))
        timestamps.append((t0 - t_start, latency_ms))
        next_tick += period

    A = np.stack(actions, axis=0)
    total_dur = time.perf_counter() - t_start
    block_calls = [l for _, l in timestamps if l > 5.0]
    deltas = np.linalg.norm(np.diff(A, axis=0), axis=1)
    p99 = np.percentile(deltas, 99)
    print(f"  [{label}] duration={total_dur:.2f}s measured_hz={len(A)/total_dur:.1f}"
          f"  block_calls={len(block_calls)} (median={np.median(block_calls) if block_calls else 0:.0f}ms"
          f"  max={max(block_calls) if block_calls else 0:.0f}ms)")
    print(f"  [{label}] action delta: median={np.median(deltas):.4f}"
          f"  p99={p99:.4f}  max={deltas.max():.4f}")
    print(f"  [{label}] seam indices: {sorted(np.argsort(deltas)[-5:])}"
          f"  values: {[f'{deltas[i]:.3f}' for i in sorted(np.argsort(deltas)[-5:])]}")
    return A, timestamps, deltas


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"connecting to ws://{HOST}:{PORT} ...")
    obs = _load_obs(row=100)
    period = 1.0 / CONTROL_HZ
    n_steps = int(RUN_SECONDS * CONTROL_HZ)

    # First: warm up server compile via one direct call (so neither broker
    # variant pays the autotune cost).
    direct = _wcp.WebsocketClientPolicy(host=HOST, port=PORT)
    t0 = time.perf_counter()
    _ = direct.infer(dict(obs))
    print(f"  warmup direct call: {(time.perf_counter()-t0)*1000:.0f} ms")
    t0 = time.perf_counter()
    _ = direct.infer(dict(obs))
    print(f"  warm direct call:   {(time.perf_counter()-t0)*1000:.0f} ms")

    # ---------- TEST 1: async pipelining ONLY (no RTC) ----------
    base1 = _wcp.WebsocketClientPolicy(host=HOST, port=PORT)
    broker1 = _bk.AsyncActionChunkBroker(
        base1,
        action_horizon=50,
        execute_horizon=EXEC_HORIZON,
        inference_delay=INFERENCE_DELAY,
        enable_rtc=False,
    )
    A1, ts1, d1 = _bench(broker1, "no-RTC pipelining", obs, n_steps, period)
    broker1.close()

    # ---------- TEST 2: full RTC ----------
    base2 = _wcp.WebsocketClientPolicy(host=HOST, port=PORT)
    broker = _bk.AsyncActionChunkBroker(
        base2,
        action_horizon=50,
        execute_horizon=EXEC_HORIZON,
        inference_delay=INFERENCE_DELAY,
        prefix_attention_horizon=50 - EXEC_HORIZON,
        schedule="exp",
        max_guidance_weight=5.0,
        enable_rtc=True,
    )
    A2, ts2, d2 = _bench(broker, "full RTC", obs, n_steps, period)
    broker.close()

    # ---------- Comparison ----------
    print("\n[SUMMARY]")
    print(f"  no-RTC pipelining: median_delta={np.median(d1):.4f}  max_delta={d1.max():.4f}")
    print(f"  full RTC:          median_delta={np.median(d2):.4f}  max_delta={d2.max():.4f}")
    print(f"  Lower median_delta = smoother motion.  Lower max_delta = smaller seam jumps.")


if __name__ == "__main__":
    main()
