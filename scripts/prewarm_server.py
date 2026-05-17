#!/usr/bin/env python3
"""Prewarm a running OpenPI policy server by sending one inference with the
exact prompt the user will use. Triggers torch.compile / inductor autotune
so subsequent calls from real clients return quickly.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app/packages/openpi-client/src")
sys.path.insert(0, "/app/src")

import numpy as np  # noqa: E402

from openpi_client import websocket_client_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--prompt", default="put the chocolate bars in the bin")
    p.add_argument("--reps", type=int, default=2, help="number of warmup calls")
    args = p.parse_args()

    print(f"Connecting to ws://{args.host}:{args.port}  prompt={args.prompt!r}")
    cli = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    obs = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {
            "cam_high":        np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist":  np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "prompt": args.prompt,
    }
    for i in range(args.reps):
        t0 = time.monotonic()
        r = cli.infer(obs)
        dt = (time.monotonic() - t0) * 1000
        a = np.asarray(r["actions"])
        print(f"  warmup[{i}]: {dt:.0f}ms  action_shape={a.shape}  range=[{a.min():.3f},{a.max():.3f}]")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
