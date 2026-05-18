#!/usr/bin/env python3
"""Measure where time is spent in the PyTorch policy inference.

Run inside the docker container as:
  python scripts/diag_pt_speed.py
"""
import sys
import time

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import torch

from openpi.policies import policy_config as _pc
from openpi.training import config as _config


def main():
    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    ckpt = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"
    pt_policy = _pc.create_trained_policy(cfg, ckpt)
    obs = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {
            "cam_high":        np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist":  np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "prompt": "put the chocolate bars in the bin",
    }
    print("warming up (2 calls, may take >1min due to torch.compile)...")
    for _ in range(2):
        pt_policy.infer(obs)
    print("warmup done")

    torch.cuda.synchronize()
    for trial in range(3):
        t0 = time.monotonic()
        r = pt_policy.infer(obs)
        torch.cuda.synchronize()
        total_ms = (time.monotonic() - t0) * 1000
        timing = r.get("policy_timing", {})
        print(f"trial {trial}: total={total_ms:.0f}ms  policy_timing={timing}")

    # Now time the model.sample_actions directly (no python overhead from policy wrapper).
    print("\n--- direct model.sample_actions timing ---")
    model = pt_policy._policy._model  # the PI0Pytorch
    device = next(model.parameters()).device

    # Build a minimal Observation-like input directly via _input_transform
    # Easier: use the policy.infer path but disable transforms.
    # Just call sample_actions with the same observation object the policy used internally.
    # Reach into transform output by reusing policy.infer's internal preprocessing:
    # The policy stores transforms; the easiest path is to time policy.infer minus
    # the transforms-only overhead. We'll measure the gap via tracing.
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(2):
        pt_policy.infer(obs)
    pr.disable()
    s = pstats.Stats(pr).sort_stats("cumulative")
    s.print_stats(25)


if __name__ == "__main__":
    main()
