#!/usr/bin/env python3
"""Verify the rotary_emb.inv_freq fix is applied at all expected places."""
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")
import torch
from openpi.training import config as _config
from openpi.policies import policy_config as _pc

cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
pt = _pc.create_trained_policy(cfg, "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch", pytorch_device="cuda")
m = pt._model

for name, buf in m.named_buffers():
    if "inv_freq" in name:
        print(f"{name}: dtype={buf.dtype}, shape={tuple(buf.shape)}, [1]={buf[1].item():.7f}")

# Compare to fp32 expected value
import numpy as np
D = 256
expected_1 = float(1.0 / (10000.0 ** ((2.0 / D) * 1)))
print(f"\nExpected inv_freq[1] (fp32): {expected_1:.7f}")
print(f"BF16-truncated inv_freq[1]:  {float(torch.tensor(expected_1, dtype=torch.bfloat16).item()):.7f}")
