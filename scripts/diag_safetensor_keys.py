#!/usr/bin/env python3
"""Audit the JAX -> PyTorch checkpoint conversion key-by-key.

For every key in the converted safetensors file:
  - Is it present in PI0Pytorch.state_dict()?
  - If yes: are the dtypes / shapes equal?
  - If no: what JAX key did it come from? (potentially dropped weight)

For every key in PI0Pytorch.state_dict():
  - Is it in the safetensors? If not, the model layer is using its
    fresh random init at inference time, which is the strongest
    possible cause of behavioural divergence.

We do this for both the LoRA-merged and the BROKEN_NO_LORA checkpoints.
"""

import sys
from pathlib import Path

import safetensors.torch as st
import torch

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

from openpi.training import config as _config


def build_model(cfg_name: str):
    cfg = _config.get_config(cfg_name)
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    model = PI0Pytorch(config=cfg.model)
    return model


def audit(ckpt_path: str, model):
    file_sd = st.load_file(ckpt_path)
    model_sd = model.state_dict()
    file_keys = set(file_sd.keys())
    model_keys = set(model_sd.keys())

    in_file_only = file_keys - model_keys
    in_model_only = model_keys - file_keys
    common = file_keys & model_keys

    print(f"  total keys in safetensors:           {len(file_keys)}")
    print(f"  total keys in PI0Pytorch.state_dict: {len(model_keys)}")
    print(f"  shared:                              {len(common)}")
    print(f"  in safetensors but not in model:     {len(in_file_only)}")
    print(f"  in model but not in safetensors:     {len(in_model_only)}")

    print("\n  --- keys in safetensors but NOT in model (silently dropped) ---")
    for k in sorted(in_file_only)[:50]:
        v = file_sd[k]
        print(f"    DROPPED  {k}  shape={tuple(v.shape)}  dtype={v.dtype}")
    if len(in_file_only) > 50:
        print(f"    ... ({len(in_file_only) - 50} more)")

    print("\n  --- keys in model but NOT in safetensors (using fresh init) ---")
    for k in sorted(in_model_only)[:50]:
        v = model_sd[k]
        print(f"    FRESH    {k}  shape={tuple(v.shape)}  dtype={v.dtype}")
    if len(in_model_only) > 50:
        print(f"    ... ({len(in_model_only) - 50} more)")

    shape_mismatch = []
    dtype_mismatch = []
    for k in sorted(common):
        f, m = file_sd[k], model_sd[k]
        if tuple(f.shape) != tuple(m.shape):
            shape_mismatch.append((k, tuple(f.shape), tuple(m.shape)))
        if f.dtype != m.dtype:
            dtype_mismatch.append((k, f.dtype, m.dtype))

    print(f"\n  --- shape mismatches: {len(shape_mismatch)} ---")
    for k, fs, ms in shape_mismatch[:20]:
        print(f"    SHAPE    {k}  file={fs}  model={ms}")

    print(f"\n  --- dtype mismatches: {len(dtype_mismatch)} ---")
    for k, fd, md in dtype_mismatch[:20]:
        print(f"    DTYPE    {k}  file={fd}  model={md}")


def main():
    cfg_name = "pi05_openarm_ngc_lora_v4"
    print(f"Building PI0Pytorch model from config {cfg_name}...")
    model = build_model(cfg_name)
    print("Built.\n")

    for label, p in [
        ("LoRA-merged   (chocolate_bars_pi05_pytorch)",
         "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch/model.safetensors"),
        ("NO-LoRA BROKEN(chocolate_bars_pi05_pytorch_BROKEN_NO_LORA)",
         "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_BROKEN_NO_LORA/model.safetensors"),
    ]:
        print("=" * 72)
        print(f"AUDIT  {label}")
        print(f"  path: {p}")
        print("=" * 72)
        if not Path(p).exists():
            print(f"  MISSING: {p}")
            continue
        audit(p, model)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
