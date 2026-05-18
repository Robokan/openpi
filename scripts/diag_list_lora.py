#!/usr/bin/env python3
"""List ALL LoRA keys in the JAX checkpoint to find what the conversion may have missed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"


def main() -> int:
    from openpi.models import model as _model
    from flax import traverse_util as traversals

    print("Loading JAX params ...")
    params = _model.restore_params(f"{CKPT_JAX}/params/", restore_type=np.ndarray, dtype=None)
    print("Top-level keys:", list(params.keys()))

    print()
    for top in params:
        flat = traversals.flatten_dict(params[top], sep="/")
        lora_keys = [k for k in flat if "lora" in k.lower()]
        print(f"\n=== {top} ({len(flat)} total, {len(lora_keys)} lora) ===")
        for k in sorted(lora_keys):
            print(f"  {k}  shape={tuple(flat[k].shape)}")
        non_lora_unique = set()
        for k in flat:
            if "lora" in k.lower():
                continue
            # remove layer indexing if any
            non_lora_unique.add(k)
        print(f"  unique non-lora paths: {len(non_lora_unique)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
