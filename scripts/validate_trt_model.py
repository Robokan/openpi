#!/usr/bin/env python3
"""Validate TensorRT model output against PyTorch model output."""

import dataclasses
import logging
import os

import numpy as np
import torch
import tyro

# Patch load_pytorch for compatibility
import safetensors.torch as _st
from openpi.models_pytorch import pi0_pytorch as _pi0pt
import openpi.models.model as _model_mod

def _load_pytorch_patched(self, train_config, weight_path: str):
    model = _pi0pt.PI0Pytorch(config=train_config.model)
    state_dict = _st.load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    return model

for _cls in vars(_model_mod).values():
    if isinstance(_cls, type) and hasattr(_cls, "load_pytorch"):
        _cls.load_pytorch = _load_pytorch_patched

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str = "pi05_openarm_ngc_lora_v4"
    checkpoint_dir: str = "/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"
    engine_path: str = "/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch/model_fp4.engine"
    num_tests: int = 5


def create_openarm_example():
    """Create a realistic OpenArm example."""
    return {
        "state": np.random.randn(16).astype(np.float32) * 0.5,  # Rough joint angle range
        "images": {
            "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "pick up the chocolate bar",
    }


def main(args: Args):
    print("=" * 60)
    print("TensorRT vs PyTorch Model Validation")
    print("=" * 60)
    
    # Load PyTorch policy
    print("\n[1/4] Loading PyTorch policy...")
    config = _config.get_config(args.config)
    pytorch_policy = _policy_config.create_trained_policy(
        config, args.checkpoint_dir
    )
    print("  PyTorch policy loaded")
    
    # Create fixed noise for reproducibility
    print("\n[2/4] Creating test inputs...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create test example
    example = create_openarm_example()
    
    # Create fixed noise (action_horizon=50, action_dim=32)
    noise = np.random.randn(50, 32).astype(np.float32)
    
    print(f"  State shape: {example['state'].shape}")
    print(f"  Noise shape: {noise.shape}")
    
    # Run PyTorch inference
    print("\n[3/4] Running PyTorch inference...")
    pytorch_results = []
    for i in range(args.num_tests):
        result = pytorch_policy.infer(example, noise=noise)
        actions = result["actions"]
        pytorch_results.append(actions)
        print(f"  Test {i+1}: actions shape={actions.shape}, "
              f"min={actions.min():.4f}, max={actions.max():.4f}, "
              f"mean={actions.mean():.4f}")
    
    pytorch_actions = np.stack(pytorch_results)
    print(f"\n  PyTorch consistency check:")
    print(f"    Mean std across runs: {pytorch_actions.std(axis=0).mean():.6f}")
    print(f"    First action[0,:8]: {pytorch_results[0][0,:8]}")
    
    # Load TRT policy
    print("\n[4/4] Loading TensorRT policy...")
    # Need to reload since TRT setup modifies the model
    trt_policy = _policy_config.create_trained_policy(
        config, args.checkpoint_dir
    )
    
    from openpi_on_thor.trt_model_forward import setup_pi0_tensorrt_engine
    trt_policy = setup_pi0_tensorrt_engine(trt_policy, args.engine_path)
    print("  TensorRT policy loaded")
    
    # Run TRT inference
    print("\n[5/5] Running TensorRT inference...")
    trt_results = []
    for i in range(args.num_tests):
        result = trt_policy.infer(example, noise=noise)
        actions = result["actions"]
        trt_results.append(actions)
        print(f"  Test {i+1}: actions shape={actions.shape}, "
              f"min={actions.min():.4f}, max={actions.max():.4f}, "
              f"mean={actions.mean():.4f}")
    
    trt_actions = np.stack(trt_results)
    print(f"\n  TRT consistency check:")
    print(f"    Mean std across runs: {trt_actions.std(axis=0).mean():.6f}")
    print(f"    First action[0,:8]: {trt_results[0][0,:8]}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    diff = np.abs(pytorch_results[0] - trt_results[0])
    print(f"\n  Max absolute difference: {diff.max():.6f}")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Std of difference: {diff.std():.6f}")

    # Cosine similarity (the metric the Jetson AI Lab tutorial uses; >0.99 is considered good)
    pt_flat = pytorch_results[0].reshape(-1).astype(np.float64)
    trt_flat = trt_results[0].reshape(-1).astype(np.float64)
    cos_overall = float(np.dot(pt_flat, trt_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(trt_flat) + 1e-12))
    # Per-timestep cosine
    pt_2d = pytorch_results[0].astype(np.float64)
    trt_2d = trt_results[0].astype(np.float64)
    per_t = []
    for t in range(pt_2d.shape[0]):
        a, b = pt_2d[t], trt_2d[t]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        per_t.append(c)
    per_t = np.array(per_t)
    print(f"\n  Cosine similarity (tutorial's primary metric, target >0.99):")
    print(f"    Overall:           {cos_overall:.6f}")
    print(f"    Per-timestep mean: {per_t.mean():.6f}")
    print(f"    Per-timestep min:  {per_t.min():.6f}")
    print(f"    Per-timestep max:  {per_t.max():.6f}")
    
    # Check for concerning values
    print("\n  Sanity checks:")
    print(f"    PyTorch has NaN: {np.isnan(pytorch_results[0]).any()}")
    print(f"    TRT has NaN: {np.isnan(trt_results[0]).any()}")
    print(f"    PyTorch has Inf: {np.isinf(pytorch_results[0]).any()}")
    print(f"    TRT has Inf: {np.isinf(trt_results[0]).any()}")
    
    # Expected ranges for joint angles (rough estimates)
    print(f"\n  Value ranges:")
    print(f"    PyTorch: [{pytorch_results[0].min():.3f}, {pytorch_results[0].max():.3f}]")
    print(f"    TRT:     [{trt_results[0].min():.3f}, {trt_results[0].max():.3f}]")
    
    # Check if values are in reasonable range for joint angles (radians)
    MAX_REASONABLE = 5.0  # ~286 degrees
    pytorch_ok = np.abs(pytorch_results[0]).max() < MAX_REASONABLE
    trt_ok = np.abs(trt_results[0]).max() < MAX_REASONABLE
    
    print(f"\n  Values in reasonable range (< {MAX_REASONABLE} rad):")
    print(f"    PyTorch: {'YES' if pytorch_ok else 'NO - DANGEROUS'}")
    print(f"    TRT:     {'YES' if trt_ok else 'NO - DANGEROUS'}")
    
    if diff.max() > 0.5:
        print("\n  WARNING: Large differences between PyTorch and TRT!")
        print("  DO NOT use TRT model on real robot until this is resolved.")
    elif not trt_ok:
        print("\n  WARNING: TRT outputs are outside safe range!")
        print("  DO NOT use TRT model on real robot.")
    else:
        print("\n  TRT model appears to match PyTorch output.")
        print("  Proceed with caution on real robot.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
