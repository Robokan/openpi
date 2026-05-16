#!/usr/bin/env python3
"""Simplified Pi0.5 inference script for OpenArm - no lerobot dependency."""

import argparse
import logging
import os
import time

import numpy as np
import torch
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Patch load_pytorch for compatibility
import safetensors.torch as _st
from openpi.models_pytorch import pi0_pytorch as _pi0pt

def _load_pytorch_patched(self, train_config, weight_path: str):
    model = _pi0pt.PI0Pytorch(config=train_config.model)
    state_dict = _st.load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    return model

import openpi.models.model as _model_mod
for _cls in vars(_model_mod).values():
    if isinstance(_cls, type) and hasattr(_cls, "load_pytorch"):
        _cls.load_pytorch = _load_pytorch_patched


def create_openarm_example():
    """Create synthetic OpenArm example."""
    # OpenArm bimanual: 3 cameras, 16-dim state (7 joints + 1 gripper per arm)
    example = {
        "state": np.random.randn(16).astype(np.float32),
        "images": {
            "cam_high": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        },
        "prompt": "put the chocolate bars in the container",
    }
    return example


def run_pytorch_inference(model, example, num_warmup=3, num_test_runs=10):
    """Run PyTorch inference and measure timing."""
    logger.info("Running PyTorch inference...")
    
    # Warmup
    logger.info(f"  Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.infer(example)
    torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"  Benchmarking ({num_test_runs} runs)...")
    total_times = []
    model_times = []
    
    for i in range(num_test_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            result = model.infer(example)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        total_times.append((t1 - t0) * 1000)
        model_times.append((t1 - t0) * 1000)  # Same for now
    
    actions = result.get("actions", result.get("action"))
    
    return {
        "actions": actions,
        "total_times": total_times,
        "model_times": model_times,
    }


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 inference for OpenArm")
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-test-runs", type=int, default=10)
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Pi0.5 OpenArm Inference")
    logger.info("=" * 60)
    
    # Load config
    logger.info(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    logger.info(f"  Model: {train_config.model}")
    
    # Load model
    logger.info(f"Loading PyTorch model from: {args.checkpoint_dir}")
    weight_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    
    model = _pi0pt.PI0Pytorch(config=train_config.model)
    state_dict = _st.load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    
    logger.info("  Model loaded successfully!")
    
    # Create example
    logger.info("Creating synthetic OpenArm example...")
    example = create_openarm_example()
    logger.info(f"  State shape: {example['state'].shape}")
    logger.info(f"  Cameras: {list(example['images'].keys())}")
    logger.info(f"  Prompt: {example['prompt']}")
    
    # Run inference
    results = run_pytorch_inference(
        model, example,
        num_warmup=args.num_warmup,
        num_test_runs=args.num_test_runs,
    )
    
    # Print results
    actions = results["actions"]
    total_times = results["total_times"]
    model_times = results["model_times"]
    
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info("=" * 60)
    logger.info(f"Actions shape: {np.array(actions).shape}")
    logger.info(f"Actions range: [{np.min(actions):.4f}, {np.max(actions):.4f}]")
    logger.info(f"Total inference time: {np.mean(total_times):.2f} ± {np.std(total_times):.2f} ms")
    logger.info(f"    (min: {np.min(total_times):.2f}, max: {np.max(total_times):.2f})")
    logger.info(f"Model inference time: {np.mean(model_times):.2f} ± {np.std(model_times):.2f} ms")
    logger.info(f"    (min: {np.min(model_times):.2f}, max: {np.max(model_times):.2f})")


if __name__ == "__main__":
    main()
