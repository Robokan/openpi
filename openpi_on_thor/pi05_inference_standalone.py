#!/usr/bin/env python3
"""Standalone Pi0.5 inference - no JAX, no config, just PyTorch."""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_openarm_example():
    """Create synthetic OpenArm example."""
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


def main():
    parser = argparse.ArgumentParser(description="Standalone Pi0.5 inference")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-test-runs", type=int, default=10)
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Pi0.5 Standalone Inference (No JAX)")
    logger.info("=" * 60)
    
    # Load config.json from checkpoint
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Config: {config}")
    
    # Import PyTorch model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models.pi0_config import Pi0Config
    
    # Create model config from JSON
    model_config = Pi0Config(
        action_dim=config.get("action_dim", 32),
        action_horizon=config.get("action_horizon", 50),
        max_token_len=config.get("max_token_len", 200),
        dtype=config.get("dtype", "bfloat16"),
        paligemma_variant=config.get("paligemma_variant", "gemma_2b_lora"),
        action_expert_variant=config.get("action_expert_variant", "gemma_300m_lora"),
        pi05=config.get("pi05", True),
    )
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint_dir}")
    weight_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    
    model = PI0Pytorch(config=model_config)
    state_dict = load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)
    
    logger.info("Model loaded successfully!")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create example
    logger.info("Creating synthetic OpenArm example...")
    example = create_openarm_example()
    
    # Run inference
    logger.info(f"Running inference (warmup={args.num_warmup}, test={args.num_test_runs})...")
    
    # Warmup
    for i in range(args.num_warmup):
        with torch.no_grad():
            _ = model.infer(example)
        torch.cuda.synchronize()
        logger.info(f"  Warmup {i+1}/{args.num_warmup} done")
    
    # Benchmark
    times = []
    for i in range(args.num_test_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            result = model.infer(example)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        times.append((t1 - t0) * 1000)
        logger.info(f"  Run {i+1}/{args.num_test_runs}: {times[-1]:.2f} ms")
    
    # Results
    actions = result.get("actions", result.get("action"))
    actions_np = np.array(actions) if not isinstance(actions, np.ndarray) else actions
    
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info("=" * 60)
    logger.info(f"Actions shape: {actions_np.shape}")
    logger.info(f"Actions range: [{np.min(actions_np):.4f}, {np.max(actions_np):.4f}]")
    logger.info(f"Inference time: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    logger.info(f"    (min: {np.min(times):.2f}, max: {np.max(times):.2f})")


if __name__ == "__main__":
    main()
