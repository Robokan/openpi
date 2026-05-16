#!/usr/bin/env python3
"""Serve a π₀.5 policy using TensorRT acceleration via WebSocket."""

import dataclasses
import logging
import os
import socket

import tyro

# Patch load_pytorch to handle dtype mismatches across transformers versions
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
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the TensorRT policy server."""

    # Training config name (e.g., "pi05_openarm_ngc_lora_v4").
    config: str = "pi05_openarm_ngc_lora_v4"

    # Checkpoint directory containing model.safetensors and assets.
    checkpoint_dir: str = "/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"

    # Path to the TensorRT engine file (.engine). Defaults to the FP8 + NVFP4 engine.
    engine_path: str = "/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch/model_fp4.engine"

    # Default prompt to use if not provided in observation.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8001


def main(args: Args) -> None:
    logging.info(f"Loading policy from {args.checkpoint_dir}")
    logging.info(f"Using config: {args.config}")

    # Verify paths exist
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if not os.path.exists(args.engine_path):
        raise FileNotFoundError(f"TensorRT engine not found: {args.engine_path}")

    # Load the policy with PyTorch model
    config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    logging.info("PyTorch policy loaded")

    # Setup TensorRT engine
    from openpi_on_thor.trt_model_forward import setup_pi0_tensorrt_engine

    policy = setup_pi0_tensorrt_engine(policy, args.engine_path)
    logging.info("TensorRT engine attached")

    # Get policy metadata
    policy_metadata = policy.metadata

    # Create and start server
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Starting TensorRT policy server")
    logging.info(f"  Host: {hostname} ({local_ip})")
    logging.info(f"  Port: {args.port}")
    logging.info(f"  Engine: {args.engine_path}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
