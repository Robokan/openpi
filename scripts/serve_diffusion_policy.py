#!/usr/bin/env python3
"""Serve a LeRobot Diffusion Policy using the same WebSocket protocol as OpenPI.

This allows SparkJAX to use Diffusion Policy checkpoints with the same client
code that works with π₀.₅.

Usage:
    # Serve the latest checkpoint on port 8001
    python scripts/serve_diffusion_policy.py \
        --checkpoint outputs/train/diffusion_openarm_v4/checkpoints/last/pretrained_model \
        --port 8001

    # Or specify a specific checkpoint
    python scripts/serve_diffusion_policy.py \
        --checkpoint outputs/train/diffusion_openarm_v4/checkpoints/045000/pretrained_model \
        --port 8001
"""

import argparse
import asyncio
import http
import logging
import time
import traceback

import msgpack
import msgpack_numpy
import numpy as np
import torch
import websockets.asyncio.server as ws_server
import websockets.frames

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register msgpack numpy support
msgpack_numpy.patch()


class WebsocketPolicyServer:
    """Simple WebSocket server matching OpenPI's protocol."""

    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8001, metadata: dict = None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"Diffusion Policy server listening on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        
        # Send metadata
        await websocket.send(msgpack.packb(self._metadata))
        
        while True:
            try:
                start_time = time.monotonic()
                
                # Receive observation
                data = await websocket.recv()
                obs = msgpack.unpackb(data, raw=False)
                
                # Run inference
                infer_start = time.monotonic()
                result = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start
                
                # Add timing info
                result["server_timing"] = {"infer_ms": infer_time * 1000}
                
                # Send response
                await websocket.send(msgpack.packb(result))
                
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                # Reset policy state for next connection
                self._policy.reset()
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.error(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason=str(e)[:120]
                )
                break

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


class DiffusionPolicyWrapper:
    """Wraps LeRobot Diffusion Policy to match OpenPI's inference interface.
    
    Handles:
    - Camera name remapping (cam_high -> ego, etc.)
    - Image normalization and tensor conversion
    - State normalization using checkpoint's stats
    - Action chunking (returns actions one at a time from chunks)
    """
    
    # Camera name mapping: SparkJAX -> Diffusion Policy
    CAM_REMAP = {
        'cam_high': 'observation.images.ego',
        'cam_left_wrist': 'observation.images.left_wrist', 
        'cam_right_wrist': 'observation.images.right_wrist',
    }
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        import json
        from pathlib import Path
        from safetensors.torch import load_file
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading Diffusion Policy from: {checkpoint_path}")
        
        # Load config
        config_path = checkpoint_path / "config.json"
        with open(config_path) as f:
            cfg_dict = json.load(f)
        
        # Convert features to proper PolicyFeature objects
        input_features = {}
        for key, feat in cfg_dict['input_features'].items():
            ft_type = FeatureType[feat['type']]
            input_features[key] = PolicyFeature(type=ft_type, shape=feat['shape'])
        
        output_features = {}
        for key, feat in cfg_dict['output_features'].items():
            ft_type = FeatureType[feat['type']]
            output_features[key] = PolicyFeature(type=ft_type, shape=feat['shape'])
        
        # Convert normalization_mapping to proper NormalizationMode
        norm_map = {}
        for key, value in cfg_dict['normalization_mapping'].items():
            norm_map[FeatureType[key]] = NormalizationMode[value]
        
        cfg_dict['input_features'] = input_features
        cfg_dict['output_features'] = output_features
        cfg_dict['normalization_mapping'] = norm_map
        
        # Create config and model
        config = DiffusionConfig(**cfg_dict)
        self.policy = DiffusionPolicy(config)
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        
        # Load weights
        weights_path = checkpoint_path / "model.safetensors"
        weights = load_file(str(weights_path))
        self.policy.load_state_dict(weights)
        
        # Action chunking state
        self._action_chunk = None
        self._chunk_index = 0
        self._n_action_steps = config.n_action_steps  # typically 8
        
        logger.info(f"Policy loaded: n_action_steps={self._n_action_steps}")
        logger.info(f"Input features: {list(config.input_features.keys())}")
        
    def reset(self):
        """Reset the action chunk state."""
        self._action_chunk = None
        self._chunk_index = 0
        self.policy.reset()
        
    def infer(self, obs: dict) -> dict:
        """Run inference matching OpenPI's interface.
        
        Args:
            obs: dict with 'state', 'images', 'prompt' from SparkJAX
            
        Returns:
            dict with 'actions' (single action) and timing info
        """
        # Check if we have cached actions
        if self._action_chunk is not None and self._chunk_index < len(self._action_chunk):
            action = self._action_chunk[self._chunk_index]
            self._chunk_index += 1
            return {
                "actions": action,
                "is_cached": True,
            }
        
        # Need to run inference for new chunk
        batch = self._prepare_batch(obs)
        
        with torch.no_grad():
            # select_action returns [horizon, action_dim] or [action_dim]
            actions = self.policy.select_action(batch)
        
        # Convert to numpy and handle shape
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # If we got a single action, wrap it
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
            
        # Store chunk and return first action
        self._action_chunk = actions
        self._chunk_index = 1  # Already returning index 0
        
        return {
            "actions": actions[0],
            "is_cached": False,
            "chunk_size": len(actions),
        }
    
    def _prepare_batch(self, obs: dict) -> dict:
        """Convert SparkJAX observation to Diffusion Policy batch format."""
        batch = {}
        
        # State: [16] float32 -> [1, 16] tensor
        state = obs.get("state", np.zeros(16, dtype=np.float32))
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        # Images: remap and convert
        images = obs.get("images", {})
        for spark_name, policy_name in self.CAM_REMAP.items():
            if spark_name in images:
                img = images[spark_name]
                # SparkJAX sends [3, H, W] uint8
                if isinstance(img, np.ndarray):
                    # Normalize to [0, 1] float and add batch dim
                    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    batch[policy_name] = img_tensor
            else:
                # Create zero image if missing
                batch[policy_name] = torch.zeros(1, 3, 224, 224, device=self.device)
        
        return batch


def main():
    parser = argparse.ArgumentParser(description="Serve Diffusion Policy via WebSocket")
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to pretrained_model directory")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8001,
        help="Port to serve on (default: 8001)")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda)")
    args = parser.parse_args()
    
    # Create wrapped policy
    policy = DiffusionPolicyWrapper(args.checkpoint, device=args.device)
    
    # Serve it
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "policy_type": "diffusion",
            "checkpoint": args.checkpoint,
        }
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
