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
import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        
        logger.info(f"Loading Diffusion Policy from: {checkpoint_path}")
        self.policy = DiffusionPolicy.from_pretrained(checkpoint_path)
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        
        # Action chunking state
        self._action_chunk = None
        self._chunk_index = 0
        self._n_action_steps = self.policy.config.n_action_steps  # typically 8
        
        logger.info(f"Policy loaded: n_action_steps={self._n_action_steps}")
        logger.info(f"Input features: {list(self.policy.config.input_features.keys())}")
        
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
    
    # Import OpenPI's WebSocket server
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    
    # Create wrapped policy
    policy = DiffusionPolicyWrapper(args.checkpoint, device=args.device)
    
    # Serve it
    logger.info(f"Starting Diffusion Policy server on {args.host}:{args.port}")
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
