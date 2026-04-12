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
import numpy as np
import torch
import websockets.asyncio.server as ws_server
import websockets.frames


def _unpack_array(obj):
    """Unpack numpy arrays using openpi_client's format."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _pack_array(obj):
    """Pack numpy arrays using openpi_client's format."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                obs = msgpack.unpackb(data, object_hook=_unpack_array, raw=False)
                
                # Run inference
                infer_start = time.monotonic()
                result = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start
                
                # Add timing info
                result["server_timing"] = {"infer_ms": infer_time * 1000}
                
                # Send response
                await websocket.send(msgpack.packb(result, default=_pack_array))
                
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
    - Returns full action chunks (SparkJAX's ActionChunkBroker handles iteration)
    """
    
    # Camera name mapping: SparkJAX -> Diffusion Policy
    CAM_REMAP = {
        'cam_high': 'observation.images.ego',
        'cam_left_wrist': 'observation.images.left_wrist', 
        'cam_right_wrist': 'observation.images.right_wrist',
    }
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", num_inference_steps: int = 10, use_ddim: bool = True):
        import json
        from pathlib import Path
        from safetensors.torch import load_file
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from diffusers import DDIMScheduler
        
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
        
        # Override num_inference_steps for faster inference (default 100 is too slow)
        cfg_dict['num_inference_steps'] = num_inference_steps
        
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
        
        # Switch to DDIM scheduler for fast inference (works with 10-20 steps vs 100 for DDPM)
        if use_ddim:
            logger.info("Switching to DDIM scheduler for fast inference")
            ddim_scheduler = DDIMScheduler(
                num_train_timesteps=self.policy.diffusion.noise_scheduler.config.num_train_timesteps,
                beta_start=self.policy.diffusion.noise_scheduler.config.beta_start,
                beta_end=self.policy.diffusion.noise_scheduler.config.beta_end,
                beta_schedule=self.policy.diffusion.noise_scheduler.config.beta_schedule,
                clip_sample=self.policy.diffusion.noise_scheduler.config.clip_sample,
                prediction_type=self.policy.diffusion.noise_scheduler.config.prediction_type,
            )
            ddim_scheduler.set_timesteps(num_inference_steps)
            self.policy.diffusion.noise_scheduler = ddim_scheduler
        
        self._n_action_steps = config.n_action_steps  # typically 8
        
        logger.info(f"Policy loaded: n_action_steps={self._n_action_steps}, num_inference_steps={num_inference_steps}, scheduler={'DDIM' if use_ddim else 'DDPM'}")
        logger.info(f"Input features: {list(config.input_features.keys())}")
        
    def reset(self):
        """Reset policy state."""
        self.policy.reset()
        
    def infer(self, obs: dict) -> dict:
        """Run inference matching OpenPI's interface.
        
        Bypasses LeRobot's internal queue to return full action chunk directly.
        This is critical for performance - one inference gives n_action_steps actions.
        
        Args:
            obs: dict with 'state', 'images', 'prompt' from SparkJAX
            
        Returns:
            dict with 'actions' as full chunk (n_action_steps, 16)
        """
        import time
        infer_start = time.monotonic()
        
        batch = self._prepare_batch(obs)
        
        with torch.no_grad():
            # Normalize inputs (like select_action does)
            batch = self.policy.normalize_inputs(batch)
            
            # Stack images if needed
            if self.policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = torch.stack(
                    [batch[key] for key in self.policy.config.image_features], dim=-4
                )
            
            # Add temporal dimension for observation history
            # For single observation, repeat n_obs_steps times
            n_obs = self.policy.config.n_obs_steps
            batch_with_history = {}
            for k, v in batch.items():
                if k in ["observation.state", "observation.images"]:
                    # Add time dimension: [B, ...] -> [B, T, ...]
                    batch_with_history[k] = v.unsqueeze(1).repeat(1, n_obs, *([1] * (v.dim() - 1)))
                else:
                    batch_with_history[k] = v
            
            # Generate full action trajectory
            actions = self.policy.diffusion.generate_actions(batch_with_history)
            
            # Unnormalize actions
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
            
            # actions shape: [batch=1, horizon, action_dim]
            # Take n_action_steps starting from n_obs_steps-1 (current timestep)
            start_idx = n_obs - 1
            end_idx = start_idx + self._n_action_steps
            actions = actions[0, start_idx:end_idx]  # [n_action_steps, action_dim]
        
        # Convert to numpy
        actions = actions.cpu().numpy().copy()
        
        infer_time = time.monotonic() - infer_start
        logger.info(f"[INFER] {infer_time*1000:.0f}ms, chunk shape={actions.shape}, first: {actions[0, :4]}")
        
        return {
            "actions": actions,
            "is_cached": False,
        }
    
    def _prepare_batch(self, obs: dict) -> dict:
        """Convert SparkJAX observation to Diffusion Policy batch format."""
        batch = {}
        
        # State: [16] float32 -> [1, 16] tensor
        state = obs.get("state", np.zeros(16, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = state.copy()  # Ensure writable
        else:
            state = np.array(state, dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        # Images: remap and convert
        images = obs.get("images", {})
        for spark_name, policy_name in self.CAM_REMAP.items():
            if spark_name in images:
                img = images[spark_name]
                # SparkJAX sends [3, H, W] uint8
                if isinstance(img, np.ndarray):
                    # Debug: log image stats
                    logger.info(f"[IMG] {spark_name}: shape={img.shape}, dtype={img.dtype}, "
                               f"min={img.min():.2f}, max={img.max():.2f}, mean={img.mean():.2f}")
                    # Normalize to [0, 1] float and add batch dim
                    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    batch[policy_name] = img_tensor
            else:
                # Create zero image if missing
                logger.warning(f"[IMG] {spark_name}: MISSING, using zeros")
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
    parser.add_argument(
        "--num-inference-steps", "-n",
        type=int,
        default=10,
        help="Number of diffusion denoising steps (default: 10, lower=faster but less accurate)")
    parser.add_argument(
        "--no-ddim",
        action="store_true",
        help="Disable DDIM scheduler (use original DDPM, requires more steps)")
    args = parser.parse_args()
    
    # Create wrapped policy
    policy = DiffusionPolicyWrapper(
        args.checkpoint, 
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        use_ddim=not args.no_ddim
    )
    
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
