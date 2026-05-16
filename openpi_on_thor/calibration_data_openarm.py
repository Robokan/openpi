#!/usr/bin/env python3
"""Calibration data loading utilities for OpenArm FP8/NVFP4 quantization.

OpenArm dataset structure:
- data/chunk-XXX/episode_XXXXXX.parquet  (state, action, frame_index, episode_index, task_index)
- videos/chunk-XXX/observation.images.{ego,left_wrist,right_wrist}/episode_XXXXXX.mp4
- meta/tasks.jsonl  (task_index -> task text)
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from openpi.policies import policy_config


class OpenArmCalibrationDataset(Dataset):
    """Dataset for FP8/NVFP4 calibration using real OpenArm data samples."""

    CAMERAS = ["ego", "left_wrist", "right_wrist"]
    CAMERA_MAP = {"ego": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}

    def __init__(
        self,
        config_obj,
        checkpoint_dir: str,
        num_samples: int = 32,
        device: str = "cuda",
        compute_dtype=torch.float16,
    ):
        print("  Initializing OpenArm calibration dataset...")

        policy = policy_config.create_trained_policy(config_obj, checkpoint_dir)
        self.input_transform = policy._input_transform
        self.policy = policy

        self.device = device
        self._pytorch_device = device
        self.compute_dtype = compute_dtype
        self.action_horizon = config_obj.model.action_horizon
        self.action_dim = config_obj.model.action_dim

        local_dir = Path(config_obj.data.base_config.local_dir)
        print(f"  Dataset path: {local_dir}")

        self.samples = []
        
        try:
            # Load tasks mapping
            tasks_map = self._load_tasks(local_dir / "meta" / "tasks.jsonl")
            print(f"  Loaded {len(tasks_map)} task definitions")

            # Find all parquet files
            parquet_files = sorted(local_dir.glob("data/chunk-*/episode_*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {local_dir}/data/")
            
            print(f"  Found {len(parquet_files)} episode files")

            # Calculate sampling strategy: spread across episodes
            samples_per_episode = max(1, num_samples // min(len(parquet_files), num_samples))
            episodes_to_sample = min(len(parquet_files), num_samples)
            
            episode_step = max(1, len(parquet_files) // episodes_to_sample)
            sampled_episodes = parquet_files[::episode_step][:episodes_to_sample]

            for parquet_path in sampled_episodes:
                try:
                    samples = self._load_episode_samples(
                        parquet_path, local_dir, tasks_map, samples_per_episode
                    )
                    self.samples.extend(samples)
                    if len(self.samples) >= num_samples:
                        break
                except Exception as e:
                    print(f"    Warning: Failed to load {parquet_path.name}: {e}")
                    continue

            self.samples = self.samples[:num_samples]
            print(f"  Successfully loaded {len(self.samples)} calibration samples with real images")

        except Exception as e:
            print(f"  Warning: Failed to load dataset: {e}")
            print("  Falling back to synthetic calibration data...")
            self.samples = self._create_synthetic_samples(num_samples)

        self.num_samples = len(self.samples)
        print(f"  Calibration dataset ready with {self.num_samples} samples")

    def _load_tasks(self, tasks_path: Path) -> dict:
        """Load task index -> task text mapping from tasks.jsonl."""
        tasks_map = {}
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    tasks_map[entry["task_index"]] = entry["task"]
        return tasks_map

    def _extract_frame_ffmpeg(self, video_path: Path, frame_idx: int) -> np.ndarray | None:
        """Extract a single frame from video using ffmpeg (handles AV1 codec)."""
        import subprocess
        import tempfile
        
        try:
            # Use ffmpeg to extract a single frame
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
                cmd = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-i', str(video_path),
                    '-vf', f'select=eq(n\\,{frame_idx})',
                    '-vframes', '1',
                    '-f', 'image2',
                    tmp.name
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.returncode == 0 and os.path.exists(tmp.name):
                    frame = cv2.imread(tmp.name)
                    if frame is not None:
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        return None

    def _load_episode_samples(
        self, parquet_path: Path, local_dir: Path, tasks_map: dict, max_samples: int
    ) -> list:
        """Load samples from a single episode parquet file with corresponding video frames."""
        samples = []
        
        # Load parquet data
        table = pq.read_table(parquet_path)
        total_rows = table.num_rows
        
        if total_rows == 0:
            return samples

        # Extract episode info from filename (episode_XXXXXX.parquet)
        episode_name = parquet_path.stem  # e.g., "episode_000000"
        chunk_name = parquet_path.parent.name  # e.g., "chunk-000"

        # Check which video files exist
        video_paths = {}
        for cam in self.CAMERAS:
            video_path = local_dir / "videos" / chunk_name / f"observation.images.{cam}" / f"{episode_name}.mp4"
            if video_path.exists():
                video_paths[cam] = video_path

        if not video_paths:
            print(f"    Warning: No video files found for {episode_name}")
            return samples

        # Sample frames evenly across the episode
        step_size = max(1, total_rows // max_samples)
        frame_indices = list(range(0, total_rows, step_size))[:max_samples]

        for row_idx in frame_indices:
            try:
                # Get state and task from parquet
                state = np.array(table.column("observation.state")[row_idx].as_py(), dtype=np.float32)
                frame_idx = int(table.column("frame_index")[row_idx].as_py())
                task_idx = int(table.column("task_index")[row_idx].as_py())
                
                prompt = tasks_map.get(task_idx, "pick up the object")

                # Extract frames from videos using ffmpeg (handles AV1)
                images = {}
                for cam, video_path in video_paths.items():
                    frame = self._extract_frame_ffmpeg(video_path, frame_idx)
                    if frame is not None:
                        # Resize to 224x224 if needed
                        if frame.shape[:2] != (224, 224):
                            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        # HWC to CHW
                        frame = np.transpose(frame, (2, 0, 1))
                        images[self.CAMERA_MAP[cam]] = frame.astype(np.uint8)

                # Need all 3 cameras for valid sample
                if len(images) == 3:
                    samples.append({
                        "state": state,
                        "images": images,
                        "prompt": prompt,
                    })

            except Exception as e:
                continue

        return samples

    def _create_synthetic_samples(self, num_samples: int) -> list:
        """Create synthetic samples as fallback."""
        samples = []
        for _ in range(num_samples):
            samples.append({
                "state": np.random.randn(16).astype(np.float32) * 0.5,
                "images": {
                    "cam_high": np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8),
                    "cam_left_wrist": np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8),
                    "cam_right_wrist": np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8),
                },
                "prompt": "pick up the object",
            })
        return samples

    def _process_data(self, data: dict):
        """Process raw sample into model input format."""
        import jax
        from openpi.models.model import Observation

        inputs = jax.tree.map(lambda x: x, data)
        inputs = self.input_transform(inputs)

        def convert_to_torch(x):
            tensor = torch.from_numpy(np.array(x))
            if tensor.dtype in [torch.float32, torch.float64]:
                tensor = tensor.to(dtype=self.compute_dtype)
            return tensor.to(self._pytorch_device)[None, ...]

        inputs = jax.tree.map(convert_to_torch, inputs)
        observation = Observation.from_dict(inputs)

        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(1, self.action_horizon, self.action_dim),
            dtype=self.compute_dtype,
            device=self.device,
        )
        return observation, noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._process_data(self.samples[idx])


def no_batch_collate_fn(batch):
    return batch[0]


def load_openarm_calibration_data(
    config_obj,
    checkpoint_dir: str,
    num_samples: int = 32,
    device: str = "cuda",
    batch_size: int = 1,
    compute_dtype=torch.float16,
):
    """Load calibration data from OpenArm dataset for FP8/NVFP4 quantization."""
    try:
        dataset = OpenArmCalibrationDataset(
            config_obj=config_obj,
            checkpoint_dir=checkpoint_dir,
            num_samples=num_samples,
            device=device,
            compute_dtype=compute_dtype,
        )

        if dataset.num_samples == 0:
            print("  Warning: No samples loaded, returning None")
            return None

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=no_batch_collate_fn,
        )

        return data_loader

    except Exception as e:
        print(f"  Warning: Failed to load OpenArm dataset: {e}")
        print("  Falling back to dummy inputs for calibration")
        return None
