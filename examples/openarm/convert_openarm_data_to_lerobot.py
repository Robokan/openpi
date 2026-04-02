#!/usr/bin/env python3
"""Convert OpenArm 22-dim teleop data to 16-dim ALOHA-style LeRobot dataset.

Reads the existing 22-dim LeRobot dataset (or raw teleop parquet data) and produces
a new 16-dim dataset with the correct joint ordering:
  [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

The 22-dim data from robot.data.joint_pos includes all articulation joints in USD
traversal order. This script extracts only the 16 meaningful DOFs.

IMPORTANT: Run scripts/discover_joint_mapping.py in the Isaac Lab container first
to verify the index mapping for your robot USD. The default mapping assumes:
  [0:7]  = left arm joints
  [7:14] = right arm joints
  [14]   = left gripper (first left finger joint)
  [17]   = right gripper (first right finger joint)

Usage:
    # Convert existing LeRobot dataset (recommended):
    uv run examples/openarm/convert_openarm_data_to_lerobot.py \\
        --input /path/to/22dim_lerobot_dataset \\
        --repo-id local/openarm-teleop-16dof

    # With custom joint mapping from discovery script:
    uv run examples/openarm/convert_openarm_data_to_lerobot.py \\
        --input /path/to/22dim_lerobot_dataset \\
        --repo-id local/openarm-teleop-16dof \\
        --joint-map /path/to/joint_mapping.json

    # Convert raw teleop parquet data (episodes/ directory):
    uv run examples/openarm/convert_openarm_data_to_lerobot.py \\
        --input /path/to/vla_teleop_data \\
        --repo-id local/openarm-teleop-16dof \\
        --raw
"""

import dataclasses
import io
import json
from pathlib import Path
import shutil
from typing import Literal

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
import tyro


# Default 22→16 index mapping (verify with discover_joint_mapping.py)
DEFAULT_LEFT_ARM_IDS = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_RIGHT_ARM_IDS = [7, 8, 9, 10, 11, 12, 13]
DEFAULT_LEFT_GRIP_ID = 14
DEFAULT_RIGHT_GRIP_ID = 17


MOTORS_16 = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "left_gripper",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
    "right_gripper",
]

CAMERAS = ["ego", "left_wrist", "right_wrist"]


@dataclasses.dataclass(frozen=True)
class JointMapping:
    """Maps 22-dim joint_pos indices to 16-dim ALOHA-style ordering."""
    left_arm_ids: list[int] = dataclasses.field(default_factory=lambda: list(DEFAULT_LEFT_ARM_IDS))
    right_arm_ids: list[int] = dataclasses.field(default_factory=lambda: list(DEFAULT_RIGHT_ARM_IDS))
    left_grip_id: int = DEFAULT_LEFT_GRIP_ID
    right_grip_id: int = DEFAULT_RIGHT_GRIP_ID

    @staticmethod
    def from_json(path: str) -> "JointMapping":
        with open(path) as f:
            data = json.load(f)
        m = data.get("aloha_16_mapping", data)
        return JointMapping(
            left_arm_ids=m.get("left_arm_indices", DEFAULT_LEFT_ARM_IDS),
            right_arm_ids=m.get("right_arm_indices", DEFAULT_RIGHT_ARM_IDS),
            left_grip_id=m.get("left_grip_index", DEFAULT_LEFT_GRIP_ID),
            right_grip_id=m.get("right_grip_index", DEFAULT_RIGHT_GRIP_ID),
        )

    def extract_16(self, data: np.ndarray) -> np.ndarray:
        """Extract 16-dim from 22-dim array, or pass through if already 16-dim."""
        last_dim = data.shape[-1] if data.ndim > 0 else 0
        if last_dim == 16:
            return data.astype(np.float32)
        indices = self.left_arm_ids + [self.left_grip_id] + self.right_arm_ids + [self.right_grip_id]
        return data[..., indices].astype(np.float32)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 4


def create_16dof_dataset(
    repo_id: str,
    fps: int = 60,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [MOTORS_16],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": [MOTORS_16],
        },
    }

    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="openarm_bimanual",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
    )


def convert_lerobot_dataset(
    input_dir: Path,
    repo_id: str,
    mapping: JointMapping,
    fps: int | None = None,
    max_episodes: int | None = None,
) -> None:
    """Convert an existing 22-dim LeRobot dataset to 16-dim."""
    info_path = input_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    detected_fps = info.get("fps", 60)
    if fps is None:
        fps = detected_fps
    total_episodes = info["total_episodes"]
    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes)
        print(f"Limiting to first {max_episodes} episodes")
    state_dim = info["features"]["observation.state"]["shape"][0]

    print(f"Input dataset: {input_dir}")
    print(f"  Episodes: {total_episodes}, FPS: {fps}, State dim: {state_dim}")

    if state_dim != 22:
        print(f"WARNING: Expected 22-dim state, got {state_dim}")

    tasks_path = input_dir / "meta" / "tasks.jsonl"
    tasks = {}
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                tasks[t["task_index"]] = t["task"]

    episodes_meta = []
    ep_path = input_dir / "meta" / "episodes.jsonl"
    if ep_path.exists():
        with open(ep_path) as f:
            for line in f:
                episodes_meta.append(json.loads(line))

    dataset = create_16dof_dataset(repo_id, fps=fps)

    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    for ep_idx in tqdm.tqdm(range(total_episodes), desc="Converting episodes"):
        chunk = ep_idx // info.get("chunks_size", 1000)
        parquet_path = input_dir / data_pattern.format(episode_chunk=chunk, episode_index=ep_idx)

        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping")
            continue

        table = pq.read_table(parquet_path)
        num_frames = len(table)

        task_idx = table.column("task_index")[0].as_py() if "task_index" in table.column_names else 0
        task_text = tasks.get(task_idx, "bimanual manipulation")

        for i in range(num_frames):
            state_22 = np.array(table.column("observation.state")[i].as_py(), dtype=np.float32)
            action_22 = np.array(table.column("action")[i].as_py(), dtype=np.float32)

            state_16 = mapping.extract_16(state_22)
            action_16 = mapping.extract_16(action_22)

            frame = {
                "observation.state": torch.from_numpy(state_16),
                "action": torch.from_numpy(action_16),
                "task": task_text,
            }

            for cam in CAMERAS:
                col_name = f"observation.images.{cam}"
                if col_name in table.column_names:
                    img_struct = table.column(col_name)[i].as_py()
                    img_bytes = img_struct["bytes"]
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    frame[col_name] = np.array(img)

            dataset.add_frame(frame)

        dataset.save_episode()
        
        # Flush images every 10 episodes to prevent queue overflow
        if (ep_idx + 1) % 10 == 0:
            dataset.stop_image_writer()
            dataset.start_image_writer()
            tqdm.tqdm.write(f"  Flushed images through episode {ep_idx}")

    # Final flush - stop image writer to ensure all images are written
    dataset.stop_image_writer()
    print(f"\nDone! Dataset saved to: {LEROBOT_HOME / repo_id}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  State/Action dim: 16")


def convert_raw_teleop(
    input_dir: Path,
    repo_id: str,
    mapping: JointMapping,
    max_episodes: int | None = None,
    include_mirrored: bool = False,
) -> None:
    """Convert raw teleop parquet data (episodes/ dir with per-column format)."""
    episodes_dir = input_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No episodes/ directory in {input_dir}")

    meta_path = input_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    fps = metadata.get("fps", 60)
    task_text = metadata.get("task_text", "bimanual manipulation")

    ep_dirs = sorted(
        [d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
        key=lambda x: int(x.name.split("_")[1]),
    )

    mirrored_dir = input_dir / "mirrored"
    mirrored_dirs: list[Path] = []
    if include_mirrored and mirrored_dir.exists():
        mirrored_dirs = sorted(
            [d for d in mirrored_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
            key=lambda x: int(x.name.split("_")[1]),
        )

    if max_episodes is not None:
        ep_dirs = ep_dirs[:max_episodes]
        mirrored_dirs = mirrored_dirs[:max_episodes]
        print(f"Limiting to first {max_episodes} episodes (per source)")

    all_dirs = mirrored_dirs + ep_dirs
    print(f"Raw teleop data: {input_dir}")
    print(f"  Original episodes: {len(ep_dirs)}")
    if mirrored_dirs:
        print(f"  Mirrored episodes: {len(mirrored_dirs)}")
    print(f"  Total: {len(all_dirs)}, FPS: {fps}")

    dataset = create_16dof_dataset(repo_id, fps=fps)

    for ep_idx, ep_dir in enumerate(tqdm.tqdm(all_dirs, desc="Converting episodes")):
        parquet_path = ep_dir / "data.parquet"
        if not parquet_path.exists():
            continue

        import pandas as pd
        df = pd.read_parquet(parquet_path)

        state_cols = sorted(
            [c for c in df.columns if c.startswith("observation.state.")],
            key=lambda x: int(x.split(".")[-1]),
        )
        action_cols = sorted(
            [c for c in df.columns if c.startswith("action.")],
            key=lambda x: int(x.split(".")[-1]),
        )

        states_22 = df[state_cols].values.astype(np.float32)
        actions_22 = df[action_cols].values.astype(np.float32)

        ep_meta_path = ep_dir / "metadata.json"
        ep_task = task_text
        if ep_meta_path.exists():
            with open(ep_meta_path) as f:
                ep_meta = json.load(f)
            ep_task = ep_meta.get("task_text", task_text)

        camera_files = {}
        for cam in CAMERAS:
            cam_dir = ep_dir / cam
            if cam_dir.exists():
                files = sorted(
                    list(cam_dir.glob("frame_*.png")) + list(cam_dir.glob("frame_*.jpg")),
                    key=lambda f: f.stem,
                )
                camera_files[cam] = files

        num_frames = len(states_22)
        if camera_files:
            min_imgs = min(len(f) for f in camera_files.values() if f)
            num_frames = min(num_frames, min_imgs)

        for i in range(num_frames):
            state_16 = mapping.extract_16(states_22[i])
            action_16 = mapping.extract_16(actions_22[i])

            frame = {
                "observation.state": torch.from_numpy(state_16),
                "action": torch.from_numpy(action_16),
                "task": ep_task,
            }

            for cam, files in camera_files.items():
                if i < len(files):
                    from PIL import Image
                    img = Image.open(files[i])
                    img_array = np.array(img)
                    if img_array.ndim == 3 and img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                    frame[f"observation.images.{cam}"] = img_array

            dataset.add_frame(frame)

        dataset.save_episode()
        
        # Flush images every 10 episodes to prevent queue overflow
        if (ep_idx + 1) % 10 == 0:
            dataset.stop_image_writer()
            dataset.start_image_writer()
            tqdm.tqdm.write(f"  Flushed images through episode {ep_idx}")

    # Final flush - stop image writer to ensure all images are written
    dataset.stop_image_writer()
    print(f"\nDone! Dataset saved to: {LEROBOT_HOME / repo_id}")


def main(
    input: str,
    repo_id: str = "local/openarm-teleop-16dof",
    joint_map: str | None = None,
    raw: bool = False,
    fps: int | None = None,
    symlink: str | None = None,
    max_episodes: int | None = None,
    include_mirrored: bool = True,
):
    """Convert OpenArm 22-dim data to 16-dim LeRobot dataset.
    
    Args:
        input: Path to input data (LeRobot dataset or raw teleop dir)
        repo_id: HuggingFace repo ID for output dataset
        joint_map: Path to joint_mapping.json from discover_joint_mapping.py
        raw: If True, treat input as raw teleop format (episodes/ with per-column parquet)
        fps: Override FPS (default: from metadata)
        symlink: Create a symlink at this path pointing to the output dataset.
            Placed next to the input by default (e.g. input_16dof).
        max_episodes: Maximum number of episodes to convert (for testing)
        include_mirrored: Include mirrored/ episodes alongside originals (default: True)
    """
    input_dir = Path(input).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")

    if joint_map:
        mapping = JointMapping.from_json(joint_map)
        print(f"Using joint mapping from: {joint_map}")
    else:
        mapping = JointMapping()
        print("Using DEFAULT joint mapping (verify with discover_joint_mapping.py):")

    print(f"  Left arm:   {mapping.left_arm_ids}")
    print(f"  Left grip:  {mapping.left_grip_id}")
    print(f"  Right arm:  {mapping.right_arm_ids}")
    print(f"  Right grip: {mapping.right_grip_id}")
    print()

    if raw:
        convert_raw_teleop(input_dir, repo_id, mapping, max_episodes=max_episodes,
                           include_mirrored=include_mirrored)
    else:
        is_lerobot = (input_dir / "meta" / "info.json").exists()
        if is_lerobot:
            convert_lerobot_dataset(input_dir, repo_id, mapping, fps=fps, max_episodes=max_episodes)
        else:
            convert_raw_teleop(input_dir, repo_id, mapping, max_episodes=max_episodes,
                               include_mirrored=include_mirrored)

    dataset_dir = LEROBOT_HOME / repo_id
    if symlink is None:
        symlink = str(input_dir.parent / f"{input_dir.name}_16dof")
    symlink_path = Path(symlink)
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()
    symlink_path.symlink_to(dataset_dir)
    print(f"Symlink: {symlink_path} -> {dataset_dir}")


if __name__ == "__main__":
    tyro.cli(main)
