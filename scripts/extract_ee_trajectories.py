#!/usr/bin/env python3
"""
Extract end-effector trajectories from ALOHA simulation dataset.

This script:
1. Loads the ALOHA sim transfer cube dataset
2. Uses forward kinematics to compute end-effector poses
3. Saves trajectories as JSON for Isaac Lab

Usage:
    uv run scripts/extract_ee_trajectories.py --output-dir /path/to/output
"""

import argparse
import json
from pathlib import Path

import gym_aloha  # noqa: F401 - registers the environment
import gymnasium as gym
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def get_ee_pose_from_joints(physics, joint_positions: np.ndarray) -> dict:
    """
    Given joint positions, compute forward kinematics to get end-effector poses.
    
    Args:
        physics: MuJoCo physics object
        joint_positions: [14] array - [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    
    Returns:
        dict with left and right end-effector poses
    """
    # Set the joint positions in the physics simulation
    # ALOHA has 14 actuators matching the joint positions
    physics.data.qpos[:14] = joint_positions
    
    # Forward kinematics - this updates body positions
    physics.forward()
    
    # Get end-effector positions and orientations
    left_ee_pos = physics.named.data.xpos['vx300s_left/gripper_link'].copy()
    left_ee_quat = physics.named.data.xquat['vx300s_left/gripper_link'].copy()
    
    right_ee_pos = physics.named.data.xpos['vx300s_right/gripper_link'].copy()
    right_ee_quat = physics.named.data.xquat['vx300s_right/gripper_link'].copy()
    
    # Extract gripper states (normalized 0-1, where 0=closed, 1=open)
    left_gripper = joint_positions[6]
    right_gripper = joint_positions[13]
    
    return {
        "left": {
            "position": left_ee_pos.tolist(),      # [x, y, z]
            "orientation": left_ee_quat.tolist(),  # [w, x, y, z] quaternion
            "gripper": float(left_gripper),        # 0-1 normalized
        },
        "right": {
            "position": right_ee_pos.tolist(),
            "orientation": right_ee_quat.tolist(),
            "gripper": float(right_gripper),
        }
    }


def extract_episode_trajectory(physics, dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Extract end-effector trajectory for a single episode."""
    # Get episode data
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    
    trajectory = []
    
    for idx in range(from_idx, to_idx):
        sample = dataset[idx]
        
        # Get joint state - this is the observation state
        state = sample["observation.state"].numpy()
        
        # Get action (commanded joint positions)
        action = sample["action"].numpy()
        
        # Compute FK to get end-effector pose
        ee_pose = get_ee_pose_from_joints(physics, state)
        
        # Add timestep info
        ee_pose["timestep"] = idx - from_idx
        ee_pose["state"] = state.tolist()
        ee_pose["action"] = action.tolist()
        
        trajectory.append(ee_pose)
    
    return {
        "episode_idx": episode_idx,
        "num_steps": len(trajectory),
        "task": "transfer cube",
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract end-effector trajectories from ALOHA dataset")
    parser.add_argument("--output-dir", type=str, default="data/aloha_ee_trajectories",
                        help="Directory to save extracted trajectories")
    parser.add_argument("--dataset", type=str, default="lerobot/aloha_sim_transfer_cube_human",
                        help="LeRobot dataset ID")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Maximum number of episodes to extract (None = all)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
    
    num_episodes = dataset.num_episodes
    print(f"Dataset has {num_episodes} episodes, {len(dataset)} total frames")
    
    if args.max_episodes:
        num_episodes = min(num_episodes, args.max_episodes)
    
    # Create ALOHA environment to get physics for FK
    print("Loading ALOHA environment for forward kinematics...")
    env = gym.make('gym_aloha/AlohaTransferCube-v0')
    physics = env.unwrapped._env.physics
    
    # Extract trajectories
    all_trajectories = []
    
    print(f"Extracting {num_episodes} episodes...")
    for ep_idx in tqdm(range(num_episodes)):
        trajectory = extract_episode_trajectory(physics, dataset, ep_idx)
        all_trajectories.append(trajectory)
        
        # Also save individual episode files
        ep_file = output_dir / f"episode_{ep_idx:04d}.json"
        with open(ep_file, "w") as f:
            json.dump(trajectory, f, indent=2)
    
    # Save summary file
    summary = {
        "dataset": args.dataset,
        "num_episodes": num_episodes,
        "total_frames": sum(t["num_steps"] for t in all_trajectories),
        "robot": "ALOHA (dual vx300s)",
        "ee_bodies": ["vx300s_left/gripper_link", "vx300s_right/gripper_link"],
        "coordinate_frame": "world",
        "cube_info": {
            "size_meters": [0.02, 0.02, 0.02],
            "color": "red",
            "note": "Cube position is randomized per episode and NOT stored in dataset.",
        },
        "units": {
            "position": "meters",
            "orientation": "quaternion [w, x, y, z]",
            "gripper": "normalized [0=closed, 1=open]",
        },
        "episodes": [
            {"idx": t["episode_idx"], "num_steps": t["num_steps"]}
            for t in all_trajectories
        ]
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Extracted {num_episodes} episodes to {output_dir}")
    print(f"  - Individual episodes: episode_XXXX.json")
    print(f"  - Summary: summary.json")
    print(f"\nEach episode contains:")
    print(f"  - left/right end-effector position [x, y, z] in meters")
    print(f"  - left/right end-effector orientation as quaternion [w, x, y, z]")
    print(f"  - left/right gripper state [0=closed, 1=open]")
    
    env.close()


if __name__ == "__main__":
    main()
