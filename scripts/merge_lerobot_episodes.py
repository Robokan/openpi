#!/usr/bin/env python3
"""Merge individual LeRobot episode datasets into a single training dataset.

Each teleoperation session creates its own episode dataset:
    ~/sparkjax_recordings/lerobot/local/episodes/ep0000/
    ~/sparkjax_recordings/lerobot/local/episodes/ep0001/
    ...

This script merges them into a single dataset for training:
    ~/sparkjax_recordings/lerobot/local/openarm-training/

Usage:
    # Merge all episodes
    python merge_lerobot_episodes.py
    
    # Merge to custom output
    python merge_lerobot_episodes.py --output local/my-dataset
    
    # Dry run (show what would be merged)
    python merge_lerobot_episodes.py --dry-run
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge individual LeRobot episode datasets")
    parser.add_argument(
        "--lerobot-home", 
        default=os.path.expanduser("~/sparkjax_recordings/lerobot"),
        help="LeRobot home directory")
    parser.add_argument(
        "--episodes-dir", 
        default="local/episodes",
        help="Directory containing episode datasets (relative to lerobot-home)")
    parser.add_argument(
        "--output", 
        default="local/openarm-training",
        help="Output dataset repo_id")
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be merged without making changes")
    args = parser.parse_args()
    
    lerobot_home = Path(args.lerobot_home)
    episodes_dir = lerobot_home / args.episodes_dir
    output_path = lerobot_home / args.output
    
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return 1
    
    # Find all episode datasets
    episode_paths = []
    for d in sorted(episodes_dir.iterdir()):
        if d.is_dir() and d.name.startswith("ep"):
            info_path = d / "meta" / "info.json"
            if info_path.exists():
                episode_paths.append(d)
    
    if not episode_paths:
        print(f"No episode datasets found in {episodes_dir}")
        return 1
    
    print(f"Found {len(episode_paths)} episode datasets:")
    total_frames = 0
    for ep_path in episode_paths:
        with open(ep_path / "meta" / "info.json") as f:
            info = json.load(f)
        frames = info.get("total_frames", 0)
        total_frames += frames
        print(f"  {ep_path.name}: {frames} frames")
    
    print(f"\nTotal: {len(episode_paths)} episodes, {total_frames} frames")
    print(f"Output: {output_path}")
    
    if args.dry_run:
        print("\n[DRY RUN - no changes made]")
        return 0
    
    # Clean output directory
    if output_path.exists():
        print(f"\nRemoving existing {output_path}")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True)
    (output_path / "meta").mkdir()
    (output_path / "videos").mkdir()
    
    # Merge episodes
    print("\nMerging episodes...")
    global_frame_idx = 0
    episodes_meta = []
    
    for ep_idx, ep_path in enumerate(episode_paths):
        print(f"  Processing {ep_path.name} -> episode_{ep_idx:06d}")
        
        # Read episode info
        with open(ep_path / "meta" / "info.json") as f:
            ep_info = json.load(f)
        
        # Copy and renumber parquet file
        src_parquet = ep_path / "data" / "chunk-000" / "episode_000000.parquet"
        if src_parquet.exists():
            df = pd.read_parquet(src_parquet)
            num_frames = len(df)
            
            # Update indices
            df["episode_index"] = ep_idx
            df["index"] = range(global_frame_idx, global_frame_idx + num_frames)
            
            # Write to output
            dst_parquet = output_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
            df.to_parquet(dst_parquet)
            
            episodes_meta.append({
                "episode_index": ep_idx,
                "length": num_frames,
                "tasks": ["bimanual manipulation"],
            })
            
            global_frame_idx += num_frames
        
        # Copy videos if they exist
        src_videos = ep_path / "videos"
        if src_videos.exists():
            for cam_dir in src_videos.iterdir():
                if cam_dir.is_dir():
                    dst_cam_dir = output_path / "videos" / cam_dir.name
                    dst_cam_dir.mkdir(exist_ok=True)
                    
                    for video_file in cam_dir.glob("episode_*.mp4"):
                        # Rename to new episode index
                        dst_video = dst_cam_dir / f"episode_{ep_idx:06d}.mp4"
                        shutil.copy(video_file, dst_video)
    
    # Create merged metadata
    print("\nCreating metadata...")
    
    # Use first episode's info as template
    with open(episode_paths[0] / "meta" / "info.json") as f:
        info = json.load(f)
    
    info["total_episodes"] = len(episode_paths)
    info["total_frames"] = global_frame_idx
    info["splits"] = {"train": f"0:{len(episode_paths)}"}
    
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # Write episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    
    # Write tasks.jsonl
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "bimanual manipulation"}) + "\n")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Merged dataset created:")
    print(f"  Path: {output_path}")
    print(f"  Episodes: {len(episode_paths)}")
    print(f"  Frames: {global_frame_idx}")
    print(f"{'='*60}")
    
    print(f"\nTo delete bad episodes before merging:")
    print(f"  rm -rf {episodes_dir}/ep0042")
    print(f"\nTo train with this dataset:")
    print(f"  --repo_id {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
