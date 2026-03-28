#!/usr/bin/env python3
"""Fast norm stats computation - reads parquet files directly, skips images.

This computes norm stats in seconds instead of hours by avoiding image loading.

Usage:
    python scripts/compute_norm_stats_fast.py /path/to/lerobot/dataset

Example:
    python scripts/compute_norm_stats_fast.py ~/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v3
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def compute_stats(data: np.ndarray) -> dict:
    """Compute mean, std, q01, q99 for the data."""
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "q01": np.percentile(data, 1, axis=0).tolist(),
        "q99": np.percentile(data, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast norm stats computation")
    parser.add_argument("dataset_path", help="Path to LeRobot dataset")
    parser.add_argument("--output", "-o", help="Output path (default: dataset/norm_stats.json)")
    args = parser.parse_args()

    dataset_path = os.path.expanduser(args.dataset_path)
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(f"{dataset_path}/data/chunk-*/*.parquet"))
    
    if not parquet_files:
        print(f"ERROR: No parquet files found in {dataset_path}/data/chunk-*/")
        return 1
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_states = []
    all_actions = []
    
    for pf in parquet_files:
        table = pq.read_table(pf, columns=['observation.state', 'action'])
        
        states = np.array([np.array(row) for row in table.column('observation.state').to_pylist()])
        actions = np.array([np.array(row) for row in table.column('action').to_pylist()])
        
        all_states.append(states)
        all_actions.append(actions)
    
    all_states = np.vstack(all_states)
    all_actions = np.vstack(all_actions)
    
    print(f"Total frames: {len(all_states)}")
    print(f"State shape: {all_states.shape}")
    print(f"Action shape: {all_actions.shape}")
    
    norm_stats = {
        "norm_stats": {
            "state": compute_stats(all_states),
            "actions": compute_stats(all_actions),
        }
    }
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Get repo_id from path (last component)
        repo_id = os.path.basename(dataset_path.rstrip('/'))
        output_dir = f"{dataset_path}"
        output_path = f"{output_dir}/norm_stats.json"
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"\nNorm stats saved to: {output_path}")
    
    # Print summary
    state_stats = norm_stats['norm_stats']['state']
    print(f"\nState statistics (16 DOF):")
    print(f"  Range (q01 to q99):")
    for i in range(min(16, len(state_stats['q01']))):
        q01, q99 = state_stats['q01'][i], state_stats['q99'][i]
        std = state_stats['std'][i]
        flag = " ⚠️ LOW VARIANCE" if std < 0.01 else ""
        print(f"    Joint {i:2d}: [{q01:7.3f}, {q99:7.3f}] std={std:.4f}{flag}")
    
    return 0


if __name__ == "__main__":
    exit(main())
