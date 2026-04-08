#!/usr/bin/env python3
"""Generate a combined 3-camera video (left_wrist | ego | right_wrist) for a LeRobot episode.

Usage:
    # Inside the openpi_pytorch container:
    python scripts/view_episode.py 0
    python scripts/view_episode.py 5 --repo-id local/openarm-teleop-16dof-v4

    # From the host (one-liner):
    cd ~/sparkpack/openpi && docker compose -f scripts/docker/compose_ngc.yml run --rm openpi_pytorch \
        python scripts/view_episode.py 0
"""

import argparse
import sys
from pathlib import Path

import av
import numpy as np


CAMERAS = [
    "observation.images.left_wrist",
    "observation.images.ego",
    "observation.images.right_wrist",
]


def main():
    parser = argparse.ArgumentParser(description="View a LeRobot episode (3 cameras side-by-side)")
    parser.add_argument("episode", type=int, help="Episode index to view")
    parser.add_argument("--repo-id", default="local/openarm-teleop-16dof-v4",
                        help="LeRobot dataset repo ID")
    parser.add_argument("--output", default=None,
                        help="Output path (default: /app/episode_N.mp4)")
    args = parser.parse_args()

    cache = Path.home() / ".cache/huggingface/lerobot" / args.repo_id
    video_dir = cache / "videos" / "chunk-000"

    if not video_dir.exists():
        print(f"ERROR: {video_dir} not found")
        sys.exit(1)

    ep_str = f"episode_{args.episode:06d}.mp4"
    containers = []
    decoders = []
    for cam in CAMERAS:
        path = video_dir / cam / ep_str
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)
        c = av.open(str(path))
        containers.append(c)
        decoders.append(c.decode(c.streams.video[0]))

    s = containers[0].streams.video[0]
    fps = float(s.average_rate)
    w, h = s.width, s.height

    default_dir = Path("/app/outputs/viz")
    default_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(default_dir / f"episode_{args.episode}.mp4")
    out = av.open(out_path, mode="w")
    out_stream = out.add_stream("libx264", rate=int(fps))
    out_stream.width = w * 3
    out_stream.height = h
    out_stream.pix_fmt = "yuv420p"

    count = 0
    try:
        while True:
            frames = []
            for dec in decoders:
                f = next(dec, None)
                if f is None:
                    raise StopIteration
                frames.append(f.to_ndarray(format="rgb24"))
            combined = np.concatenate(frames, axis=1)
            for packet in out_stream.encode(av.VideoFrame.from_ndarray(combined, format="rgb24")):
                out.mux(packet)
            count += 1
    except StopIteration:
        pass

    for packet in out_stream.encode():
        out.mux(packet)
    out.close()
    for c in containers:
        c.close()

    print(f"Episode {args.episode}: {count} frames, {count/fps:.1f}s @ {fps}fps")
    print(f"Layout: left_wrist | ego | right_wrist ({w*3}x{h})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
