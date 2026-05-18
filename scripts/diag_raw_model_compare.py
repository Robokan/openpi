#!/usr/bin/env python3
"""Compare JAX vs PyTorch raw model output AND post-transform output side by side.

Uses both policies' real input/output transform pipelines (same code path as
the serving stack) so any difference is purely from the model. Captures
the raw model output (pre-output-transform) by monkey-patching the wrapped
_sample_actions function.

Run inside openpi-pt-server (openpi-thor:latest):
    docker exec openpi-pt-server bash -lc \
        "cd /app && PYTHONPATH=/app/src:/app/packages/openpi-client/src \
         python scripts/diag_raw_model_compare.py"
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert a.shape == b.shape, f"{name} shape mismatch: {a.shape} vs {b.shape}"
    cos = _cosine(a, b)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    print(f"  [{name:40s}] cos={cos:+.4f}  |a|={na:.4f}  |b|={nb:.4f}  ratio_b/a={nb / max(na, 1e-12):.3f}")


def _load_obs():
    import subprocess

    import cv2
    import pyarrow.parquet as pq

    ds = Path("/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parquet = sorted(ds.glob("data/chunk-*/episode_*.parquet"))[0]
    table = pq.read_table(parquet)
    row = 100
    state = np.asarray(table.column("observation.state")[row].as_py(), dtype=np.float32)
    fidx = int(table.column("frame_index")[row].as_py())
    tidx = int(table.column("task_index")[row].as_py())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (ds / "meta" / "tasks.jsonl").read_text().splitlines()}
    prompt = tasks.get(tidx, "do the task")
    chunk, ep = parquet.parent.name, parquet.stem
    images = {}
    for cam_in, cam_out in (("ego", "cam_high"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist")):
        vp = ds / "videos" / chunk / f"observation.images.{cam_in}" / f"{ep}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            name = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp),
             "-vf", f"select=eq(n\\,{fidx})", "-vframes", "1", "-f", "image2", name],
            capture_output=True, timeout=15, check=True,
        )
        frame = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def _wrap_capture_raw(policy, captured: dict, key: str):
    """Monkey-patch policy._sample_actions to capture its return value."""
    orig = policy._sample_actions

    def wrapped(rng_or_device, observation, **kwargs):
        out = orig(rng_or_device, observation, **kwargs)
        # JAX returns a jax.Array, PT returns torch.Tensor
        import jax.numpy as jnp
        import torch
        if isinstance(out, torch.Tensor):
            captured[key] = out.detach().float().cpu().numpy()
        else:
            captured[key] = np.asarray(out)
        return out

    policy._sample_actions = wrapped


def main() -> int:
    print("=" * 90)
    print("JAX vs PYTORCH:  raw model output AND post-transform output")
    print("=" * 90)

    raw_obs = _load_obs()
    print(f"state[:16]={raw_obs['state']}")
    print(f"prompt={raw_obs['prompt']!r}")
    print()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")

    print("--- loading JAX policy ---")
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    print("--- loading PyTorch policy ---")
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)
    print()

    captured: dict = {}
    _wrap_capture_raw(jax_policy, captured, "jax_raw")
    _wrap_capture_raw(pt_policy, captured, "pt_raw")

    # Use same noise for both: (action_horizon, action_dim) -> Policy broadcasts batch dim.
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, size=(cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
    print(f"noise shape={noise.shape}, mean={noise.mean():.4f}, std={noise.std():.4f}\n")

    # Warm-up both (esp. PT for torch.compile)
    print("Warming up PT (torch.compile may take ~90s) ...")
    _ = pt_policy.infer(raw_obs, noise=noise)
    print("Warmup done.\n")

    print("Running JAX inference ...")
    jax_out_dict = jax_policy.infer(raw_obs, noise=noise)
    print("Running PT inference ...")
    pt_out_dict = pt_policy.infer(raw_obs, noise=noise)
    print()

    jax_raw = captured["jax_raw"][0]  # (H, D)
    pt_raw = captured["pt_raw"][0]
    print(f"JAX raw model output: shape={jax_raw.shape} range=[{jax_raw.min():.3f}, {jax_raw.max():.3f}]")
    print(f"PT  raw model output: shape={pt_raw.shape} range=[{pt_raw.min():.3f}, {pt_raw.max():.3f}]")
    print()

    print("RAW model output  (before output transforms: AbsoluteActions, Unnormalize, OpenArmOutputs)")
    _stats("raw full chunk (50, 32)", jax_raw, pt_raw)
    _stats("raw first action (32)  ", jax_raw[0], pt_raw[0])
    _stats("raw first action [:16] ", jax_raw[0, :16], pt_raw[0, :16])
    print()

    jax_act = jax_out_dict["actions"]  # (H, 16) post-transform
    pt_act = pt_out_dict["actions"]
    print(f"JAX post-transform actions shape={jax_act.shape}")
    print(f"PT  post-transform actions shape={pt_act.shape}")
    print()

    print("POST-transform actions  (what SparkJAX consumes)")
    _stats("post full chunk", jax_act, pt_act)
    _stats("post first action", jax_act[0], pt_act[0])
    print()

    print("Per-horizon-step norms of raw output:")
    for h in [0, 5, 10, 20, 30, 49]:
        jn = float(np.linalg.norm(jax_raw[h]))
        pn = float(np.linalg.norm(pt_raw[h]))
        print(f"  h={h:2d}:  |JAX|={jn:.4f}  |PT|={pn:.4f}  ratio={pn / max(jn, 1e-12):.3f}")
    print()

    print("Per-horizon-step norms of post-transform action:")
    for h in [0, 5, 10, 20, 30, 49]:
        jn = float(np.linalg.norm(jax_act[h]))
        pn = float(np.linalg.norm(pt_act[h]))
        print(f"  h={h:2d}:  |JAX|={jn:.4f}  |PT|={pn:.4f}  ratio={pn / max(jn, 1e-12):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
