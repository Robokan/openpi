#!/usr/bin/env python3
"""Compare JAX policy with LoRA weights zeroed to PyTorch policy with LoRA
never merged (BROKEN_NO_LORA). Both use:
   - The same trained action expert / action_in/out_proj / time_mlp
   - The same base PaliGemma weights (no LoRA delta applied)

If JAX-no-LoRA matches PT-no-LoRA, the LoRA application path is the
source of divergence and the Gemma backbones agree. If they still
diverge, the divergence is in the (non-LoRA) architecture itself
(Gemma forward pass / action expert / etc.), which would be a parity
bug in pi0_pytorch.py independent of LoRA.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import safetensors.torch as _st
import torch as _torch

import openpi.models_pytorch.pi0_pytorch as _pi0pt
from openpi.policies import policy_config as _pc
from openpi.training import config as _config

CAMERAS = ["ego", "left_wrist", "right_wrist"]
CAMERA_MAP = {"ego": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}


def _patch_load_pytorch():
    original = _pi0pt.PI0Pytorch  # noqa: F841

    def _load(self, train_config, weight_path):
        model = _pi0pt.PI0Pytorch(config=train_config.model)
        sd = _st.load_file(weight_path)
        model.load_state_dict(sd, strict=False)
        return model

    from openpi.models import pi0_config as _pi0c
    _pi0c.Pi0Config.load_pytorch = _load


_patch_load_pytorch()


def _load_tasks(p):
    m = {}
    if p.exists():
        with open(p) as f:
            for line in f:
                e = json.loads(line.strip())
                m[e["task_index"]] = e["task"]
    return m


def _ffmpeg_frame(vp, idx):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        name = tmp.name
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp),
               "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-f", "image2", name]
        r = subprocess.run(cmd, capture_output=True, timeout=15)
        if r.returncode != 0:
            return None
        f = cv2.imread(name)
        return None if f is None else cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    finally:
        if os.path.exists(name):
            try:
                os.unlink(name)
            except OSError:
                pass


def load_real_samples(local_dir, n):
    tasks = _load_tasks(local_dir / "meta" / "tasks.jsonl")
    parquets = sorted(local_dir.glob("data/chunk-*/episode_*.parquet"))
    samples = []
    step = max(1, len(parquets) // max(n, 1))
    for pq_path in parquets[::step]:
        if len(samples) >= n:
            break
        t = pq.read_table(pq_path)
        if t.num_rows == 0:
            continue
        row = t.num_rows // 2
        state = np.array(t.column("observation.state")[row].as_py(), dtype=np.float32)
        fidx = int(t.column("frame_index")[row].as_py())
        tidx = int(t.column("task_index")[row].as_py())
        prompt = tasks.get(tidx, "pick up the object")
        chunk, ep = pq_path.parent.name, pq_path.stem
        imgs, ok = {}, True
        for cam in CAMERAS:
            vp = local_dir / "videos" / chunk / f"observation.images.{cam}" / f"{ep}.mp4"
            if not vp.exists():
                ok = False
                break
            f = _ffmpeg_frame(vp, fidx)
            if f is None:
                ok = False
                break
            if f.shape[:2] != (224, 224):
                f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA)
            imgs[CAMERA_MAP[cam]] = np.transpose(f, (2, 0, 1)).astype(np.uint8)
        if ok:
            samples.append({"state": state, "images": imgs, "prompt": prompt})
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_openarm_ngc_lora_v4")
    p.add_argument("--jax-ckpt", default="/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999")
    p.add_argument("--pt-ckpt-no-lora", default="/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_BROKEN_NO_LORA")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    A_H = cfg.model.action_horizon
    A_D = cfg.model.action_dim
    print(f"Config={args.config}  action_horizon={A_H}  action_dim={A_D}")

    local_dir = Path(cfg.data.base_config.local_dir)
    samples = load_real_samples(local_dir, args.n)
    if not samples:
        print("ERROR: no samples loaded")
        return 1

    rng = np.random.default_rng(args.seed)
    noises = [rng.standard_normal((A_H, A_D)).astype(np.float32) for _ in samples]

    # ----------------- JAX with LoRA zeroed out -----------------
    print("\n=== Loading JAX policy (LoRA weights about to be zeroed) ===")
    print(f"  ckpt: {args.jax_ckpt}")
    jax_policy = _pc.create_trained_policy(cfg, args.jax_ckpt)

    # Reach into the underlying JAX nnx module and zero out lora_a/lora_b.
    # Approach: walk the nnx state and zero any leaf whose flax-style name contains "lora_a" or "lora_b".
    n_zeroed = 0

    def _zero_lora(obj, path=""):
        nonlocal n_zeroed
        # Skip primitives
        if obj is None:
            return obj
        # nnx Variable / jax.Array detection by attribute
        try:
            import jax.numpy as jnp
            if hasattr(obj, "value") and not isinstance(obj, (dict, list, tuple)):
                v = obj.value
                # jnp arrays only
                if hasattr(v, "shape") and hasattr(v, "dtype"):
                    if "lora_a" in path or "lora_b" in path:
                        obj.value = jnp.zeros_like(v)
                        n_zeroed += 1
                        return obj
        except Exception:
            pass
        # Recurse
        if isinstance(obj, dict):
            for k, v in obj.items():
                _zero_lora(v, f"{path}/{k}")
        elif hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                if k.startswith("_"):
                    continue
                _zero_lora(v, f"{path}/{k}")
        return obj

    _zero_lora(jax_policy._model)  # noqa: SLF001
    print(f"  -> zeroed {n_zeroed} LoRA leaf tensors")

    # Now run the policy on each sample
    jax_actions = []
    for i, (s, ns) in enumerate(zip(samples, noises)):
        result = jax_policy.infer(s, noise=ns)
        a = np.asarray(result["actions"])
        jax_actions.append(a)
        ms = result.get("policy_timing", {}).get("infer_ms", -1)
        print(f"  JAX-no-LoRA [{i}] shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] infer={ms:.1f}ms")

    # Release JAX memory
    del jax_policy
    try:
        import jax as _jax
        _jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    try:
        _torch.cuda.empty_cache()
    except Exception:
        pass

    # ----------------- PyTorch no-LoRA -----------------
    print("\n=== Loading PyTorch policy (no-LoRA checkpoint) ===")
    print(f"  ckpt: {args.pt_ckpt_no_lora}")
    pt_policy = _pc.create_trained_policy(cfg, args.pt_ckpt_no_lora)
    pt_actions = []
    for i, (s, ns) in enumerate(zip(samples, noises)):
        result = pt_policy.infer(s, noise=ns)
        a = np.asarray(result["actions"])
        pt_actions.append(a)
        ms = result.get("policy_timing", {}).get("infer_ms", -1)
        print(f"  PT-no-LoRA  [{i}] shape={a.shape} range=[{a.min():.3f},{a.max():.3f}] infer={ms:.1f}ms")

    # ----------------- Compare -----------------
    print("\n=== JAX-no-LoRA vs PT-no-LoRA  ===")
    print("Both policies use base PaliGemma weights + the same trained action expert + projections.")
    print("Both fed the same observation and the same fp32 noise.")
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    cos_overall = []
    cos_delta_overall = []
    for i, (aj, ap) in enumerate(zip(jax_actions, pt_actions)):
        state = samples[i]["state"][:16]
        d = np.abs(aj - ap)
        first_d = d[0]
        jf = aj.reshape(-1).astype(np.float64)
        pf = ap.reshape(-1).astype(np.float64)
        cos_abs = float(np.dot(jf, pf) / (np.linalg.norm(jf)*np.linalg.norm(pf) + 1e-12))
        cos_overall.append(cos_abs)

        # First-step deltas relative to state
        jd = (aj[0, :16] - state).astype(np.float64)
        pd = (ap[0, :16] - state).astype(np.float64)
        cos_d = float(np.dot(jd, pd) / (np.linalg.norm(jd)*np.linalg.norm(pd) + 1e-12))
        cos_delta_overall.append(cos_d)

        print(f"\n--- Sample {i}   prompt={samples[i]['prompt']!r}")
        print(f"  state (first 16):  {state}")
        print(f"  JAX-no-LoRA first action: {aj[0, :16]}")
        print(f"  PT-no-LoRA  first action: {ap[0, :16]}")
        print(f"  JAX delta from state:    {(aj[0, :16] - state)}")
        print(f"  PT  delta from state:    {(ap[0, :16] - state)}")
        print(f"  max|diff|={d.max():.4f}  mean={d.mean():.4f}")
        print(f"  cos(abs)   = {cos_abs:.6f}")
        print(f"  cos(delta) = {cos_d:.6f}    <- direction of intended motion agreement")
        print(f"  ||JAX_delta||={np.linalg.norm(aj[0, :16] - state):.4f}    "
              f"||PT_delta||={np.linalg.norm(ap[0, :16] - state):.4f}")

    print("\n=== AGGREGATE (no-LoRA case) ===")
    print(f"  cos(abs)   mean={np.mean(cos_overall):.6f}  min={np.min(cos_overall):.6f}")
    print(f"  cos(delta) mean={np.mean(cos_delta_overall):.6f}  min={np.min(cos_delta_overall):.6f}")
    print()
    print("  Interpretation:")
    print("    - cos(delta) ~= 1.0 in no-LoRA case   -> LoRA is THE bug (architectures agree)")
    print("    - cos(delta) << 1.0 in no-LoRA case  -> pi0_pytorch backbone has a parity bug independent of LoRA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
