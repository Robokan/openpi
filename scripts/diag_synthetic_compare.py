#!/usr/bin/env python3
"""Compare JAX vs PyTorch on synthetic inputs for an arbitrary pi05 config.

This is a CONTROL test: we use pi05_libero (or similar) which NVIDIA's
tutorial validates as "working in PyTorch + TensorRT". We compare the
PyTorch output to the JAX output on the SAME synthetic observation.

  If JAX vs PT match closely (cos > 0.99, max|diff| < 0.05):
    -> pi0_pytorch is correct for THIS config and the divergence we
       see on our LoRA + discrete-state-input OpenArm config is
       specific to that path.

  If JAX vs PT diverge for pi05_libero too:
    -> pi0_pytorch has a generic parity bug that NVIDIA's tutorial
       did not catch (because they never compare PT vs JAX,
       only PT vs TRT to validate quantization fidelity).
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

from openpi.policies import policy_config as _pc  # noqa: E402
from openpi.training import config as _config     # noqa: E402


def get_obs_spec_for_config(cfg_name: str):
    """Return what keys the dataset transform expects, plus state dim and prompt."""
    if cfg_name == "pi05_libero":
        # LiberoInputs reads "observation/image", "observation/wrist_image",
        # "observation/state". Images shape (224, 224, 3) uint8.
        return {
            "image_keys": ["observation/image", "observation/wrist_image"],
            "state_dim": 8,
            "prompt": "pick up the alphabet soup and place it in the basket",
        }
    elif cfg_name == "pi05_droid":
        # DroidInputs reads observation.exterior_image_1_left, etc.
        return {
            "image_keys": ["observation/exterior_image_1_left",
                           "observation/exterior_image_2_left",
                           "observation/wrist_image_left"],
            "state_dim": 8,
            "prompt": "pick up the red cup",
        }
    else:
        raise ValueError(f"unsupported config for this synthetic test: {cfg_name}")


def make_synthetic_samples(spec, prompts, seed=0, mode="zeros"):
    """mode: 'zeros' for deterministic zero inputs (best for parity check),
            'noise' for random plausible-looking inputs (closer to deployment).
    """
    rng = np.random.default_rng(seed)
    samples = []
    for pr in prompts:
        d = {}
        for k in spec["image_keys"]:
            if mode == "zeros":
                d[k] = np.full((224, 224, 3), 128, dtype=np.uint8)  # neutral gray
            else:
                d[k] = (rng.normal(127, 30, size=(224, 224, 3)).clip(0, 255)).astype(np.uint8)
        if mode == "zeros":
            d["observation/state"] = np.zeros(spec["state_dim"], dtype=np.float32)
        else:
            d["observation/state"] = rng.normal(0, 0.1, size=spec["state_dim"]).astype(np.float32).clip(-1, 1)
        d["prompt"] = pr
        samples.append(d)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--jax-ckpt", default="/openpi_assets/openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--pt-ckpt",  default="/openpi_assets/openpi-assets/checkpoints/pi05_libero_pytorch")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mode", choices=("zeros", "noise"), default="zeros",
                   help="zeros = neutral inputs (best for parity), noise = random plausible inputs")
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    A_H = cfg.model.action_horizon
    A_D = cfg.model.action_dim
    print(f"Config={args.config}  pi05={cfg.model.pi05}  discrete_state_input={cfg.model.discrete_state_input}")
    print(f"  action_horizon={A_H}  action_dim={A_D}  variant={cfg.model.paligemma_variant}")

    spec = get_obs_spec_for_config(args.config)
    prompts = [spec["prompt"], "do the task"][:args.n]
    samples = make_synthetic_samples(spec, prompts, seed=args.seed, mode=args.mode)
    print(f"  input mode = {args.mode}")
    rng = np.random.default_rng(args.seed + 100)
    noises = [rng.standard_normal((A_H, A_D)).astype(np.float32) for _ in samples]

    # ----- JAX -----
    print("\n=== Loading JAX policy ===")
    jax_policy = _pc.create_trained_policy(cfg, args.jax_ckpt)
    jax_actions = []
    for i, (s, n) in enumerate(zip(samples, noises)):
        r = jax_policy.infer(s, noise=n)
        a = np.asarray(r["actions"])
        jax_actions.append(a)
        print(f"  JAX [{i}] shape={a.shape}  range=[{a.min():.3f},{a.max():.3f}]"
              f"  infer={r.get('policy_timing', {}).get('infer_ms', -1):.1f}ms")

    del jax_policy
    try:
        import jax as _jax
        _jax.clear_caches()
    except Exception:
        pass
    gc.collect()

    # ----- PT -----
    print("\n=== Loading PyTorch policy ===")
    pt_policy = _pc.create_trained_policy(cfg, args.pt_ckpt)
    pt_actions = []
    for i, (s, n) in enumerate(zip(samples, noises)):
        r = pt_policy.infer(s, noise=n)
        a = np.asarray(r["actions"])
        pt_actions.append(a)
        print(f"  PT  [{i}] shape={a.shape}  range=[{a.min():.3f},{a.max():.3f}]"
              f"  infer={r.get('policy_timing', {}).get('infer_ms', -1):.1f}ms")

    # ----- Compare -----
    print(f"\n=== JAX vs PT  ({args.config}) ===")
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    cos_all, max_abs_all = [], []
    for i, (aj, ap) in enumerate(zip(jax_actions, pt_actions)):
        d = np.abs(aj - ap)
        jf, pf = aj.reshape(-1).astype(np.float64), ap.reshape(-1).astype(np.float64)
        cos = float(np.dot(jf, pf) / (np.linalg.norm(jf) * np.linalg.norm(pf) + 1e-12))
        cos_all.append(cos)
        max_abs_all.append(float(d.max()))
        print(f"\n--- sample {i}  prompt={samples[i]['prompt']!r}")
        print(f"  JAX first action: {aj[0]}")
        print(f"  PT  first action: {ap[0]}")
        print(f"  diff first step:  {aj[0] - ap[0]}")
        print(f"  cos(abs)   = {cos:.6f}")
        print(f"  max|diff|  = {d.max():.4f}  mean|diff|={d.mean():.4f}")
        # Per-joint look
        print(f"  per-joint|diff| first step: {np.abs(aj[0] - ap[0])}")

    print("\n=== AGGREGATE ===")
    print(f"  cos(abs):   mean={np.mean(cos_all):.6f}  min={np.min(cos_all):.6f}")
    print(f"  max|diff|:  worst={max(max_abs_all):.4f}")
    if min(cos_all) > 0.99 and max(max_abs_all) < 0.1:
        print(f"\n  CONCLUSION: pi0_pytorch is correct for {args.config}.")
        print( "             The divergence we see on pi05_openarm_ngc_lora_v4 is specific")
        print( "             to LoRA + discrete-state-input path.")
    else:
        print(f"\n  CONCLUSION: pi0_pytorch diverges from JAX even for {args.config}.")
        print( "             This is a generic parity bug NVIDIA's tutorial did not catch")
        print( "             (they only compare PT vs TRT, never PT vs JAX).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
