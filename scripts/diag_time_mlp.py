#!/usr/bin/env python3
"""Compare the time-MLP path used by pi05's AdaRMSNorm between JAX and PyTorch.

JAX:   sincos(t) -> time_mlp_in -> swish -> time_mlp_out -> swish -> adarms_cond
PT :   sincos(t) -> time_mlp_in -> silu  -> time_mlp_out -> silu  -> adarms_cond

If these match, the time-MLP -> adarms_cond chain is not the source of divergence.
"""

import math
import sys
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F

sys.path.insert(0, "/app/src")

from openpi.models import model as _model  # noqa: E402
from flax.nnx import traversals  # noqa: E402


def posemb_sincos_numpy(pos, dim, min_period=4e-3, max_period=4.0):
    """Replicates JAX posemb_sincos exactly."""
    fraction = np.linspace(0.0, 1.0, dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    radians = (1.0 / period * 2 * np.pi) * pos[:, None]   # (B, D/2)
    return np.concatenate([np.sin(radians), np.cos(radians)], axis=-1).astype(np.float32)


def main():
    jax_ckpt = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
    pt_ckpt  = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch/model.safetensors"

    # ---------- JAX side: pull time_mlp weights ----------
    params = _model.restore_params(jax_ckpt + "/params", restore_type=np.ndarray, dtype="float32")
    proj = params["projection_params"] if "projection_params" in params else params

    def get(k):
        # We pop down to .kernel/.bias which may be wrapped
        node = proj[k]
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        return node

    jax_proj_keys = list(params.keys())
    print(f"JAX params keys: {jax_proj_keys}")
    # The actual projection_params layout is just the names directly.
    in_w = np.asarray(params["time_mlp_in"]["kernel"])   # (in, out)
    in_b = np.asarray(params["time_mlp_in"]["bias"])     # (out,)
    out_w = np.asarray(params["time_mlp_out"]["kernel"])
    out_b = np.asarray(params["time_mlp_out"]["bias"])
    print(f"JAX time_mlp_in.kernel  shape={in_w.shape}  bias={in_b.shape}")
    print(f"JAX time_mlp_out.kernel shape={out_w.shape}  bias={out_b.shape}")

    width = in_w.shape[0]  # action_expert width
    print(f"action_expert width = {width}")

    # ---------- PT side: pull time_mlp weights from safetensors ----------
    sd = st.load_file(pt_ckpt)
    pt_keys = [k for k in sd.keys() if "time_mlp" in k]
    print(f"PT time_mlp keys: {pt_keys}")

    pt_in_w  = sd["time_mlp_in.weight"].float().numpy()     # (out, in)
    pt_in_b  = sd["time_mlp_in.bias"].float().numpy()
    pt_out_w = sd["time_mlp_out.weight"].float().numpy()
    pt_out_b = sd["time_mlp_out.bias"].float().numpy()

    print(f"PT time_mlp_in.weight shape={pt_in_w.shape}  bias={pt_in_b.shape}")
    print(f"PT time_mlp_out.weight shape={pt_out_w.shape} bias={pt_out_b.shape}")

    # Verify weight transposition: PT weight should equal JAX kernel.T
    if pt_in_w.shape != in_w.T.shape:
        print(f"!! shape mismatch: PT in.weight={pt_in_w.shape}  JAX in.kernel.T={in_w.T.shape}")
    else:
        diff = np.abs(pt_in_w - in_w.T).max()
        print(f"weight diff (in):  max|pt - jax.T|  = {diff:.6e}")
    diff = np.abs(pt_in_b - in_b).max()
    print(f"bias  diff (in):  max|pt - jax|      = {diff:.6e}")
    if pt_out_w.shape == out_w.T.shape:
        diff = np.abs(pt_out_w - out_w.T).max()
        print(f"weight diff (out): max|pt - jax.T|  = {diff:.6e}")
    diff = np.abs(pt_out_b - out_b).max()
    print(f"bias  diff (out): max|pt - jax|      = {diff:.6e}")

    # ---------- Run forward on representative timesteps ----------
    timesteps = np.array([1.0, 0.9, 0.5, 0.1, 0.0], dtype=np.float32)
    pos = posemb_sincos_numpy(timesteps, width)
    print(f"\nposemb_sincos: timesteps={timesteps.tolist()}  output shape={pos.shape}")
    print(f"posemb output range: [{pos.min():.4f}, {pos.max():.4f}]  mean={pos.mean():.4f}")

    # JAX-equivalent forward: out = swish(swish(pos @ in_w + in_b) @ out_w + out_b)
    h1 = pos @ in_w + in_b
    swish = lambda x: x / (1.0 + np.exp(-x))   # silu
    h1s = swish(h1)
    h2 = h1s @ out_w + out_b
    jax_out = swish(h2)

    # PT-equivalent forward using PT weight layout: y = x @ W.T + b
    h1_pt = pos @ pt_in_w.T + pt_in_b
    h1s_pt = swish(h1_pt)
    h2_pt = h1s_pt @ pt_out_w.T + pt_out_b
    pt_out = swish(h2_pt)

    print("\n=== JAX time_mlp out (per-timestep stats) ===")
    for i, t in enumerate(timesteps):
        v = jax_out[i]
        print(f"  t={t:.2f}  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}  ||v||={np.linalg.norm(v):.4f}")
    print("=== PT  time_mlp out (per-timestep stats) ===")
    for i, t in enumerate(timesteps):
        v = pt_out[i]
        print(f"  t={t:.2f}  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}  ||v||={np.linalg.norm(v):.4f}")

    diff_all = np.abs(jax_out - pt_out)
    print(f"\n=== Per-element diff JAX vs PT (after full time MLP) ===")
    print(f"  max|diff| = {diff_all.max():.6e}")
    print(f"  mean|diff| = {diff_all.mean():.6e}")
    # Sign agreement
    sg = (np.sign(jax_out) == np.sign(pt_out))
    print(f"  fraction of elements with same sign: {sg.mean():.4f}")
    # Cosine sim per timestep
    for i, t in enumerate(timesteps):
        a, b = jax_out[i].astype(np.float64), pt_out[i].astype(np.float64)
        cos = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
        print(f"  t={t:.2f}: cos = {cos:.8f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
