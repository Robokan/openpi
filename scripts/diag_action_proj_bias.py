#!/usr/bin/env python3
"""Compare action_out_proj weight AND bias values between JAX and PT.
Also check what suffix_out -> v_t gives for IDENTICAL inputs."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import jax.numpy as jnp
import torch

from openpi.training import config as _config
from openpi.policies import policy_config as _pc

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"

cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
print("Loading both policies ...")
jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)

jax_model = jax_policy._model
pt_model = pt_policy._model

# Extract action_out_proj weights and biases
import flax.nnx as nnx
jax_aop = jax_model.action_out_proj
print(f"JAX action_out_proj: dir = {[a for a in dir(jax_aop) if not a.startswith('_')]}")
# nnx.Linear stores kernel + bias
jax_w = np.asarray(jax_aop.kernel.value.astype(jnp.float32))
jax_b = np.asarray(jax_aop.bias.value.astype(jnp.float32))
print(f"JAX action_out_proj.kernel: shape={jax_w.shape}  norm={np.linalg.norm(jax_w):.6f}")
print(f"JAX action_out_proj.bias  : shape={jax_b.shape}  norm={np.linalg.norm(jax_b):.6f}")
print(f"JAX bias values [:16]: {jax_b[:16]}")
print()

pt_w = pt_model.action_out_proj.weight.detach().float().cpu().numpy()
pt_b = pt_model.action_out_proj.bias.detach().float().cpu().numpy()
print(f"PT  action_out_proj.weight: shape={pt_w.shape}  norm={np.linalg.norm(pt_w):.6f}")
print(f"PT  action_out_proj.bias  : shape={pt_b.shape}  norm={np.linalg.norm(pt_b):.6f}")
print(f"PT  bias values [:16]: {pt_b[:16]}")
print()

# JAX nnx.Linear stores kernel as (in, out). PT stores weight as (out, in).
print("Comparing weights (after transposing PT to match JAX):")
pt_w_for_compare = pt_w.T  # (in=1024, out=32) to match JAX
cos_w = float((jax_w.flatten() @ pt_w_for_compare.flatten()) / (np.linalg.norm(jax_w) * np.linalg.norm(pt_w_for_compare) + 1e-12))
print(f"  weight cos={cos_w:.9f}  max|diff|={float(np.max(np.abs(jax_w - pt_w_for_compare))):.6f}")
print()

print("Comparing biases:")
cos_b = float((jax_b @ pt_b) / (np.linalg.norm(jax_b) * np.linalg.norm(pt_b) + 1e-12))
print(f"  bias cos={cos_b:.9f}  max|diff|={float(np.max(np.abs(jax_b - pt_b))):.9f}")
print(f"  bias per-dim diff: {jax_b - pt_b}")

# Now do similar check for action_in_proj
print("\n" + "=" * 80)
print("action_in_proj check")
print("=" * 80)
jax_aip = jax_model.action_in_proj
jax_aip_w = np.asarray(jax_aip.kernel.value.astype(jnp.float32))
jax_aip_b = np.asarray(jax_aip.bias.value.astype(jnp.float32))
pt_aip_w = pt_model.action_in_proj.weight.detach().float().cpu().numpy()
pt_aip_b = pt_model.action_in_proj.bias.detach().float().cpu().numpy()
print(f"JAX action_in_proj.bias norm={np.linalg.norm(jax_aip_b):.6f}")
print(f"PT  action_in_proj.bias norm={np.linalg.norm(pt_aip_b):.6f}")
print(f"  bias cos={float((jax_aip_b @ pt_aip_b) / (np.linalg.norm(jax_aip_b) * np.linalg.norm(pt_aip_b) + 1e-12)):.9f}")
print(f"  max|diff|={float(np.max(np.abs(jax_aip_b - pt_aip_b))):.9f}")

# And time_mlp
print("\n" + "=" * 80)
print("time_mlp_in / time_mlp_out check")
print("=" * 80)
for name in ["time_mlp_in", "time_mlp_out"]:
    jax_m = getattr(jax_model, name)
    pt_m = getattr(pt_model, name)
    jax_b_v = np.asarray(jax_m.bias.value.astype(jnp.float32))
    pt_b_v = pt_m.bias.detach().float().cpu().numpy()
    cos = float((jax_b_v @ pt_b_v) / (np.linalg.norm(jax_b_v) * np.linalg.norm(pt_b_v) + 1e-12))
    print(f"  {name}.bias cos={cos:.9f}  norm_jax={np.linalg.norm(jax_b_v):.6f}  norm_pt={np.linalg.norm(pt_b_v):.6f}  max|d|={float(np.max(np.abs(jax_b_v - pt_b_v))):.9f}")
