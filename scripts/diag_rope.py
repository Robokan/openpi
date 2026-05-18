#!/usr/bin/env python3
"""Check RoPE parameters: theta/base, head_dim, max_position_embeddings.
Also compute cos/sin at suffix positions (968-1017) and compare structure."""

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import torch
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config as _pc

cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
pt_policy = _pc.create_trained_policy(cfg, "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch", pytorch_device="cuda")
m = pt_policy._model

pg_lm = m.paligemma_with_expert.paligemma.language_model
expert_m = m.paligemma_with_expert.gemma_expert.model

print("=" * 60)
print("PaliGemma language model config")
print("=" * 60)
print(f"  hidden_size: {pg_lm.config.hidden_size}")
print(f"  num_heads: {pg_lm.config.num_attention_heads}")
print(f"  num_kv_heads: {pg_lm.config.num_key_value_heads}")
print(f"  head_dim: {pg_lm.config.head_dim}")
print(f"  max_position_embeddings: {pg_lm.config.max_position_embeddings}")
print(f"  rope_theta: {pg_lm.config.rope_theta}")
print(f"  rotary_emb type: {type(pg_lm.rotary_emb).__name__}")
print(f"  rotary_emb attrs: {[a for a in dir(pg_lm.rotary_emb) if not a.startswith('_')][:10]}")

print()
print("=" * 60)
print("Gemma Expert config")
print("=" * 60)
print(f"  hidden_size: {expert_m.config.hidden_size}")
print(f"  num_heads: {expert_m.config.num_attention_heads}")
print(f"  num_kv_heads: {expert_m.config.num_key_value_heads}")
print(f"  head_dim: {expert_m.config.head_dim}")
print(f"  max_position_embeddings: {expert_m.config.max_position_embeddings}")
print(f"  rope_theta: {expert_m.config.rope_theta}")
print(f"  rotary_emb type: {type(expert_m.rotary_emb).__name__}")

print()
print("=" * 60)
print("Compute cos/sin at suffix positions 968-1017")
print("=" * 60)

# Test the rotary embedding
positions = torch.arange(968, 1018, device="cuda").unsqueeze(0)  # (1, 50)
dummy = torch.zeros(1, 50, pg_lm.config.head_dim, device="cuda", dtype=torch.bfloat16)
cos_pg, sin_pg = pg_lm.rotary_emb(dummy, positions)
print(f"  paligemma cos shape: {cos_pg.shape}  dtype: {cos_pg.dtype}")
print(f"  paligemma cos at pos 968 [:8]: {cos_pg[0, 0, :8].float().cpu().numpy()}")
print(f"  paligemma sin at pos 968 [:8]: {sin_pg[0, 0, :8].float().cpu().numpy()}")
print(f"  paligemma cos at pos 1000 [:8]: {cos_pg[0, 32, :8].float().cpu().numpy()}")

# Now manually compute JAX-style RoPE
print()
print("JAX-style RoPE at pos 968:")
D = pg_lm.config.head_dim
freq_exp = (2.0 / D) * np.arange(D // 2, dtype=np.float32)
timescale = 10000.0 ** freq_exp
pos_968 = 968.0
radians = pos_968 / timescale
jax_cos = np.cos(radians)
jax_sin = np.sin(radians)
print(f"  JAX-style cos[:8]: {jax_cos[:8]}")
print(f"  JAX-style sin[:8]: {jax_sin[:8]}")

# Compare: PT cos has shape (1, 50, head_dim=256). PT cos = concat([cos(angle), cos(angle)], dim=-1). So PT cos[..., :D//2] == cos(angle).
# JAX cos has shape (1, 50, 1, D//2) (after newaxis). cos[..., :] = cos(angle).
print()
print("Diff between PT cos[:D//2] and JAX-style cos:")
pt_cos_half = cos_pg[0, 0, : D // 2].float().cpu().numpy()
print(f"  max|diff|: {float(np.max(np.abs(pt_cos_half - jax_cos))):.9f}")
