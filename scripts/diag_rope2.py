#!/usr/bin/env python3
"""Dump PT inv_freq, compare to JAX-style. Check if rope dims differ."""
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")
import torch, numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config as _pc

cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
pt = _pc.create_trained_policy(cfg, "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch", pytorch_device="cuda")
m = pt._model
pg_lm = m.paligemma_with_expert.paligemma.language_model

re = pg_lm.rotary_emb
print(f"rotary_emb type: {type(re).__name__}")
print(f"inv_freq shape: {re.inv_freq.shape}")
print(f"inv_freq dtype: {re.inv_freq.dtype}")
print(f"inv_freq[:8]: {re.inv_freq[:8].cpu().numpy()}")
print(f"inv_freq[-4:]: {re.inv_freq[-4:].cpu().numpy()}")
print()

# JAX-style equivalent
D = pg_lm.config.head_dim
print(f"head_dim from config: {D}")
freq_exp = (2.0 / D) * np.arange(D // 2, dtype=np.float32)
jax_inv_freq = 1.0 / (10000.0 ** freq_exp)
print(f"JAX-style inv_freq[:8]: {jax_inv_freq[:8]}")
print(f"JAX-style inv_freq[-4:]: {jax_inv_freq[-4:]}")
print()
print(f"inv_freq shapes: PT={re.inv_freq.shape} vs JAX=({D//2},)")
print(f"max|diff|: {float(np.max(np.abs(re.inv_freq.cpu().numpy() - jax_inv_freq))):.9f}")
print()

# Check if perhaps the head_dim used for rope is different than 256
# Let's compute inv_freq manually with different possible head_dims
for d in [64, 128, 192, 256, 512]:
    fe = (2.0 / d) * np.arange(d // 2, dtype=np.float32)
    inv = 1.0 / (10000.0 ** fe)
    # Find which one matches PT
    if len(re.inv_freq) == d // 2:
        diff = float(np.max(np.abs(re.inv_freq.cpu().numpy() - inv)))
        print(f"  if rope dim were {d}: shape matches PT. max diff = {diff:.9f}")
    else:
        print(f"  if rope dim were {d}: shape (D/2)={d//2} does NOT match PT shape {re.inv_freq.shape[0]}")

# Direct sample cos/sin
positions = torch.tensor([[968]], device="cuda")
dummy = torch.zeros(1, 1, D, device="cuda", dtype=torch.bfloat16)

# Manually compute what HF rotary_emb does
inv_freq_expanded = re.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1).to(positions.device)
position_ids_expanded = positions[:, None, :].float()
print(f"inv_freq_expanded shape: {inv_freq_expanded.shape}")
print(f"position_ids_expanded shape: {position_ids_expanded.shape}")
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
print(f"freqs shape: {freqs.shape}")
print(f"freqs[0, 0, :8] (= inv_freq[:8] * pos): {freqs[0, 0, :8].cpu().numpy()}")
print(f"Expected: inv_freq[:8] * 968 = {(re.inv_freq[:8].float().cpu().numpy() * 968.0)}")

cos_manual = freqs.cos()
print(f"cos_manual[0, 0, :8]: {cos_manual[0, 0, :8].cpu().numpy()}")

cos, sin = re(dummy, positions)
print(f"\nre output cos[0, 0, :8]: {cos[0, 0, :8].float().cpu().numpy()}")
print(f"re output sin[0, 0, :8]: {sin[0, 0, :8].float().cpu().numpy()}")
print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
print(f"\nAre cos_manual[..., :128] and re cos[..., :128] equal?")
diff = (cos_manual[0, 0, :128].cpu().numpy() - cos[0, 0, :128].float().cpu().numpy())
print(f"  max|diff|: {float(np.max(np.abs(diff))):.9f}")
print(f"  first 8 diffs: {diff[:8]}")

# JAX-style manual at pos 968
radians = 968.0 / (10000.0 ** ((2.0 / D) * np.arange(D // 2, dtype=np.float32)))
print(f"\nJAX-style cos at pos 968 first 8: {np.cos(radians[:8])}")
print(f"JAX-style sin at pos 968 first 8: {np.sin(radians[:8])}")
