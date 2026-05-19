#!/usr/bin/env python3
"""Compare JAX `_apply_rope` vs PT `apply_rotary_pos_emb` output for identical inputs.

If JAX RoPE != PT RoPE for the same positions, RoPE is the source of divergence.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    import jax, jax.numpy as jnp, torch
    from openpi.models import gemma as jax_gemma
    from transformers.models.gemma import modeling_gemma

    # Random Q, K with the same shape used in PaliGemma layer 0
    np.random.seed(42)
    B, T, H, D = 1, 16, 8, 256
    q_np = np.random.randn(B, T, H, D).astype(np.float32)
    positions_np = np.arange(T).astype(np.int32)[None]  # (B, T)

    # === JAX ===
    q_jax = jnp.asarray(q_np)
    pos_jax = jnp.asarray(positions_np)
    q_jax_rope = jax_gemma._apply_rope(q_jax, positions=pos_jax)
    q_jax_rope_np = np.asarray(q_jax_rope.astype(jnp.float32))
    print(f"JAX  RoPE: norm={np.linalg.norm(q_jax_rope_np):.4f}, shape={q_jax_rope_np.shape}")

    # === PT (HF GemmaRotaryEmbedding) ===
    # The PT RoPE forward returns cos/sin of shape (B, T, D), then
    # apply_rotary_pos_emb multiplies into (B, H, T, D).
    # We need to instantiate GemmaRotaryEmbedding with a config.
    class _C:
        head_dim = D
        hidden_size = D * H
        num_attention_heads = H
        max_position_embeddings = 65536
        rope_theta = 10000.0
        rope_scaling = None
    config = _C()
    rotary = modeling_gemma.GemmaRotaryEmbedding(config=config)

    # PT expects Q shape (B, H, T, D), positions shape (B, T)
    q_pt = torch.from_numpy(q_np).permute(0, 2, 1, 3).contiguous()  # (B, H, T, D)
    pos_pt = torch.from_numpy(positions_np.astype(np.int64))  # (B, T)
    dummy = torch.zeros(B, T, D)
    cos, sin = rotary(dummy, pos_pt)
    # Apply RoPE only on Q (K unused, pass q twice)
    q_pt_rope, _ = modeling_gemma.apply_rotary_pos_emb(q_pt, q_pt, cos, sin, unsqueeze_dim=1)
    # Transpose back to (B, T, H, D) for comparison with JAX
    q_pt_rope_np = q_pt_rope.permute(0, 2, 1, 3).contiguous().numpy()
    print(f"PT   RoPE: norm={np.linalg.norm(q_pt_rope_np):.4f}, shape={q_pt_rope_np.shape}")

    print()
    print(f"  cos(jax, pt) = {_cos(q_jax_rope_np, q_pt_rope_np):+.7f}")
    print(f"  max|diff|    = {float(np.max(np.abs(q_jax_rope_np - q_pt_rope_np))):.6f}")
    print(f"  ratio (pt/jax) = {np.linalg.norm(q_pt_rope_np)/np.linalg.norm(q_jax_rope_np):.6f}")

    # Check IF PT uses different ordering convention
    # JAX: x1 = first half, x2 = second half, output_first = x1*cos - x2*sin, output_second = x2*cos + x1*sin
    # PT (interleaved variant exists in some impls): could be different
    print()
    print("== Detailed first-position check ==")
    print(f"  JAX position 0, head 0, first 8 dims: {q_jax_rope_np[0, 0, 0, :8]}")
    print(f"  PT  position 0, head 0, first 8 dims: {q_pt_rope_np[0, 0, 0, :8]}")
    print(f"  JAX position 5, head 0, first 8 dims: {q_jax_rope_np[0, 5, 0, :8]}")
    print(f"  PT  position 5, head 0, first 8 dims: {q_pt_rope_np[0, 5, 0, :8]}")


if __name__ == "__main__":
    main()
