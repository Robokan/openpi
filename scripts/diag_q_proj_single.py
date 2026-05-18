#!/usr/bin/env python3
"""Compare JAX's q_einsum (with LoRA at runtime) vs PT's q_proj (with merged
weight) for a SINGLE attention layer. Same input.

If JAX != PT at this single matmul, the LoRA application math/storage is wrong.
If JAX == PT, the bug is downstream (attention compute, layernorm, etc.)
"""
import sys, numpy as np, torch
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

def _cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).flatten(), np.asarray(b, dtype=np.float64).flatten()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    from openpi.models import model as _model
    from flax import traverse_util as traversals
    import jax.numpy as jnp

    params = _model.restore_params(
        "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999/params/",
        restore_type=np.ndarray, dtype="float32",
    )
    pg = traversals.flatten_dict(params["PaliGemma"], sep="/")

    # Pick: paligemma layer 0 q_einsum
    L = 0
    w  = pg["llm/layers/attn/q_einsum/w"][L]       # (N=8, D=2048, H=256), fp32
    la = pg["llm/layers/attn/q_einsum/lora_a"][L]  # (N=8, D=2048, L=16)
    lb = pg["llm/layers/attn/q_einsum/lora_b"][L]  # (N=8, L=16, H=256)
    scaling = 1.0

    # Random input
    np.random.seed(42)
    x_np = np.random.randn(1, 4, 2048).astype(np.float32)

    # === JAX-style fp32 (gold standard) ===
    base = np.einsum("BTD,NDH->BTNH", x_np, w)
    lora_int = np.einsum("BTD,NDL->BTNL", x_np, la)
    lora_out = np.einsum("BTNL,NLH->BTNH", lora_int, lb)
    jax_fp32 = base + scaling * lora_out  # (1, 4, 8, 256)

    # === JAX-style bf16 (simulating JAX runtime) ===
    def _bf(arr):
        return torch.from_numpy(arr).to(torch.bfloat16).to(torch.float32).numpy()
    x_bf = _bf(x_np); w_bf = _bf(w); la_bf = _bf(la); lb_bf = _bf(lb)
    base_bf = np.einsum("BTD,NDH->BTNH", x_bf, w_bf)
    lora_int_bf = np.einsum("BTD,NDL->BTNL", x_bf, la_bf)
    lora_out_bf = np.einsum("BTNL,NLH->BTNH", lora_int_bf, lb_bf)
    jax_bf16 = base_bf + scaling * lora_out_bf

    # === PT-style merged (gold standard) ===
    merged_w = w + scaling * np.matmul(la, lb)
    pt_fp32 = np.einsum("BTD,NDH->BTNH", x_np, merged_w)

    # === PT-style merged + bf16 cast ===
    merged_w_bf = _bf(merged_w)
    pt_bf16 = np.einsum("BTD,NDH->BTNH", x_bf, merged_w_bf)

    print("=" * 100)
    print("Single q_einsum (paligemma layer 0) parity test, scaling=1.0")
    print("=" * 100)
    print(f"  jax_fp32 norm:  {np.linalg.norm(jax_fp32):.6f}")
    print(f"  pt_fp32 norm:   {np.linalg.norm(pt_fp32):.6f}")
    print(f"  jax_bf16 norm:  {np.linalg.norm(jax_bf16):.6f}")
    print(f"  pt_bf16 norm:   {np.linalg.norm(pt_bf16):.6f}")
    print()
    print(f"  cos(jax_fp32, pt_fp32)   = {_cos(jax_fp32, pt_fp32):+.7f}  ratio={np.linalg.norm(pt_fp32)/np.linalg.norm(jax_fp32):.6f}")
    print(f"  cos(jax_bf16, pt_bf16)   = {_cos(jax_bf16, pt_bf16):+.7f}  ratio={np.linalg.norm(pt_bf16)/np.linalg.norm(jax_bf16):.6f}")
    print(f"  cos(jax_fp32, pt_bf16)   = {_cos(jax_fp32, pt_bf16):+.7f}  ratio={np.linalg.norm(pt_bf16)/np.linalg.norm(jax_fp32):.6f}")
    print(f"  cos(jax_bf16, pt_fp32)   = {_cos(jax_bf16, pt_fp32):+.7f}  ratio={np.linalg.norm(pt_fp32)/np.linalg.norm(jax_bf16):.6f}")
    print()
    print("Conclusion:")
    print(f"  If jax_fp32 == pt_fp32 (cos=1.0, ratio=1.0): merge is mathematically equivalent to runtime LoRA application.")
    print(f"  If jax_bf16 vs pt_bf16 differ: bf16 quantization handles merged_w differently than separate w + lora.")

    # Also try gemma_expert q_einsum (rank=32)
    print()
    print("=" * 100)
    print("Single q_einsum_1 (gemma_expert layer 0) parity test, scaling=1.0")
    print("=" * 100)
    w  = pg["llm/layers/attn/q_einsum_1/w"][L]       # (N=8, D=1024, H=256)
    la = pg["llm/layers/attn/q_einsum_1/lora_a"][L]  # (N=8, D=1024, L=32)
    lb = pg["llm/layers/attn/q_einsum_1/lora_b"][L]  # (N=8, L=32, H=256)
    x_np = np.random.randn(1, 4, 1024).astype(np.float32)

    base = np.einsum("BTD,NDH->BTNH", x_np, w)
    lora_int = np.einsum("BTD,NDL->BTNL", x_np, la)
    lora_out = np.einsum("BTNL,NLH->BTNH", lora_int, lb)
    jax_fp32 = base + 1.0 * lora_out
    merged_w = w + 1.0 * np.matmul(la, lb)
    pt_fp32 = np.einsum("BTD,NDH->BTNH", x_np, merged_w)

    print(f"  jax_fp32 norm:  {np.linalg.norm(jax_fp32):.6f}")
    print(f"  pt_fp32 norm:   {np.linalg.norm(pt_fp32):.6f}")
    print(f"  cos(jax_fp32, pt_fp32) = {_cos(jax_fp32, pt_fp32):+.7f}  ratio={np.linalg.norm(pt_fp32)/np.linalg.norm(jax_fp32):.6f}")


if __name__ == "__main__":
    main()
