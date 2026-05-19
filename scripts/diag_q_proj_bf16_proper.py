#!/usr/bin/env python3
"""PROPERLY compare JAX-style runtime LoRA (with bf16 rounding of the
rank-16 intermediate) against PT-style pre-merged weight, for a single
attention layer. The previous diag_q_proj_single.py simulated bf16
inputs/weights but did NOT round the intermediate `lora_int` between
the two LoRA matmuls. That intermediate rounding is the suspected
source of the per-layer drift.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")


def _cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).flatten(), np.asarray(b, dtype=np.float64).flatten()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _bf(arr):
    """Round a numpy fp32 array to bf16 (and back to fp32 for numpy ops)."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16).to(torch.float32).numpy()


def _einsum_bf16(eqn, *args):
    """Simulate a bf16 matmul: bf16 inputs, fp32 accumulator (Tensor Core
    behavior), bf16 output."""
    args = [_bf(a) for a in args]
    out_fp32 = np.einsum(eqn, *args)
    return _bf(out_fp32)


def main():
    from openpi.models import model as _model
    from flax import traverse_util as traversals

    params = _model.restore_params(
        "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999/params/",
        restore_type=np.ndarray, dtype="float32",
    )
    pg = traversals.flatten_dict(params["PaliGemma"], sep="/")

    rng = np.random.default_rng(42)

    # ============================================================
    # Test all LoRA-targeted layers in paligemma layer 0
    # ============================================================
    print("=" * 100)
    print("PROPER bf16 single-matmul test: JAX runtime LoRA vs PT pre-merged")
    print("(JAX rounds rank-r intermediate to bf16 between LoRA mats; PT merges in fp32 then rounds final weight)")
    print("=" * 100)

    L = 0
    # name, stem, eqn_w, eqn_la, eqn_lb, x_shape_fn (given w)
    targets = [
        ("paligemma q (rank=16)", "llm/layers/attn/q_einsum",   "BTD,NDH->BTNH", "BTD,NDL->BTNL", "BTNL,NLH->BTNH", lambda w: (1, 4, w.shape[1])),
        ("paligemma k (rank=16)", "llm/layers/attn/kv_einsum",  "BTD,DH->BTH",   "BTD,DL->BTL",   "BTL,LH->BTH",    lambda w: (1, 4, w.shape[0])),
        ("paligemma v (rank=16)", "llm/layers/attn/kv_einsum",  "BTD,DH->BTH",   "BTD,DL->BTL",   "BTL,LH->BTH",    lambda w: (1, 4, w.shape[0])),
        ("gemma_expert q (rank=32)", "llm/layers/attn/q_einsum_1",   "BTD,NDH->BTNH", "BTD,NDL->BTNL", "BTNL,NLH->BTNH", lambda w: (1, 4, w.shape[1])),
        ("gemma_expert k (rank=32)", "llm/layers/attn/kv_einsum_1",  "BTD,DH->BTH",   "BTD,DL->BTL",   "BTL,LH->BTH",    lambda w: (1, 4, w.shape[0])),
        ("gemma_expert v (rank=32)", "llm/layers/attn/kv_einsum_1",  "BTD,DH->BTH",   "BTD,DL->BTL",   "BTL,LH->BTH",    lambda w: (1, 4, w.shape[0])),
    ]

    print(f"{'layer':40s} {'cos':>10s} {'ratio':>10s} {'max|err|':>12s} {'rel|err|':>12s}")
    print("-" * 90)

    for name, stem, eqn_w, eqn_la, eqn_lb, x_shape_fn in targets:
        try:
            w = pg[stem + "/w"]
        except KeyError:
            print(f"{name:40s} (skip, no LoRA: {stem}/w)")
            continue
        try:
            la = pg[stem + "/lora_a"]; lb = pg[stem + "/lora_b"]
        except KeyError:
            print(f"{name:40s} (no LoRA, base only)")
            continue

        # Pick the right layer index
        w = w[L]; la = la[L]; lb = lb[L]

        if "kv_einsum" in stem:
            # kv_einsum w shape: (N=2 [k,v], num_kv=1, D, H)
            n_idx = 0 if " k " in f" {name} " else 1
            w = w[n_idx, 0]; la = la[n_idx, 0]; lb = lb[n_idx, 0]
        # q_einsum: (N=8, D, H) — already correct after [L]

        x_shape = x_shape_fn(w)
        x = rng.standard_normal(x_shape, dtype=np.float32).astype(np.float32)

        # === JAX-style runtime LoRA in proper bf16 (with intermediate rounding) ===
        base_bf = _einsum_bf16(eqn_w, x, w)
        lora_int_bf = _einsum_bf16(eqn_la, x, la)  # rounded to bf16
        lora_out_bf = _einsum_bf16(eqn_lb, lora_int_bf, lb)  # rounded to bf16
        # Final add: bf16 + scaling * bf16 -> bf16
        jax_bf = _bf(base_bf + 1.0 * lora_out_bf)

        # === PT-style: merge in fp32, then bf16 weight, then bf16 matmul ===
        # np.matmul broadcasts over leading dims so works for q_einsum (N axis) too
        merged = w + 1.0 * np.matmul(la, lb)
        merged_bf = _bf(merged)
        pt_bf = _einsum_bf16(eqn_w, x, merged_bf)

        # === fp32 reference (gold) ===
        base_fp = np.einsum(eqn_w, x, w)
        lora_int_fp = np.einsum(eqn_la, x, la)
        lora_out_fp = np.einsum(eqn_lb, lora_int_fp, lb)
        ref_fp32 = base_fp + 1.0 * lora_out_fp

        cos = _cos(jax_bf, pt_bf)
        ratio = float(np.linalg.norm(pt_bf) / (np.linalg.norm(jax_bf) + 1e-30))
        diff = jax_bf - pt_bf
        max_err = float(np.max(np.abs(diff)))
        rel_err = float(np.linalg.norm(diff) / (np.linalg.norm(jax_bf) + 1e-30))

        # Also report each vs fp32 reference
        cos_jax_ref = _cos(jax_bf, ref_fp32)
        cos_pt_ref = _cos(pt_bf, ref_fp32)
        ratio_jax_ref = float(np.linalg.norm(jax_bf) / (np.linalg.norm(ref_fp32) + 1e-30))
        ratio_pt_ref = float(np.linalg.norm(pt_bf) / (np.linalg.norm(ref_fp32) + 1e-30))

        print(f"{name:40s} {cos:>+10.6f} {ratio:>10.4f} {max_err:>12.4e} {rel_err:>12.4e}")
        print(f"    jax_bf vs fp32_ref: cos={cos_jax_ref:+.7f} ratio={ratio_jax_ref:.6f}")
        print(f"    pt_bf  vs fp32_ref: cos={cos_pt_ref:+.7f} ratio={ratio_pt_ref:.6f}")
        print()

    print()
    print("=" * 100)
    print("INTERPRETATION:")
    print("=" * 100)
    print("  If cos(jax_bf, pt_bf) >> 0.99999 AND ratio ~ 1.0:")
    print("     bf16 LoRA intermediate rounding is NOT the source of the per-layer drift.")
    print("  If cos(jax_bf, pt_bf) < 0.999 OR ratio not ~ 1.0:")
    print("     bf16 LoRA intermediate rounding IS the source, refactor PT to apply LoRA at runtime.")
    print()
    print("  Compare jax_bf vs fp32_ref to pt_bf vs fp32_ref:")
    print("     The one closer to fp32_ref is the more accurate inference path.")
    print("     JAX paying the cost of bf16 intermediate rounding could actually be LESS accurate than PT merge.")


if __name__ == "__main__":
    main()
