#!/usr/bin/env python3
"""Unit-test LoRA merge: for one attention layer's q_einsum, compare:
  (A) JAX-runtime:  result = einsum(x, w) + scaling * einsum(x, lora_a, lora_b)
  (B) PT-converter: merged_w = w + scaling * (lora_a @ lora_b); result = einsum(x, merged_w)

If these differ in fp32, the merge math is wrong.
If they match in fp32 but differ in bf16, there's a precision issue at conversion time.
"""

from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")


def _cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    from openpi.models import model as _model
    from flax import traverse_util as traversals

    params = _model.restore_params(
        "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999/params/",
        restore_type=np.ndarray, dtype="float32",
    )
    pg = traversals.flatten_dict(params["PaliGemma"], sep="/")

    # Test each LoRA pair
    test_pairs = [
        ("llm/layers/attn/q_einsum/w",          "llm/layers/attn/q_einsum/lora_a",          "llm/layers/attn/q_einsum/lora_b",          "BTD,NDH->BTNH", 16),
        ("llm/layers/attn/kv_einsum/w",         "llm/layers/attn/kv_einsum/lora_a",         "llm/layers/attn/kv_einsum/lora_b",         "BSD,2KDH->2BSKH", 16),
        ("llm/layers/attn/attn_vec_einsum/w",   "llm/layers/attn/attn_vec_einsum/lora_a",   "llm/layers/attn/attn_vec_einsum/lora_b",   "BTNH,NHD->BTD", 16),
        ("llm/layers/attn/q_einsum_1/w",        "llm/layers/attn/q_einsum_1/lora_a",        "llm/layers/attn/q_einsum_1/lora_b",        "BTD,NDH->BTNH", 32),
        ("llm/layers/attn/kv_einsum_1/w",       "llm/layers/attn/kv_einsum_1/lora_a",       "llm/layers/attn/kv_einsum_1/lora_b",       "BSD,2KDH->2BSKH", 32),
        ("llm/layers/attn/attn_vec_einsum_1/w", "llm/layers/attn/attn_vec_einsum_1/lora_a", "llm/layers/attn/attn_vec_einsum_1/lora_b", "BTNH,NHD->BTD", 32),
    ]

    print(f"{'layer':50s}  {'shapes':30s}  {'scaling':8s}  {'cos':10s}  {'maxd':10s}")
    print("-" * 130)
    for w_k, la_k, lb_k, eqn, rank in test_pairs:
        if w_k not in pg or la_k not in pg or lb_k not in pg:
            print(f"  MISSING: {w_k}")
            continue
        w = pg[w_k][0]   # take layer 0 only (drop the num_layers dim)
        la = pg[la_k][0]
        lb = pg[lb_k][0]

        alpha = rank  # alpha==rank for our config, rslora=False
        scaling = alpha / rank  # = 1.0

        # Random input
        np.random.seed(0)
        if eqn == "BTD,NDH->BTNH":
            D = w.shape[-2]
            x = np.random.randn(1, 4, D).astype(np.float32)
            # JAX-runtime computation
            base_out = np.einsum(eqn, x, w)
            lora_a_out = np.einsum("BTD,NDL->BTNL", x, la)
            lora_b_out = np.einsum("BTNL,NLH->BTNH", lora_a_out, lb)
            jax_runtime = base_out + scaling * lora_b_out

            # PT-merged computation
            delta = np.matmul(la, lb)
            merged_w = w + scaling * delta
            pt_merged = np.einsum(eqn, x, merged_w)
        elif eqn == "BSD,2KDH->2BSKH":
            # numpy einsum doesn't accept '2', use 'P' instead
            sub_eqn = "BSD,PKDH->PBSKH"
            D = w.shape[-2]
            x = np.random.randn(1, 4, D).astype(np.float32)
            base_out = np.einsum(sub_eqn, x, w)
            lora_a_out = np.einsum("BSD,PKDL->PBSKL", x, la)
            lora_b_out = np.einsum("PBSKL,PKLH->PBSKH", lora_a_out, lb)
            jax_runtime = base_out + scaling * lora_b_out

            delta = np.matmul(la, lb)
            merged_w = w + scaling * delta
            pt_merged = np.einsum(sub_eqn, x, merged_w)
        elif eqn == "BTNH,NHD->BTD":
            N, H, D = w.shape
            x = np.random.randn(1, 4, N, H).astype(np.float32)
            base_out = np.einsum(eqn, x, w)
            lora_a_out = np.einsum("BTNH,NHL->BTNL", x, la)
            lora_b_out = np.einsum("BTNL,NLD->BTD", lora_a_out, lb)
            jax_runtime = base_out + scaling * lora_b_out

            delta = np.matmul(la, lb)
            merged_w = w + scaling * delta
            pt_merged = np.einsum(eqn, x, merged_w)
        else:
            continue

        cos = _cos(jax_runtime, pt_merged)
        maxd = float(np.max(np.abs(jax_runtime - pt_merged)))
        name = w_k.split("/")[-2]
        shapes = f"w={tuple(w.shape)} r={rank}"
        ratio = float(np.linalg.norm(pt_merged) / np.linalg.norm(jax_runtime))
        print(f"  {name:50s}  {shapes:30s}  s={scaling:6.3f}  cos={cos:+.7f}  maxd={maxd:.2e}  ratio={ratio:.6f}")

        # === BF16 comparison ===
        # JAX runtime: cast w, la, lb, x to bf16; compute everything in bf16
        # PT runtime: cast merged_w to bf16; compute in bf16
        import torch as _t
        def _to_bf16(arr):
            return _t.from_numpy(arr).to(_t.bfloat16).to(_t.float32).numpy()

        w_bf = _to_bf16(w); la_bf = _to_bf16(la); lb_bf = _to_bf16(lb); x_bf = _to_bf16(x)
        if eqn == "BTD,NDH->BTNH":
            jax_bf = np.einsum(eqn, x_bf, w_bf) + scaling * np.einsum("BTNL,NLH->BTNH",
                np.einsum("BTD,NDL->BTNL", x_bf, la_bf), lb_bf)
            merged_bf = _to_bf16(w + scaling * np.matmul(la, lb))  # merge in fp32 then cast
            pt_bf = np.einsum(eqn, x_bf, merged_bf)
        elif eqn == "BSD,2KDH->2BSKH":
            sub_eqn = "BSD,PKDH->PBSKH"
            jax_bf = np.einsum(sub_eqn, x_bf, w_bf) + scaling * np.einsum("PBSKL,PKLH->PBSKH",
                np.einsum("BSD,PKDL->PBSKL", x_bf, la_bf), lb_bf)
            merged_bf = _to_bf16(w + scaling * np.matmul(la, lb))
            pt_bf = np.einsum(sub_eqn, x_bf, merged_bf)
        elif eqn == "BTNH,NHD->BTD":
            jax_bf = np.einsum(eqn, x_bf, w_bf) + scaling * np.einsum("BTNL,NLD->BTD",
                np.einsum("BTNH,NHL->BTNL", x_bf, la_bf), lb_bf)
            merged_bf = _to_bf16(w + scaling * np.matmul(la, lb))
            pt_bf = np.einsum(eqn, x_bf, merged_bf)
        cos_bf = _cos(jax_bf, pt_bf)
        maxd_bf = float(np.max(np.abs(jax_bf - pt_bf)))
        ratio_bf = float(np.linalg.norm(pt_bf) / np.linalg.norm(jax_bf))
        # Compare also against the FP32 truth
        cos_jax_truth = _cos(jax_runtime, jax_bf)
        cos_pt_truth = _cos(jax_runtime, pt_bf)
        print(f"    BF16: cos={cos_bf:+.7f}  maxd={maxd_bf:.2e}  ratio={ratio_bf:.6f}  "
              f"cos(truth,JAX_bf)={cos_jax_truth:.7f}  cos(truth,PT_bf)={cos_pt_truth:.7f}")

        # Also print magnitudes
        delta_norm = np.linalg.norm(scaling * np.matmul(la, lb))
        w_norm = np.linalg.norm(w)
        merged_norm = np.linalg.norm(merged_w)
        print(f"    |w|={w_norm:.3f}  |scaling*delta|={delta_norm:.3f}  |merged|={merged_norm:.3f}  "
              f"delta/w = {delta_norm/w_norm:.4f}")
    print()
    print("If cos != 1.0 or maxd is large: merge MATH is wrong.")
    print("If cos==1.0 and maxd~0: merge is mathematically correct.")
    print()
    print("MLP merge tests:")
    print("-" * 130)
    for w_k, la_k, lb_k, label in [
        ("llm/layers/mlp/gating_einsum",   "llm/layers/mlp/gating_einsum_lora_a",   "llm/layers/mlp/gating_einsum_lora_b",   "mlp.gating_einsum"),
        ("llm/layers/mlp/linear",          "llm/layers/mlp/linear_lora_a",          "llm/layers/mlp/linear_lora_b",          "mlp.linear"),
        ("llm/layers/mlp_1/gating_einsum", "llm/layers/mlp_1/gating_einsum_lora_a", "llm/layers/mlp_1/gating_einsum_lora_b", "mlp_1.gating_einsum"),
        ("llm/layers/mlp_1/linear",        "llm/layers/mlp_1/linear_lora_a",        "llm/layers/mlp_1/linear_lora_b",        "mlp_1.linear"),
    ]:
        w = pg[w_k][0]
        la = pg[la_k][0]
        lb = pg[lb_k][0]
        rank = la.shape[-1] if "linear" in w_k.split("/")[-1] else la.shape[-1]
        alpha = rank
        scaling = 1.0

        if w.shape[0] == 2:  # gating_einsum shape (2, D, F)
            # JAX _dot: base + (x @ lora_a) @ lora_b
            D, F = w.shape[1], w.shape[2]
            x = np.random.randn(1, 4, D).astype(np.float32)
            base_out_0 = np.dot(x, w[0])
            lora_intm_0 = np.dot(np.dot(x, la[0]), lb[0])
            jax_runtime = base_out_0 + scaling * lora_intm_0
            delta_0 = np.dot(la[0], lb[0])
            pt_merged = np.dot(x, w[0] + scaling * delta_0)
        else:  # linear: shape (F, D)
            F, D = w.shape
            x = np.random.randn(1, 4, F).astype(np.float32)
            base_out = np.dot(x, w)
            lora_intm = np.dot(np.dot(x, la), lb)
            jax_runtime = base_out + scaling * lora_intm
            delta = np.dot(la, lb)
            pt_merged = np.dot(x, w + scaling * delta)
        cos = _cos(jax_runtime, pt_merged)
        maxd = float(np.max(np.abs(jax_runtime - pt_merged)))
        delta_norm = np.linalg.norm(scaling * np.matmul(la, lb))
        w_norm = np.linalg.norm(w)
        print(f"  {label:40s} w={tuple(w.shape)} r={rank}  cos={cos:+.7f}  maxd={maxd:.2e}  |delta|/|w|={delta_norm/w_norm:.4f}")


if __name__ == "__main__":
    main()
