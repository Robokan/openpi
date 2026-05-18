#!/usr/bin/env python3
"""Compare PT no-LoRA vs LoRA-merged checkpoints, weight-by-weight, against
the JAX computation `scaling * (lora_a @ lora_b)`.

If the difference between checkpoints matches the LoRA delta exactly, the
conversion is correct, and the bug is elsewhere (e.g. runtime).
If it doesn't match, the conversion has a bug in HOW it merges LoRA.
"""
from __future__ import annotations
import sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0, "/app/src")

import os
CKPT_PT_LORA = os.environ.get("CKPT_PT_LORA",
    "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_FIXED/model.safetensors")
CKPT_PT_NO_LORA = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_NO_LORA_v2/model.safetensors"
CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999/params/"


def load_pt(p):
    out = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).float().numpy()
    return out


def _cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    nm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if nm < 1e-30: return 0.0
    return float((a @ b) / nm)


def main():
    print("Loading PT checkpoints...")
    pt_lora = load_pt(CKPT_PT_LORA)
    pt_no_lora = load_pt(CKPT_PT_NO_LORA)
    print(f"  LoRA-merged: {len(pt_lora)} tensors")
    print(f"  No-LoRA:     {len(pt_no_lora)} tensors")

    print()
    print("Loading JAX checkpoint to compute expected delta...")
    from openpi.models import model as _model
    from flax import traverse_util as traversals
    params = _model.restore_params(CKPT_JAX, restore_type=np.ndarray, dtype="float32")
    jax_pg = traversals.flatten_dict(params["PaliGemma"], sep="/")

    # Compute expected per-layer LoRA delta for each JAX einsum
    # then verify PT checkpoint diff matches (after slicing).

    # Test the q_proj for paligemma (first 3 layers)
    print()
    print("=" * 100)
    print("Test 1: q_proj diffs (paligemma layers 0, 5, 10, 17)")
    print("=" * 100)
    lora_a_4d = jax_pg["llm/layers/attn/q_einsum/lora_a"]  # (18, 8, 2048, 16)
    lora_b_4d = jax_pg["llm/layers/attn/q_einsum/lora_b"]  # (18, 8, 16, 256)

    for L in [0, 5, 10, 17]:
        # JAX expected delta for layer L: shape (N, D, H) = (8, 2048, 256)
        delta_jax = np.matmul(lora_a_4d[L], lora_b_4d[L])  # (8, 2048, 256)
        # Apply same slice as converter: transpose(0,2,1).reshape(N*H, D)
        delta_jax_pt = delta_jax.transpose(0, 2, 1).reshape(8 * 256, 2048)  # (2048, 2048)

        key = f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.self_attn.q_proj.weight"
        diff_pt = pt_lora[key] - pt_no_lora[key]
        cos = _cos(delta_jax_pt, diff_pt)
        nm_jax = np.linalg.norm(delta_jax_pt)
        nm_pt = np.linalg.norm(diff_pt)
        print(f"  layer {L:2d}: cos(jax_delta, pt_diff)={cos:+.7f}  |jax|={nm_jax:.4f}  |pt|={nm_pt:.4f}  ratio={nm_pt/nm_jax:.6f}")
        if abs(cos) < 0.99:
            print(f"    !! BAD COS — printing sample values")
            print(f"    delta_jax_pt[0, :5] = {delta_jax_pt[0, :5]}")
            print(f"    diff_pt[0, :5]     = {diff_pt[0, :5]}")

    print()
    print("=" * 100)
    print("Test 2: q_proj diffs (gemma_expert layers 0, 5, 10, 17)")
    print("=" * 100)
    lora_a_4d = jax_pg["llm/layers/attn/q_einsum_1/lora_a"]  # (18, 8, 1024, 32)
    lora_b_4d = jax_pg["llm/layers/attn/q_einsum_1/lora_b"]  # (18, 8, 32, 256)
    for L in [0, 5, 10, 17]:
        delta_jax = np.matmul(lora_a_4d[L], lora_b_4d[L])  # (8, 1024, 256)
        delta_jax_pt = delta_jax.transpose(0, 2, 1).reshape(8 * 256, 1024)
        key = f"paligemma_with_expert.gemma_expert.model.layers.{L}.self_attn.q_proj.weight"
        diff_pt = pt_lora[key] - pt_no_lora[key]
        cos = _cos(delta_jax_pt, diff_pt)
        nm_jax = np.linalg.norm(delta_jax_pt)
        nm_pt = np.linalg.norm(diff_pt)
        print(f"  layer {L:2d}: cos={cos:+.7f}  |jax|={nm_jax:.4f}  |pt|={nm_pt:.4f}  ratio={nm_pt/nm_jax:.6f}")

    print()
    print("=" * 100)
    print("Test 3: o_proj diffs (paligemma layer 0)")
    print("=" * 100)
    lora_a_4d = jax_pg["llm/layers/attn/attn_vec_einsum/lora_a"]  # (18, 8, 256, 16)
    lora_b_4d = jax_pg["llm/layers/attn/attn_vec_einsum/lora_b"]  # (18, 8, 16, 2048)
    for L in [0, 5, 10, 17]:
        delta_jax = np.matmul(lora_a_4d[L], lora_b_4d[L])  # (8, 256, 2048)
        # PaliGemma o_proj slice: transpose(2,0,1).reshape(N*H, D) — even though shape says (N*H, D), data is in (D, N*H) order due to D==N*H==2048
        delta_jax_pt = delta_jax.transpose(2, 0, 1).reshape(8 * 256, 2048)
        key = f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.self_attn.o_proj.weight"
        diff_pt = pt_lora[key] - pt_no_lora[key]
        cos = _cos(delta_jax_pt, diff_pt)
        nm_jax = np.linalg.norm(delta_jax_pt)
        nm_pt = np.linalg.norm(diff_pt)
        print(f"  layer {L:2d}: cos={cos:+.7f}  |jax|={nm_jax:.4f}  |pt|={nm_pt:.4f}  ratio={nm_pt/nm_jax:.6f}")

    print()
    print("=" * 100)
    print("Test 4: o_proj diffs (gemma_expert layer 0)")
    print("=" * 100)
    lora_a_4d = jax_pg["llm/layers/attn/attn_vec_einsum_1/lora_a"]
    lora_b_4d = jax_pg["llm/layers/attn/attn_vec_einsum_1/lora_b"]
    for L in [0, 5, 10, 17]:
        delta_jax = np.matmul(lora_a_4d[L], lora_b_4d[L])  # (8, 256, 1024)
        # gemma_expert o_proj slice: reshape(N*H, D).transpose(1, 0)
        delta_jax_pt = delta_jax.reshape(8 * 256, 1024).transpose(1, 0)
        key = f"paligemma_with_expert.gemma_expert.model.layers.{L}.self_attn.o_proj.weight"
        diff_pt = pt_lora[key] - pt_no_lora[key]
        cos = _cos(delta_jax_pt, diff_pt)
        nm_jax = np.linalg.norm(delta_jax_pt)
        nm_pt = np.linalg.norm(diff_pt)
        print(f"  layer {L:2d}: cos={cos:+.7f}  |jax|={nm_jax:.4f}  |pt|={nm_pt:.4f}  ratio={nm_pt/nm_jax:.6f}")

    print()
    print("=" * 100)
    print("Test 5: MLP gate/up/down_proj diffs (paligemma + gemma_expert, layer 0)")
    print("=" * 100)
    for module, suffix, label in [
        ("llm/layers/mlp", "", "paligemma"),
        ("llm/layers/mlp_1", "_1", "gemma_expert"),
    ]:
        lora_gate_a = jax_pg[f"{module}/gating_einsum_lora_a"]  # (18, 2, D, rank)
        lora_gate_b = jax_pg[f"{module}/gating_einsum_lora_b"]  # (18, 2, rank, F)
        lora_lin_a = jax_pg[f"{module}/linear_lora_a"]  # (18, F, rank)
        lora_lin_b = jax_pg[f"{module}/linear_lora_b"]  # (18, rank, D)
        for L in [0]:
            # gate_proj: gating_einsum[i, 0] is shape (D, F), delta = la[0] @ lb[0]
            delta_gate = np.matmul(lora_gate_a[L, 0], lora_gate_b[L, 0])  # (D, F)
            # converter: gate_proj_weight = gating_einsum[i, 0].transpose() so shape is (F, D)
            delta_gate_pt = delta_gate.transpose()
            if label == "paligemma":
                key = f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.mlp.gate_proj.weight"
            else:
                key = f"paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.gate_proj.weight"
            diff_pt = pt_lora[key] - pt_no_lora[key]
            cos = _cos(delta_gate_pt, diff_pt)
            nm_jax = np.linalg.norm(delta_gate_pt)
            nm_pt = np.linalg.norm(diff_pt)
            print(f"  {label} L{L} gate_proj: cos={cos:+.7f}  |jax|={nm_jax:.4f}  |pt|={nm_pt:.4f}  ratio={nm_pt/nm_jax:.6f}")

            # up_proj: gating_einsum[i, 1]
            delta_up = np.matmul(lora_gate_a[L, 1], lora_gate_b[L, 1]).transpose()
            if label == "paligemma":
                key = f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.mlp.up_proj.weight"
            else:
                key = f"paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.up_proj.weight"
            diff_pt = pt_lora[key] - pt_no_lora[key]
            cos = _cos(delta_up, diff_pt)
            print(f"  {label} L{L} up_proj:   cos={cos:+.7f}  |jax|={np.linalg.norm(delta_up):.4f}  |pt|={np.linalg.norm(diff_pt):.4f}  ratio={np.linalg.norm(diff_pt)/np.linalg.norm(delta_up):.6f}")

            # down_proj: linear[i] of shape (F, D), transpose -> (D, F)
            delta_down = np.matmul(lora_lin_a[L], lora_lin_b[L]).transpose()  # (D, F)
            if label == "paligemma":
                key = f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.mlp.down_proj.weight"
            else:
                key = f"paligemma_with_expert.gemma_expert.model.layers.{L}.mlp.down_proj.weight"
            diff_pt = pt_lora[key] - pt_no_lora[key]
            cos = _cos(delta_down, diff_pt)
            print(f"  {label} L{L} down_proj: cos={cos:+.7f}  |jax|={np.linalg.norm(delta_down):.4f}  |pt|={np.linalg.norm(diff_pt):.4f}  ratio={np.linalg.norm(diff_pt)/np.linalg.norm(delta_down):.6f}")

    print()
    print("Summary: any cos != ~1.0 means the conversion mismatch for that weight type.")


if __name__ == "__main__":
    main()
