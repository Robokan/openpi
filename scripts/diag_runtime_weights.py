#!/usr/bin/env python3
"""Verify that PT runtime weights (after load_pytorch + to_bfloat16_for_selected_params)
match the JAX expected `base + scaling*(la@lb)` value to bf16 precision.

If at-save the delta is correct (cos=1, ratio=1) but at-runtime it ISN'T,
then load_pytorch / to_bfloat16_for_selected_params is corrupting LoRA.
"""
from __future__ import annotations
import sys
import os
import numpy as np
import torch

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")


def _cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    from flax import traverse_util as traversals

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    p_fixed = _pc.create_trained_policy(cfg,
        "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_FIXED")
    pt_model = p_fixed._model
    pt_model.eval()

    # Load JAX raw params for reference
    params = _model.restore_params(
        "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999/params/",
        restore_type=np.ndarray, dtype="float32"
    )
    jpg = traversals.flatten_dict(params["PaliGemma"], sep="/")

    print(f"PT inference precision (after load+cast): {pt_model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype}")
    print(f"PT action_in_proj precision: {pt_model.action_in_proj.weight.dtype}")
    print()

    # ===== Verify paligemma q_proj at runtime =====
    print("=" * 100)
    print("PaliGemma q_proj runtime check")
    print("=" * 100)
    la_full = jpg["llm/layers/attn/q_einsum/lora_a"]
    lb_full = jpg["llm/layers/attn/q_einsum/lora_b"]
    w_full = jpg["llm/layers/attn/q_einsum/w"]
    for L in [0, 5, 17]:
        w = w_full[L]; la = la_full[L]; lb = lb_full[L]
        # Expected merged weight after slicing
        delta = np.matmul(la, lb)
        merged = w + 1.0 * delta  # (N, D, H)
        expected_pt = merged.transpose(0, 2, 1).reshape(8 * 256, 2048)  # (N*H, D)

        # Read PT runtime weight
        layer = pt_model.paligemma_with_expert.paligemma.model.language_model.layers[L]
        pt_w = layer.self_attn.q_proj.weight.detach().float().cpu().numpy()

        # Also expected at bf16
        expected_pt_bf = torch.from_numpy(expected_pt).to(torch.bfloat16).float().numpy()

        cos_fp32 = _cos(expected_pt, pt_w)
        cos_bf16 = _cos(expected_pt_bf, pt_w)
        nm_exp = np.linalg.norm(expected_pt)
        nm_pt = np.linalg.norm(pt_w)
        max_diff = float(np.max(np.abs(expected_pt - pt_w)))
        print(f"  layer {L:2d}: cos(exp_fp32, pt_runtime)={cos_fp32:+.7f}  cos(exp_bf16, pt_runtime)={cos_bf16:+.7f}  "
              f"ratio={nm_pt/nm_exp:.6f}  max|diff|={max_diff:.4e}")

    # ===== Verify gemma_expert q_proj at runtime =====
    print()
    print("=" * 100)
    print("Gemma_expert q_proj runtime check")
    print("=" * 100)
    la_full = jpg["llm/layers/attn/q_einsum_1/lora_a"]
    lb_full = jpg["llm/layers/attn/q_einsum_1/lora_b"]
    w_full = jpg["llm/layers/attn/q_einsum_1/w"]
    for L in [0, 5, 17]:
        w = w_full[L]; la = la_full[L]; lb = lb_full[L]
        delta = np.matmul(la, lb)
        merged = w + 1.0 * delta
        # gemma_expert uses same q-slice as paligemma
        expected_pt = merged.transpose(0, 2, 1).reshape(8 * 256, 1024)

        layer = pt_model.paligemma_with_expert.gemma_expert.model.layers[L]
        pt_w = layer.self_attn.q_proj.weight.detach().float().cpu().numpy()
        expected_pt_bf = torch.from_numpy(expected_pt).to(torch.bfloat16).float().numpy()

        cos_fp32 = _cos(expected_pt, pt_w)
        cos_bf16 = _cos(expected_pt_bf, pt_w)
        nm_exp = np.linalg.norm(expected_pt)
        nm_pt = np.linalg.norm(pt_w)
        max_diff = float(np.max(np.abs(expected_pt - pt_w)))
        print(f"  layer {L:2d}: cos(exp_fp32, pt_runtime)={cos_fp32:+.7f}  cos(exp_bf16, pt_runtime)={cos_bf16:+.7f}  "
              f"ratio={nm_pt/nm_exp:.6f}  max|diff|={max_diff:.4e}")

    # ===== Check MLP linear (down_proj) for gemma_expert (32-rank LoRA, big magnitude) =====
    print()
    print("=" * 100)
    print("Gemma_expert mlp.down_proj runtime check (linear LoRA, rank=32)")
    print("=" * 100)
    la_full = jpg["llm/layers/mlp_1/linear_lora_a"]
    lb_full = jpg["llm/layers/mlp_1/linear_lora_b"]
    w_full = jpg["llm/layers/mlp_1/linear"]
    for L in [0, 5, 17]:
        w = w_full[L]; la = la_full[L]; lb = lb_full[L]
        delta = np.matmul(la, lb)
        merged = w + 1.0 * delta  # (F, D)
        # down_proj slice: linear[i].transpose() → (D, F)
        expected_pt = merged.transpose()

        layer = pt_model.paligemma_with_expert.gemma_expert.model.layers[L]
        pt_w = layer.mlp.down_proj.weight.detach().float().cpu().numpy()
        expected_pt_bf = torch.from_numpy(expected_pt).to(torch.bfloat16).float().numpy()

        cos_fp32 = _cos(expected_pt, pt_w)
        cos_bf16 = _cos(expected_pt_bf, pt_w)
        ratio_fp32 = np.linalg.norm(pt_w) / np.linalg.norm(expected_pt)
        max_diff = float(np.max(np.abs(expected_pt - pt_w)))
        print(f"  layer {L:2d}: cos(exp_fp32, pt_runtime)={cos_fp32:+.7f}  cos(exp_bf16, pt_runtime)={cos_bf16:+.7f}  "
              f"ratio={ratio_fp32:.6f}  max|diff|={max_diff:.4e}")


if __name__ == "__main__":
    main()
