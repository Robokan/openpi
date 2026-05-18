#!/usr/bin/env python3
"""Replicate the EXACT converter merge for q_einsum (paligemma layer 0),
then compare to what's actually in the PT checkpoints.

If the manual replication MATCHES `pt_lora - pt_no_lora`, the converter is OK.
If it DOESN'T match, the converter has a bug we haven't pinpointed yet.
"""
import sys, numpy as np
sys.path.insert(0, "/app/src")
from safetensors import safe_open
from openpi.models import model as _model
from flax import traverse_util as traversals

CKPT_PT_LORA    = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch/model.safetensors"
CKPT_PT_NO_LORA = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch_NO_LORA_v2/model.safetensors"
CKPT_JAX        = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999/params/"


def main():
    params = _model.restore_params(CKPT_JAX, restore_type=np.ndarray, dtype="float32")
    pg = traversals.flatten_dict(params["PaliGemma"], sep="/")

    # Pick paligemma layer 0 q_einsum
    w_full = pg["llm/layers/attn/q_einsum/w"]    # (18, 8, 2048, 256)
    la_full = pg["llm/layers/attn/q_einsum/lora_a"]  # (18, 8, 2048, 16)
    lb_full = pg["llm/layers/attn/q_einsum/lora_b"]  # (18, 8, 16, 256)

    print("JAX dtypes/shapes:")
    print(f"  w  : {w_full.shape} {w_full.dtype}")
    print(f"  la : {la_full.shape} {la_full.dtype}")
    print(f"  lb : {lb_full.shape} {lb_full.dtype}")

    L = 0
    w = w_full[L]   # (8, 2048, 256)
    la = la_full[L]  # (8, 2048, 16)
    lb = lb_full[L]  # (8, 16, 256)

    # === Replicate converter EXACTLY ===
    # Step 1: compute delta (fp32) and merged_w (fp32)
    delta = np.matmul(la.astype(np.float32), lb.astype(np.float32))  # (8, 2048, 256)
    scaling = 1.0
    # In the converter: `merged[base_k] = (w.astype(np.float32) + scaling * delta).astype(w.dtype)`
    # w.dtype is fp32, so the .astype(w.dtype) is a no-op.
    merged_w_full = (w_full.astype(np.float32) + scaling * np.matmul(la_full.astype(np.float32), lb_full.astype(np.float32))).astype(w_full.dtype)
    merged_w = merged_w_full[L]  # take layer 0

    # Step 2: replicate slice_paligemma_state_dict for q_proj
    # `.transpose(0, 2, 1).reshape(N*H, D)`
    N, H, D = 8, 256, 2048
    pt_w_manual = merged_w.transpose(0, 2, 1).reshape(N * H, D)  # (2048, 2048)

    # No-LoRA case: just slice w (no merge)
    pt_w_nolora_manual = w.transpose(0, 2, 1).reshape(N * H, D)

    # Manual diff
    pt_diff_manual = pt_w_manual - pt_w_nolora_manual

    # === Read actual PT-saved values ===
    key = f"paligemma_with_expert.paligemma.model.language_model.layers.{L}.self_attn.q_proj.weight"
    with safe_open(CKPT_PT_LORA, framework="pt") as f:
        pt_w_actual = f.get_tensor(key).float().numpy()
    with safe_open(CKPT_PT_NO_LORA, framework="pt") as f:
        pt_w_nolora_actual = f.get_tensor(key).float().numpy()
    pt_diff_actual = pt_w_actual - pt_w_nolora_actual

    print()
    print("=== Compare manual vs actual saved values ===")
    print(f"  pt_w_manual[0, :5]        = {pt_w_manual[0, :5]}")
    print(f"  pt_w_actual[0, :5]        = {pt_w_actual[0, :5]}")
    print(f"  pt_w_nolora_manual[0, :5] = {pt_w_nolora_manual[0, :5]}")
    print(f"  pt_w_nolora_actual[0, :5] = {pt_w_nolora_actual[0, :5]}")

    diff_lora_manual_actual = pt_w_manual - pt_w_actual
    diff_nolora_manual_actual = pt_w_nolora_manual - pt_w_nolora_actual
    print()
    print(f"  |pt_w_manual - pt_w_actual|        = {np.linalg.norm(diff_lora_manual_actual):.6e}  (should be 0)")
    print(f"  |pt_w_nolora_manual - pt_w_nolora_actual| = {np.linalg.norm(diff_nolora_manual_actual):.6e}  (should be 0)")

    print()
    print("=== Compare delta forms ===")
    # JAX delta in JAX layout (what JAX runtime applies):
    delta_jax_layout = delta  # (8, 2048, 256)
    delta_jax_pt_layout = delta_jax_layout.transpose(0, 2, 1).reshape(N * H, D)
    # Compare against actual saved diff
    a, b = delta_jax_pt_layout.flatten(), pt_diff_actual.flatten()
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"  cos(jax_delta_pt_layout, pt_diff_actual) = {cos:.7f}")
    print(f"  |jax_delta_pt_layout| = {np.linalg.norm(delta_jax_pt_layout):.6f}")
    print(f"  |pt_diff_actual|      = {np.linalg.norm(pt_diff_actual):.6f}")
    print(f"  ratio (pt/jax)        = {np.linalg.norm(pt_diff_actual)/np.linalg.norm(delta_jax_pt_layout):.6f}")

    # Now compare manual diff (which uses the same merge math):
    a2 = pt_diff_manual.flatten(); b2 = pt_diff_actual.flatten()
    cos2 = float((a2 @ b2) / (np.linalg.norm(a2) * np.linalg.norm(b2)))
    print()
    print(f"  cos(pt_diff_manual, pt_diff_actual) = {cos2:.7f}  (should be 1.0)")
    print(f"  |pt_diff_manual| = {np.linalg.norm(pt_diff_manual):.6f}")
    print(f"  |pt_diff_actual| = {np.linalg.norm(pt_diff_actual):.6f}")
    print(f"  ratio = {np.linalg.norm(pt_diff_actual)/np.linalg.norm(pt_diff_manual):.6f}")
    print()
    print(f"  pt_diff_manual[0, :5] = {pt_diff_manual[0, :5]}")
    print(f"  pt_diff_actual[0, :5] = {pt_diff_actual[0, :5]}")
    print(f"  (manual - actual)[0, :5] = {(pt_diff_manual - pt_diff_actual)[0, :5]}")

    # === What if no-LoRA checkpoint was made with restore_precision="bfloat16" somehow? ===
    # Test: compute no-LoRA expected with bf16 cast
    import torch
    w_bf = torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16).to(torch.float32).numpy()
    pt_w_nolora_bf16 = w_bf.transpose(0, 2, 1).reshape(N * H, D)
    print()
    print(f"  If no-LoRA was bf16:")
    print(f"  |pt_w_nolora_bf16 - pt_w_nolora_actual| = {np.linalg.norm(pt_w_nolora_bf16 - pt_w_nolora_actual):.6e}")

    # Test: maybe lora was applied DOUBLE somehow
    pt_double = w.astype(np.float32) + 2 * delta
    pt_double_sliced = pt_double.transpose(0, 2, 1).reshape(N * H, D)
    pt_diff_double = pt_double_sliced - pt_w_nolora_manual
    cos_d = float((pt_diff_double.flatten() @ pt_diff_actual.flatten()) / (np.linalg.norm(pt_diff_double) * np.linalg.norm(pt_diff_actual)))
    print(f"  cos with scaling=2.0:  {cos_d:.7f}  |scaling=2|={np.linalg.norm(pt_diff_double):.6f}  |actual|={np.linalg.norm(pt_diff_actual):.6f}")


if __name__ == "__main__":
    main()
