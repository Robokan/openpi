"""Runtime quantization for PI0Pytorch using torchao.

Composes with `lora_runtime.install_runtime_lora`:
  base weights -> FP8 / NVFP4   (quantized)
  lora_a / lora_b -> fp16/bf16  (unquantized, additive)

This is the QLoRA pattern: the bulk of the parameters are in low precision,
while the small LoRA adapters stay in higher precision so the adapter's
contribution isn't quantized away.

Target Blackwell (sm_100) for Tensor Core FP8/FP4 acceleration. Ada/Hopper
will also work via emulation for FP8 (NVFP4 requires Blackwell).

Selection via environment variable `OPENPI_PT_QUANT`:
  fp8_w      Float8 weight-only (activations stay bf16, w*a in bf16, weight stored fp8)
  fp8_wa     Float8 dynamic activation + weight (full FP8 GEMM on tensor cores)
  nvfp4      NVFP4 W4A4 inference (Blackwell native, 4-bit weight + activation + microscale)
"""
from __future__ import annotations

import logging
import re
from typing import Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)


# Modules to quantize: any q/k/v/o/gate/up/down_proj inside `.layers.N.self_attn`
# or `.layers.N.mlp` (i.e. Gemma's projection Linears). We deliberately exclude
# - lm_head        (tied to embed_tokens; quantizing breaks tying)
# - embed_tokens   (Embedding, not Linear)
# - multi_modal_projector (small)
# - vision_tower   (SigLIP — TODO: enable in a later pass)
# - action_*_proj / time_mlp_*  (tiny dense heads)
_PROJ_RE = re.compile(r"\.(?:self_attn|mlp)\.[a-z_]+_proj$")


def _is_gemma_projection(module: nn.Module, fqn: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return bool(_PROJ_RE.search(fqn))


def _safety_check(model: nn.Module) -> int:
    """Count how many modules will be quantized — sanity check before quant."""
    n = 0
    for fqn, m in model.named_modules():
        if _is_gemma_projection(m, fqn):
            n += 1
    return n


def install_fp8_weight_only(model: nn.Module) -> int:
    """FP8 weight-only quantization on Gemma projections.

    Weights are stored as FP8 (E4M3), dequantized at matmul time to compute
    bf16 GEMM. Activations are unaffected. ~2x memory reduction on the
    quantized weights, modest speedup, minimal accuracy loss.

    Safe baseline; recommended first quantization to try.
    """
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    n = _safety_check(model)
    logger.info(f"FP8 weight-only: targeting {n} Linear modules")
    quantize_(model, Float8WeightOnlyConfig(), filter_fn=_is_gemma_projection)
    return n


def install_fp8_dynamic(model: nn.Module) -> int:
    """FP8 dynamic activation + weight quantization (full FP8 GEMM).

    Activations are dynamically quantized to FP8 per-row at runtime; weights
    are pre-quantized to FP8 per-row. Tensor Cores compute FP8 GEMM with
    fp32 accumulator. Best memory + speed reduction, slightly more accuracy
    loss than weight-only.
    """
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

    n = _safety_check(model)
    logger.info(f"FP8 dynamic activation + weight: targeting {n} Linear modules")
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(), filter_fn=_is_gemma_projection)
    return n


def install_nvfp4(model: nn.Module) -> int:
    """NVFP4 W4A4 inference quantization (Blackwell native FP4).

    Both weights and activations are quantized to NVFP4 (E2M1 with an FP8
    microscale per group of 16). 4x memory reduction, large speedup on
    Blackwell tensor cores, expected accuracy hit larger than FP8 — must
    validate per-task.

    NVFP4 (torchao 0.13) requires bf16 weights/bias on the source Linear
    (fp32 + bias is unsupported), so we cast targeted modules to bf16 first.
    Other modules (LoRA buffers, action heads, conditioning) keep their
    upstream dtype.
    """
    from torchao.prototype.mx_formats import NVFP4InferenceConfig
    from torchao.quantization import quantize_

    n = _safety_check(model)
    logger.info(f"NVFP4: targeting {n} Linear modules; casting to bf16 first")
    for fqn, m in model.named_modules():
        if _is_gemma_projection(m, fqn):
            m.weight.data = m.weight.data.to(torch.bfloat16)
            if m.bias is not None:
                m.bias.data = m.bias.data.to(torch.bfloat16)
    quantize_(model, NVFP4InferenceConfig(), filter_fn=_is_gemma_projection)
    return n


QuantMode = Literal["fp8_w", "fp8_wa", "nvfp4"]


def install_quantization(model: nn.Module, mode: QuantMode) -> int:
    """Entry point dispatched from `OPENPI_PT_QUANT`."""
    if mode == "fp8_w":
        return install_fp8_weight_only(model)
    if mode == "fp8_wa":
        return install_fp8_dynamic(model)
    if mode == "nvfp4":
        return install_nvfp4(model)
    raise ValueError(f"Unknown OPENPI_PT_QUANT mode: {mode!r} (choose fp8_w / fp8_wa / nvfp4)")
