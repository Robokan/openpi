"""Runtime LoRA application for PyTorch PaliGemma+Expert models.

When `OPENPI_PT_RUNTIME_LORA=1`, the JAX→PT converter saves LoRA `lora_a`/
`lora_b` tensors separately (no pre-merge into base weight). This module
provides the runtime hook that re-applies LoRA on every forward pass,
matching the JAX two-matmul order exactly.

Why this exists
---------------
The previous code path pre-merged `w_base + scaling * la @ lb` into the
base weight at conversion time, then cast to bf16 for inference. JAX
keeps `w`, `la`, `lb` separate at runtime, casts each to bf16, does TWO
bf16 matmuls (with a bf16 rounding on the rank-r intermediate), then
adds. These two paths are mathematically equivalent in fp32 but differ
in bf16, and the difference compounds through 18 transformer layers.

LoRA tensor shapes (after conversion)
-------------------------------------
For each LoRA-targeted linear, we attach two tensors and a scaling
scalar to the existing `nn.Linear` module:

  q_proj  (PaliGemma layer L)
      lora_a:  (N, D, L)        # JAX-native
      lora_b:  (N, L, H)        # JAX-native
  k_proj / v_proj
      lora_a:  (D, L)
      lora_b:  (L, H)
  o_proj
      lora_a:  (N*H, L)         # JAX (N,H,L) flattened
      lora_b:  (L, D)           # JAX (N,L,D) summed over N (fp32)
  gate_proj / up_proj / down_proj
      lora_a:  (D_in, L)
      lora_b:  (L, D_out)

Forward patches do exactly what JAX `lora.Einsum.__call__` does for each
case: two matmuls, rounded through bf16 between them when the input is
bf16.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    import openpi.models_pytorch.pi0_pytorch as _pi0_pt

logger = logging.getLogger(__name__)


_LORA_SUFFIXES = ("lora_a", "lora_b", "lora_scaling")


def _has_lora(module: nn.Module) -> bool:
    return all(hasattr(module, s) for s in ("lora_a", "lora_b"))


def _patch_standard_linear_forward(module: nn.Module) -> None:
    """For k_proj, v_proj, gate_proj, up_proj, down_proj — single Linear
    with a plain (x @ la) @ lb LoRA path."""
    base_forward = nn.Linear.forward.__get__(module, nn.Linear)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = base_forward(x)
        la = module.lora_a.to(x.dtype)
        lb = module.lora_b.to(x.dtype)
        lora_int = x @ la
        lora_out = lora_int @ lb
        return out + module.lora_scaling * lora_out

    module.forward = forward


def _patch_q_proj_forward(module: nn.Module, num_heads: int, head_dim: int) -> None:
    """q_proj (paligemma or gemma_expert) uses per-head LoRA.
    JAX flow:  lora_int = einsum(BTD,NDL->BTNL); lora_out = einsum(BTNL,NLH->BTNH).
    PT base produces (B, T, N*H); we add the flattened LoRA contribution.
    """
    base_forward = nn.Linear.forward.__get__(module, nn.Linear)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = base_forward(x)  # (B, T, N*H)
        la = module.lora_a.to(x.dtype)  # (N, D, L)
        lb = module.lora_b.to(x.dtype)  # (N, L, H)
        # Match JAX einsum order: (BTD x NDL -> BTNL) then (BTNL x NLH -> BTNH)
        lora_int = torch.einsum("btd,ndl->btnl", x, la)
        lora_out = torch.einsum("btnl,nlh->btnh", lora_int, lb)
        lora_flat = lora_out.flatten(2)  # (B, T, N*H)
        return out + module.lora_scaling * lora_flat

    module.forward = forward


def _patch_o_proj_forward(module: nn.Module) -> None:
    """o_proj uses LoRA with multi-head input and N-summed lb.
    JAX flow:  lora_int = einsum(BTNH,NHL->BTL); lora_out = einsum(BTL,NLD->BTD).
    The N axis on lb is implicitly summed (since not in output, not contracted
    with input). We pre-sum it once at attach time (fp32) into a (L, D) matrix
    so runtime is a single (B,T,L)@(L,D) matmul.

    `lora_a` stored shape: (N*H, L) (flattened JAX (N, H, L))
    `lora_b` stored shape: (L, D)    (JAX (N, L, D).sum(axis=0))
    """
    base_forward = nn.Linear.forward.__get__(module, nn.Linear)

    def forward(x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, N*H)
        out = base_forward(x)  # (B, T, D)
        la = module.lora_a.to(x.dtype)  # (N*H, L)
        lb = module.lora_b.to(x.dtype)  # (L, D)
        lora_int = x @ la                # (B, T, L)
        lora_out = lora_int @ lb         # (B, T, D)
        return out + module.lora_scaling * lora_out

    module.forward = forward


def _attach_lora(
    module: nn.Module,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> None:
    """Attach lora_a/lora_b as buffers on the module and set scaling."""
    if hasattr(module, "lora_a"):
        del module.lora_a
    if hasattr(module, "lora_b"):
        del module.lora_b
    module.register_buffer("lora_a", lora_a, persistent=False)
    module.register_buffer("lora_b", lora_b, persistent=False)
    module.lora_scaling = float(scaling)


def install_runtime_lora(model: "_pi0_pt.PI0Pytorch", lora_state_dict: dict[str, torch.Tensor]) -> int:
    """Attach LoRA tensors and patch projection forwards on the PT model.

    Args:
        model: a PI0Pytorch instance (already loaded with base weights).
        lora_state_dict: keys follow the convention
              {pt_module_path}.lora_a
              {pt_module_path}.lora_b
              {pt_module_path}.lora_scaling   (scalar tensor or float)
            where `pt_module_path` is e.g.
              "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj"

    Returns the number of LoRA-patched modules.
    """
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora_state_dict.items():
        if not k.endswith(tuple(f".{s}" for s in _LORA_SUFFIXES)):
            continue
        path, _, suffix = k.rpartition(".")
        grouped.setdefault(path, {})[suffix] = v

    if not grouped:
        return 0

    n_patched = 0
    for path, lora_dict in grouped.items():
        if "lora_a" not in lora_dict or "lora_b" not in lora_dict:
            logger.warning(f"LoRA: incomplete pair for {path}, skipping")
            continue
        try:
            module = _resolve_submodule(model, path)
        except AttributeError:
            logger.warning(f"LoRA: cannot resolve module {path}, skipping")
            continue
        if not isinstance(module, nn.Linear):
            logger.warning(f"LoRA: {path} is not Linear (got {type(module).__name__}), skipping")
            continue

        scaling_val = lora_dict.get("lora_scaling", None)
        if scaling_val is None:
            scaling = 1.0
        elif isinstance(scaling_val, torch.Tensor):
            scaling = float(scaling_val.item())
        else:
            scaling = float(scaling_val)

        _attach_lora(module, lora_dict["lora_a"], lora_dict["lora_b"], scaling)

        # Pick the appropriate forward patch based on the projection name
        if path.endswith(".q_proj"):
            # Infer head dims from the base Linear weight: out_features = N*H
            out_features = module.weight.shape[0]
            # lora_a shape (N, D, L) tells us N directly
            num_heads = lora_dict["lora_a"].shape[0]
            head_dim = out_features // num_heads
            _patch_q_proj_forward(module, num_heads=num_heads, head_dim=head_dim)
        elif path.endswith(".o_proj"):
            _patch_o_proj_forward(module)
        else:
            _patch_standard_linear_forward(module)
        n_patched += 1

    logger.info(f"Runtime LoRA: patched {n_patched} projection modules.")
    return n_patched


def _resolve_submodule(model: nn.Module, dotted_path: str) -> nn.Module:
    obj: nn.Module = model
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj
