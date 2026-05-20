"""Real-Time Chunking (RTC) for action-chunk flow policies.

Implements the inference-time inpainting algorithm from
"Real-Time Execution of Action Chunking Flow Policies" (Black et al. 2025).

The algorithm modifies the velocity field of a flow-matching denoiser via a
Pi-GDM (pseudoinverse guidance) correction at each step so that the newly
generated chunk is consistent with the unexecuted portion of the previous
chunk. This eliminates the "stop-and-go" pauses at chunk boundaries that
arise from synchronous (blocking) inference.

Reference implementation:
    https://github.com/Physical-Intelligence/real-time-chunking-kinetix
    (`src/model.py::Model.realtime_action`)

Conventions:
- The paper uses tau in [0, 1] where tau=0 is pure noise and tau=1 is clean.
- openpi PyTorch uses `time` in [1, 0] where time=1 is noise and time=0 is
  clean (dt < 0). Internally we convert: ``tau_paper = 1 - time_ours``.
"""
from __future__ import annotations

import math

import torch

PrefixAttentionSchedule = str  # "ones" | "zeros" | "linear" | "exp"


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixAttentionSchedule = "exp",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Soft mask weighting the prefix (unexecuted) portion of the previous chunk.

    With ``start=2, end=6, total=10``, the output is::

        [1, 1, 4/5, 3/5, 2/5, 1/5, 0, 0, 0, 0]
                 ^                  ^
              start              end

    ``start`` (inclusive) is the index where the chunk is allowed to change
    (= inference_delay ``d``). Before ``start`` the actions are guaranteed to
    have executed by the time the new chunk arrives, so the new chunk must
    match exactly. ``end`` (exclusive) is where the chunk stops paying
    attention to the prefix at all. Between [start, end) the new chunk is
    allowed to deviate but with a decaying pull toward the previous chunk's
    values.

    For our deployment: ``start = inference_delay``,
    ``end = action_chunk_size - execute_horizon``, ``total = action_chunk_size``.

    Args:
        start: First index at which weight becomes < 1 (= inference delay d).
        end: First index at which weight is 0 (exclusive).
        total: Chunk length H.
        schedule: "ones" (W=1 everywhere), "zeros" (W=1 only on the frozen
            prefix [0:start]), "linear" (linear ramp from 1 to 0 between
            [start, end)), or "exp" (exponential ramp, the paper's default).
        device, dtype: where the resulting weight vector lives.

    Returns:
        A 1-D weight tensor of shape ``(total,)``.
    """
    start = min(start, end)
    idx = torch.arange(total, device=device, dtype=dtype)
    if schedule == "ones":
        w = torch.ones(total, device=device, dtype=dtype)
    elif schedule == "zeros":
        w = (idx < start).to(dtype)
    elif schedule in ("linear", "exp"):
        # Linear ramp: at idx=start-1 weight=1, decays to 0 over (end-start+1)
        # steps. Equivalent to the Kinetix expression:
        #   w = clip((start - 1 - arange(total)) / (end - start + 1) + 1, 0, 1)
        w = torch.clip((start - 1 - idx) / (end - start + 1) + 1, 0.0, 1.0)
        if schedule == "exp":
            # paper: w * expm1(w) / (e - 1)  --- monotone S-curve in [0, 1]
            w = w * torch.expm1(w) / (math.e - 1.0)
    else:
        raise ValueError(f"Invalid schedule: {schedule!r}")
    return torch.where(idx >= end, torch.zeros_like(w), w)


def guidance_weight_from_time(
    tau: torch.Tensor | float,
    max_guidance_weight: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor | float:
    """Compute the Pi-GDM guidance weight for a flow-matching step at time tau.

    From the paper, with ``r_tau^2 = (1-tau)^2 / (tau^2 + (1-tau)^2)``::

        guidance = min(beta, ((1 - tau) / tau) / r_tau^2)
                 = min(beta, ((1 - tau) / tau) * (tau^2 + (1-tau)^2) / (1-tau)^2)

    The Kinetix code factors this as ``c * inv_r2``::

        inv_r2 = (tau^2 + (1-tau)^2) / (1-tau)^2
        c      = (1 - tau) / tau

    At tau -> 0 (pure noise), ``c -> inf`` -> always clamped to beta.
    At tau -> 1 (clean),     ``c -> 0`` -> guidance vanishes.

    Args:
        tau: Paper-convention timestep in [0, 1]. Scalar tensor or float.
        max_guidance_weight: ``beta`` from the paper (default 5.0).
        eps: numerical floor on denominators.

    Returns:
        Scalar guidance weight (same type as tau in/out).
    """
    if isinstance(tau, torch.Tensor):
        one_minus_t = 1.0 - tau
        inv_r2 = (tau ** 2 + one_minus_t ** 2) / (one_minus_t ** 2 + eps)
        c = one_minus_t / (tau + eps)
        gw = c * inv_r2
        return torch.minimum(gw, torch.tensor(max_guidance_weight, dtype=gw.dtype, device=gw.device))
    # Python float fallback (used in tests / standalone diagnostics).
    one_minus_t = 1.0 - tau
    inv_r2 = (tau ** 2 + one_minus_t ** 2) / (one_minus_t ** 2 + eps)
    c = one_minus_t / (tau + eps)
    return min(c * inv_r2, max_guidance_weight)


def ours_time_to_paper_tau(time_ours: torch.Tensor | float) -> torch.Tensor | float:
    """Map openpi's time (1->0, noise->clean) to paper's tau (0->1)."""
    if isinstance(time_ours, torch.Tensor):
        return 1.0 - time_ours
    return 1.0 - float(time_ours)


__all__ = [
    "PrefixAttentionSchedule",
    "get_prefix_weights",
    "guidance_weight_from_time",
    "ours_time_to_paper_tau",
]
