"""Async, RTC-aware action-chunk broker.

This is the client-side counterpart to ``openpi.models_pytorch.rtc`` on the
server. It implements the runtime described in
"Real-Time Execution of Action Chunking Flow Policies" (Black et al., 2025).

The broker:
- Serves actions from the current chunk on every call to ``infer``.
- Kicks off background inference for the **next** chunk early (when the
  consumer reaches ``start_next_at``, default ``H - inference_delay``).
- Includes the unexecuted suffix of the current chunk PLUS the predicted
  inference delay in the request so the server can run RTC inpainting.
- On chunk arrival, splices: the first ``d`` actions are still from the OLD
  chunk (they were physically already in flight while inference ran), then
  the new chunk takes over.

Unlike the synchronous ``ActionChunkBroker``, the consumer never blocks on
inference: the robot keeps moving while inference happens in the background.

If RTC is not supported by the server (older versions), this broker falls
back to a naive async path -- pipelined inference without the inpainting
correction. The seam may then have small discontinuities; use the standard
``ActionChunkBroker`` instead if you don't want that.

Example::

    base_policy = WebsocketClientPolicy(host="...", port=8002)
    policy = AsyncActionChunkBroker(
        base_policy,
        action_horizon=50,
        execute_horizon=25,        # how many actions to consume before swap
        inference_delay=4,         # ~= measured infer_ms / control_period_ms
        prefix_attention_horizon=25,  # paper default = H - execute_horizon
    )
    for t in range(N):
        action = policy.infer(obs)
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict
import logging
import threading

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

_log = logging.getLogger(__name__)


class AsyncActionChunkBroker(_base_policy.BasePolicy):
    """Async + RTC-aware chunk broker for openpi.

    Parameters
    ----------
    policy:
        The underlying ``BasePolicy`` (typically a ``WebsocketClientPolicy``).
        Its ``infer`` must return a dict with at minimum ``"actions"``
        (shape ``(H, action_dim)``) and ideally ``"_rtc_chunk_model_space"``
        (shape ``(H, action_dim_model)``).
    action_horizon:
        ``H``. Number of actions per chunk.
    execute_horizon:
        ``s``. Number of actions to execute from each chunk before swapping
        to the next one. Must satisfy ``inference_delay <= s <= H``.
    inference_delay:
        ``d``. Number of control timesteps the server inference is expected
        to take. We start a new inference call at index ``s - d`` of the
        current chunk so that the new chunk arrives just as the old one is
        being exhausted at index ``s``.
    prefix_attention_horizon:
        Index past which the new chunk's actions are NOT guided by the prev
        chunk. The paper sets this to ``H - s`` (e.g. ``H=50``, ``s=25`` ->
        ``pah=25``). With ``pah=H`` the entire chunk is softly pulled toward
        the prev chunk.
    schedule:
        Soft-mask schedule for the prefix weights: ``"exp"`` (default),
        ``"linear"``, ``"ones"``, or ``"zeros"``.
    max_guidance_weight:
        ``beta`` from the paper, default 5.0.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        execute_horizon: int | None = None,
        inference_delay: int = 4,
        prefix_attention_horizon: int | None = None,
        schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        enable_rtc: bool = True,
    ) -> None:
        self._policy = policy
        self._H = action_horizon
        self._s = int(execute_horizon if execute_horizon is not None else action_horizon // 2)
        self._d = int(inference_delay)
        if not (0 <= self._d <= self._s <= self._H):
            raise ValueError(
                f"require 0 <= inference_delay={self._d} <= execute_horizon={self._s} <= action_horizon={self._H}"
            )
        # Default: full overlap from d up to H-s, then free tail.
        self._pah = int(prefix_attention_horizon if prefix_attention_horizon is not None else self._H - self._s)
        if not (self._d <= self._pah <= self._H):
            raise ValueError(
                f"require inference_delay={self._d} <= prefix_attention_horizon={self._pah} <= action_horizon={self._H}"
            )
        self._schedule = schedule
        self._beta = float(max_guidance_weight)
        # If False, the broker pipelines inferences asynchronously but never
        # asks the server for RTC inpainting (no ``_rtc_*`` keys in the
        # request). Use this when RTC inference is slower than the chunk
        # consumption time -- async pipelining alone still removes the
        # idle-at-seam wobble, and the seam discontinuities tend to be small
        # in practice because successive obs are very similar.
        self._enable_rtc = bool(enable_rtc)

        # Threading and chunk state.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-async")
        self._lock = threading.Lock()
        self._current: Dict[str, Any] | None = None  # current chunk's full result dict
        self._cur_step: int = 0
        self._pending: Future | None = None
        self._kickoff_index: int = self._s - self._d  # absolute index in current chunk
        self._rtc_used_last = False
        self._closed = False

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._lock:
            self._raise_if_closed()
            if self._current is None:
                # First call -- block to produce the initial chunk
                # synchronously. We route this through the SAME executor that
                # background calls will use, so the underlying websocket is
                # only ever touched by one thread. (Sharing the websockets.sync
                # ClientConnection across threads -- main for the first call,
                # background for the rest -- leaves the library in a state
                # where ``Future.done()`` from the background never flips True
                # until the main thread also blocks on it.)
                f = self._executor.submit(self._policy.infer, self._strip_rtc_keys(obs))
                self._current = f.result()
                self._cur_step = 0
                self._maybe_log_first(obs)

            # If a background inference finished since last call, swap it in.
            self._promote_ready_locked()

            # Kick off background inference for the NEXT chunk if we've
            # crossed the threshold AND no inference is in flight.
            if self._cur_step >= self._kickoff_index and self._pending is None and self._cur_step < self._H:
                self._submit_locked(obs)

            # Serve one action from the current chunk.
            if self._cur_step >= self._H:
                # Past the end of the chunk; the background call should have
                # arrived. If not, block on it. This degenerates to
                # synchronous behavior for one chunk and indicates the
                # configured ``inference_delay`` underestimates actual
                # server latency. _block_for_pending_locked resets _cur_step
                # to ``d`` itself.
                self._block_for_pending_locked()

            result = self._slice_at(self._current, self._cur_step)
            self._cur_step += 1
            return result

    @override
    def reset(self) -> None:
        with self._lock:
            self._raise_if_closed()
            if self._pending is not None:
                self._pending.cancel()
            self._policy.reset()
            self._current = None
            self._cur_step = 0
            self._pending = None
            self._rtc_used_last = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "AsyncActionChunkBroker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("AsyncActionChunkBroker is closed")

    def _strip_rtc_keys(self, obs: Dict) -> Dict:
        """Strip any RTC keys the caller may have included by accident."""
        return {k: v for k, v in obs.items() if not k.startswith("_rtc_")}

    def _submit_locked(self, obs: Dict) -> None:
        """Kick off background inference using the unexecuted suffix as RTC prev."""
        import time as _t  # noqa: PLC0415
        self._submit_t = _t.perf_counter()
        cur = self._current
        if not self._enable_rtc or cur is None or "_rtc_chunk_model_space" not in cur:
            # No RTC: server doesn't support it, RTC disabled, or first-call
            # fallback. Background inference still helps with pipelining
            # (avoids the synchronous stall); seam discontinuities tend to be
            # small in practice because successive obs are very similar.
            req = self._strip_rtc_keys(obs)
        else:
            prev_full = np.asarray(cur["_rtc_chunk_model_space"], dtype=np.float32)
            # The "prev chunk" that the server should treat as the previous
            # plan is the full model-space chunk (H, D). We tell the server
            # how many controller timesteps will have elapsed by the time the
            # response arrives -- that's `inference_delay`.
            req = dict(self._strip_rtc_keys(obs))
            req["_rtc_prev_chunk"] = prev_full
            req["_rtc_inference_delay"] = int(self._d)
            req["_rtc_prefix_attention_horizon"] = int(self._pah)
            req["_rtc_schedule"] = str(self._schedule)
            req["_rtc_max_guidance_weight"] = float(self._beta)

        self._pending = self._executor.submit(self._policy.infer, req)

    def _promote_ready_locked(self) -> None:
        """If background inference finished, stash it; swap when at the seam.

        Two-step dance:
        1) If the pending future is done, harvest it into ``_pending_chunk``
           and clear ``_pending``. (This may happen well BEFORE we're ready
           to swap, e.g. if RTC inference is faster than the remaining chunk
           tail.)
        2) If we already have a stashed ``_pending_chunk`` AND the consumer
           has reached the swap index ``s``, perform the splice. We must do
           this on every call (not just when the future completes) because
           the future completes once, but the swap-point check is per-tick.
        """
        if self._pending is not None and self._pending.done():
            try:
                new_chunk = self._pending.result()
            except Exception as e:  # noqa: BLE001
                _log.exception(f"RTC background inference failed: {e!r}; keeping current chunk")
                self._pending = None
                return
            self._pending = None
            self._pending_chunk = new_chunk  # type: ignore[attr-defined]

        # Swap when we've reached the seam, regardless of whether the
        # pending future completed this tick or earlier.
        if getattr(self, "_pending_chunk", None) is not None and self._cur_step >= self._s:
            self._swap_locked()

    def _swap_locked(self) -> None:
        new_chunk = getattr(self, "_pending_chunk", None)
        if new_chunk is None:
            return
        self._current = new_chunk
        # The new chunk's index 0 corresponds to the global time of OLD index
        # `s`. We start executing the new chunk from its index `s - cur_step`
        # offset... wait, no. With proper RTC, the new chunk's index 0 was
        # PLANNED for the same global time as OLD chunk index 0. But we
        # already executed [0:s] of the old chunk. So we should jump into the
        # new chunk at index `s`. But then we'd serve only H-s actions before
        # needing a new chunk... HMMMM.
        # Actually -- re-reading Kinetix eval_flow.py: they shift the new
        # chunk by `execute_horizon` after swapping so its index 0 maps to
        # the next chunk's "starting point". We do that here:
        #   - At swap time we are at global step `s` (of the old chunk).
        #   - The new chunk was generated based on observations at step `s-d`
        #     (we sent the request `d` steps ago) -- so new_chunk[0]
        #     corresponds to global step `s-d`, and new_chunk[d] is where
        #     the robot will *actually be next* (step `s`).
        # Therefore: start serving from new_chunk[d] and our effective
        # execute_horizon on the new chunk is `s - d` actions until the
        # next swap, OR the kickoff index becomes `2*s - d` etc.
        # Simpler model used here: start serving from new_chunk[d], reset
        # `_cur_step = d`, and set the next kickoff at `_cur_step = s`.
        self._cur_step = self._d
        self._kickoff_index = self._s
        try:
            delattr(self, "_pending_chunk")
        except AttributeError:
            pass
        if new_chunk.get("_rtc_used"):
            self._rtc_used_last = True

    def _block_for_pending_locked(self) -> None:
        if self._pending is None:
            raise RuntimeError(
                f"Action chunk exhausted (cur_step={self._cur_step}, H={self._H}) "
                "but no background inference in flight. Check execute_horizon vs "
                "action_horizon."
            )
        try:
            self._current = self._pending.result()
        except Exception:
            self._pending = None
            raise
        self._pending = None
        self._cur_step = self._d
        self._kickoff_index = self._s

    def _slice_at(self, chunk_result: Dict, idx: int) -> Dict:
        """Return chunk_result with all (H, ...) arrays sliced at index `idx`."""

        def slicer(x):
            if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == self._H:
                return x[idx, ...]
            return x

        sliced = {k: slicer(v) for k, v in chunk_result.items() if not k.startswith("_rtc_")}
        # Preserve non-array metadata fields (timing, etc.) at every step.
        return sliced

    def _maybe_log_first(self, obs: Dict) -> None:
        _log.info(
            "AsyncActionChunkBroker: first chunk served. H=%d s=%d d=%d pah=%d schedule=%s beta=%.1f",
            self._H, self._s, self._d, self._pah, self._schedule, self._beta,
        )
        if self._current is not None and "_rtc_chunk_model_space" not in self._current:
            _log.warning(
                "Server response did not include '_rtc_chunk_model_space'; RTC will be DISABLED "
                "for this connection. Subsequent calls will pipeline async but without ΠGDM guidance."
            )
