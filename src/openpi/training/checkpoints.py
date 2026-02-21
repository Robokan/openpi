from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import re
from typing import Protocol

from etils import epath
import flax.traverse_util
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


# Flag to enable LoRA-only checkpoint saving (saves memory on unified memory systems)
SAVE_LORA_ONLY = True


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Limit concurrent GB during save/restore to prevent RAM OOM on unified memory systems
    # See: https://github.com/Physical-Intelligence/openpi/issues/827
    pytree_handler = ocp.PyTreeCheckpointHandler(
        save_concurrent_gb=16,  # With LoRA-only saves, checkpoints are small (~100MB) so this is plenty
        restore_concurrent_gb=16,
    )
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": pytree_handler,
            "params": pytree_handler,
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def _filter_lora_params(params: at.Params) -> at.Params:
    """Filter params to only include LoRA weights (params with 'lora' in their path).
    
    This dramatically reduces checkpoint size and memory usage during save.
    """
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    lora_pattern = re.compile(r".*lora.*", re.IGNORECASE)
    filtered = {k: v for k, v in flat.items() if lora_pattern.match(k)}
    
    if not filtered:
        logging.warning("No LoRA params found - saving empty params dict. "
                       "This is expected if not using LoRA fine-tuning.")
        return {}
    
    logging.info(f"Filtered to {len(filtered)} LoRA param tensors for checkpoint save")
    return flax.traverse_util.unflatten_dict(filtered, sep="/")


def _filter_lora_opt_state(opt_state) -> dict:
    """Filter optimizer state to only include LoRA-related entries."""
    # Optimizer state is a nested structure - we need to filter it carefully
    # For simplicity, we'll just save the step count and let optimizer reinitialize
    # The LoRA weights themselves are the important part
    return {"step": getattr(opt_state, "count", 0) if hasattr(opt_state, "count") else 0}


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Clear JAX caches before save to free up memory for checkpoint serialization
    # This helps prevent OOM on unified memory systems like DGX Spark
    jax.clear_caches()
    
    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    
    # If SAVE_LORA_ONLY is enabled, filter to only save LoRA weights
    # This dramatically reduces memory usage during checkpoint save
    if SAVE_LORA_ONLY:
        logging.info("Saving LoRA-only checkpoint (SAVE_LORA_ONLY=True)")
        params = _filter_lora_params(params)
        # For train_state, we save minimal info since optimizer state is large
        # The full optimizer state will be reinitialized on resume
        minimal_train_state = {
            "step": int(train_state.step),
        }
        items = {
            "assets": save_assets,
            "train_state": minimal_train_state,
            "params": {"params": params},
        }
    else:
        items = {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        }
    
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
