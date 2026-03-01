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
# Enable on DGX Spark to avoid OOM during checkpoint saves
SAVE_LORA_ONLY = False

# Reduce Orbax checkpoint manager verbosity
logging.getLogger("orbax").setLevel(logging.WARNING)
logging.getLogger("orbax.checkpoint").setLevel(logging.WARNING)


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
    # Convert flax.nnx.State to pure dict if necessary
    if hasattr(params, 'to_pure_dict'):
        params_dict = params.to_pure_dict()
    else:
        params_dict = params
    
    flat = flax.traverse_util.flatten_dict(params_dict, sep="/")
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


def _keypath_to_str(keypath) -> str:
    """Convert JAX KeyPath to "/" separated string.
    
    Handles nnx.State which has .value attributes for Variables.
    Strips the '.value' suffix for matching against checkpoint paths.
    """
    parts = []
    for key in keypath:
        # Handle different KeyEntry types
        if hasattr(key, 'key'):
            # DictKey, GetAttrKey
            key_str = str(key.key)
            # Skip .value attributes (nnx.Variable internals)
            # GetAttrKey has '.value' (with dot), also check without dot
            if key_str in ('value', '.value'):
                continue
            parts.append(key_str)
        elif hasattr(key, 'idx'):
            # SequenceKey, FlattenedIndexKey
            parts.append(str(key.idx))
        else:
            # Fallback
            key_str = str(key)
            if key_str in ('value', '.value'):
                continue
            parts.append(key_str)
    return "/".join(parts)


def _merge_lora_params(base_params: at.Params, lora_params: dict) -> at.Params:
    """Merge LoRA params from checkpoint into base model params.
    
    The checkpoint only contains LoRA weights. We need to overlay them
    onto the initialized base model parameters, preserving the original type.
    
    Uses jax.tree_util to traverse and update while preserving pytree structure.
    """
    # Flatten the LoRA params for lookup
    lora_flat = flax.traverse_util.flatten_dict(lora_params, sep="/")
    
    if not lora_flat:
        logging.warning("No LoRA params to merge")
        return base_params
    
    # Debug: log first few LoRA keys
    lora_keys = list(lora_flat.keys())
    logging.info(f"LoRA checkpoint has {len(lora_keys)} params")
    if lora_keys:
        logging.info(f"Sample LoRA keys: {lora_keys[:3]}")
    
    merged_count = [0]  # Use list to allow mutation in nested function
    seen_lora_paths = []  # Track paths that contain 'lora'
    
    def update_leaf(path, leaf):
        """Update leaf if it matches a LoRA param path."""
        path_str = _keypath_to_str(path)
        
        # Track LoRA-related paths for debugging
        if 'lora' in path_str.lower() and len(seen_lora_paths) < 5:
            seen_lora_paths.append(path_str)
        
        if path_str in lora_flat:
            merged_count[0] += 1
            return lora_flat[path_str]
        return leaf
    
    # Use jax.tree_util.tree_map_with_path to update leaves while preserving structure
    merged = jax.tree_util.tree_map_with_path(
        update_leaf,
        base_params,
    )
    
    # Debug: log path format if no matches
    if merged_count[0] == 0 and seen_lora_paths:
        logging.warning(f"No path matches! Sample base model lora paths: {seen_lora_paths}")
    
    logging.info(f"Merged {merged_count[0]} LoRA params from checkpoint into base model")
    return merged


def _get_checkpoint_step_dir(checkpoint_dir: epath.Path, step: int) -> epath.Path | None:
    """Find the checkpoint directory for a given step."""
    step_dir = checkpoint_dir / str(step)
    if step_dir.exists():
        return step_dir
    # Try with orbax suffix patterns
    candidates = list(checkpoint_dir.glob(f"{step}*"))
    if candidates:
        return candidates[0]
    return None


def _is_lora_only_checkpoint(checkpoint_dir: epath.Path, step: int) -> bool:
    """Check if a checkpoint is LoRA-only by examining its structure."""
    step_dir = _get_checkpoint_step_dir(checkpoint_dir, step)
    if step_dir is None:
        return False
    
    # Check for minimal train_state (indicates LoRA-only save)
    train_state_dir = step_dir / "train_state"
    if train_state_dir.exists():
        # LoRA-only checkpoints have minimal train_state with just {"step": N}
        # Full checkpoints have nested optimizer state with array_metadatas
        # Check direct subdirectory for array_metadatas (avoid ** which isn't supported by epath)
        array_metadatas_dir = train_state_dir / "array_metadatas"
        if not array_metadatas_dir.exists():
            # No array_metadatas directory = minimal train_state = LoRA-only
            return True
    return False


def _read_checkpoint_metadata(checkpoint_dir: epath.Path, step: int) -> dict | None:
    """Read the checkpoint metadata to discover saved structure."""
    import json
    
    step_dir = _get_checkpoint_step_dir(checkpoint_dir, step)
    if step_dir is None:
        return None
    
    params_metadata_path = step_dir / "params" / "_METADATA"
    if not params_metadata_path.exists():
        return None
    
    try:
        with open(params_metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read checkpoint metadata: {e}")
        return None


def _build_restore_template_from_metadata(metadata: dict, base_params: at.Params) -> dict:
    """Build a restore template that matches the checkpoint's saved structure.
    
    Uses the checkpoint metadata to discover exact paths, then extracts matching
    values from the base model to create a structurally-compatible template.
    """
    import numpy as np
    
    if hasattr(base_params, 'to_pure_dict'):
        base_dict = base_params.to_pure_dict()
    else:
        base_dict = base_params
    
    base_flat = flax.traverse_util.flatten_dict(base_dict, sep="/")
    
    tree_metadata = metadata.get("tree_metadata", {})
    template_flat = {}
    
    for path_tuple_str, meta in tree_metadata.items():
        # Parse the path tuple string like "('params', 'PaliGemma', ...)"
        # Convert to our "/" separated format
        try:
            # Safely evaluate the tuple string
            import ast
            path_tuple = ast.literal_eval(path_tuple_str)
            # Skip the leading 'params' if present (it's added by orbax wrapper)
            if path_tuple and path_tuple[0] == 'params':
                path_tuple = path_tuple[1:]
            path_key = "/".join(path_tuple)
            
            # Get shape info from metadata
            value_meta = meta.get("value_metadata", {})
            shape = value_meta.get("write_shape", [])
            
            # Try to find matching param in base model
            if path_key in base_flat:
                template_flat[path_key] = base_flat[path_key]
            else:
                # Create placeholder with correct shape for restore
                # orbax will fill in the actual values
                template_flat[path_key] = np.zeros(shape, dtype=np.float32)
                
        except Exception as e:
            logging.warning(f"Failed to parse metadata path {path_tuple_str}: {e}")
            continue
    
    logging.info(f"Built restore template with {len(template_flat)} params from checkpoint metadata")
    return flax.traverse_util.unflatten_dict(template_flat, sep="/")


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
        
        # Determine which step to restore
        restore_step = step if step is not None else checkpoint_manager.latest_step()
        
        # Check if this is a LoRA-only checkpoint
        checkpoint_dir = epath.Path(checkpoint_manager.directory)
        is_lora_only = _is_lora_only_checkpoint(checkpoint_dir, restore_step)
        
        if is_lora_only and SAVE_LORA_ONLY:
            logging.info(f"Restoring LoRA-only checkpoint from step {restore_step}")
            
            # Read checkpoint metadata to discover exact saved structure
            metadata = _read_checkpoint_metadata(checkpoint_dir, restore_step)
            
            if metadata is not None:
                # Build restore template from checkpoint metadata (not from initialized model)
                # This ensures exact structure match with what was saved
                lora_template = _build_restore_template_from_metadata(metadata, params)
                logging.info("Using metadata-based restore template")
            else:
                # Fallback: use initialized model's LoRA structure
                logging.warning("Could not read checkpoint metadata, using initialized model structure")
                lora_template = _filter_lora_params(params)
            
            minimal_train_state = {"step": 0}
            
            try:
                restored = checkpoint_manager.restore(
                    restore_step,
                    items={
                        "train_state": minimal_train_state,
                        "params": {"params": lora_template},
                    },
                )
            except ValueError as e:
                logging.error(f"LoRA checkpoint restore failed: {e}")
                logging.error("The checkpoint structure does not match. "
                             "Try starting fresh with --overwrite instead of --resume.")
                raise
            
            # Merge restored LoRA params into base model
            restored_lora_params = restored["params"].get("params", restored["params"])
            merged_params = _merge_lora_params(params, restored_lora_params)
            
            # Restore step count from checkpoint, keep optimizer state from initialized model
            restored_step = restored["train_state"].get("step", 0)
            logging.info(f"Restored LoRA weights, resuming from step {restored_step}")
            
            # Update train_state with restored step but keep optimizer state
            if hasattr(train_state, 'step'):
                train_state = dataclasses.replace(train_state, step=restored_step)
            
            return _merge_params(train_state, {"params": merged_params})
        else:
            # Standard full checkpoint restore
            restored = checkpoint_manager.restore(
                restore_step,
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
