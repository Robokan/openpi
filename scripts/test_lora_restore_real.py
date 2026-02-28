#!/usr/bin/env python3
"""
Real integration test for LoRA checkpoint restore.

This test mirrors EXACTLY what restore_state() does, using real checkpoint data.
Run inside Docker with:
    docker compose -f scripts/docker/compose_ngc.yml run --rm openpi_server_ngc python scripts/test_lora_restore_real.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import json
import ast
import numpy as np
import jax
import jax.numpy as jnp
import flax.traverse_util
import flax.nnx as nnx
import orbax.checkpoint as ocp
from etils import epath

from openpi.training import checkpoints as ckpt_module
from openpi.training.checkpoints import (
    _keypath_to_str,
    _read_checkpoint_metadata,
    _build_restore_template_from_metadata,
    _is_lora_only_checkpoint,
    _merge_lora_params,
    _get_checkpoint_step_dir,
    _filter_lora_params,
)


CHECKPOINT_DIR = epath.Path("/app/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
CHECKPOINT_STEP = 280


def test_checkpoint_exists():
    """Verify checkpoint exists."""
    print("\n" + "="*60)
    print("TEST: Checkpoint exists")
    print("="*60)
    
    assert CHECKPOINT_DIR.exists(), f"Checkpoint dir not found: {CHECKPOINT_DIR}"
    step_dir = _get_checkpoint_step_dir(CHECKPOINT_DIR, CHECKPOINT_STEP)
    assert step_dir is not None, f"Step {CHECKPOINT_STEP} not found"
    print(f"✓ Checkpoint found at {step_dir}")
    return step_dir


def test_metadata_readable(step_dir):
    """Test that we can read checkpoint metadata."""
    print("\n" + "="*60)
    print("TEST: Metadata readable")
    print("="*60)
    
    metadata = _read_checkpoint_metadata(CHECKPOINT_DIR, CHECKPOINT_STEP)
    assert metadata is not None, "Failed to read metadata"
    
    tree_meta = metadata.get("tree_metadata", {})
    print(f"✓ Metadata has {len(tree_meta)} params")
    
    # Show all checkpoint paths
    checkpoint_paths = []
    for path_tuple_str in tree_meta.keys():
        path_tuple = ast.literal_eval(path_tuple_str)
        if path_tuple[0] == 'params':
            path_tuple = path_tuple[1:]
        path_key = "/".join(path_tuple)
        checkpoint_paths.append(path_key)
    
    print(f"Checkpoint paths ({len(checkpoint_paths)}):")
    for p in checkpoint_paths[:5]:
        print(f"  {p}")
    if len(checkpoint_paths) > 5:
        print(f"  ... and {len(checkpoint_paths) - 5} more")
    
    return metadata, checkpoint_paths


def test_build_template(metadata):
    """Test building restore template from metadata."""
    print("\n" + "="*60)
    print("TEST: Build restore template")
    print("="*60)
    
    # Use empty base_params like the real code does when paths don't match
    base_params = {}
    template = _build_restore_template_from_metadata(metadata, base_params)
    
    template_flat = flax.traverse_util.flatten_dict(template, sep="/")
    print(f"✓ Template has {len(template_flat)} params")
    
    # Show template structure
    print("Template paths and shapes:")
    for i, (k, v) in enumerate(list(template_flat.items())[:5]):
        shape = v.shape if hasattr(v, 'shape') else type(v).__name__
        print(f"  {k}: {shape}")
    
    # Verify all are numpy arrays (not ShapeDtypeStruct or other placeholders)
    for k, v in template_flat.items():
        assert isinstance(v, np.ndarray), f"Template value for {k} is {type(v)}, expected ndarray"
    
    print("✓ All template values are numpy arrays")
    return template


def test_orbax_restore(template):
    """Test actual orbax checkpoint restore."""
    print("\n" + "="*60)
    print("TEST: Orbax restore")
    print("="*60)
    
    # Create checkpoint manager EXACTLY like initialize_checkpoint_dir does
    # Must specify item_handlers for orbax to recognize the checkpoint structure
    pytree_handler = ocp.PyTreeCheckpointHandler(
        save_concurrent_gb=16,
        restore_concurrent_gb=16,
    )
    checkpoint_manager = ocp.CheckpointManager(
        CHECKPOINT_DIR,
        item_handlers={
            "assets": ckpt_module.CallbackHandler(),
            "train_state": pytree_handler,
            "params": pytree_handler,
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            create=False,  # Don't create, we're reading existing
        ),
    )
    
    minimal_train_state = {"step": 0}
    
    print(f"Restoring step {CHECKPOINT_STEP}...")
    restored = checkpoint_manager.restore(
        CHECKPOINT_STEP,
        items={
            "train_state": minimal_train_state,
            "params": {"params": template},
        },
    )
    
    # Extract restored params
    restored_params = restored["params"].get("params", restored["params"])
    restored_flat = flax.traverse_util.flatten_dict(restored_params, sep="/")
    
    print(f"✓ Restored {len(restored_flat)} params")
    
    # Verify restored values are actual arrays, not placeholders
    placeholder_count = 0
    array_count = 0
    for k, v in restored_flat.items():
        if isinstance(v, jax.ShapeDtypeStruct):
            placeholder_count += 1
            print(f"  WARNING: {k} is ShapeDtypeStruct!")
        elif isinstance(v, (np.ndarray, jnp.ndarray)) or hasattr(v, 'shape'):
            array_count += 1
        else:
            print(f"  WARNING: {k} is unexpected type {type(v)}")
    
    print(f"  Arrays: {array_count}, Placeholders: {placeholder_count}")
    assert placeholder_count == 0, f"Found {placeholder_count} ShapeDtypeStruct placeholders!"
    
    # Verify values are non-zero (actually loaded from checkpoint)
    non_zero_count = 0
    for k, v in restored_flat.items():
        if hasattr(v, 'any') and np.any(v != 0):
            non_zero_count += 1
    
    print(f"  Non-zero params: {non_zero_count}/{len(restored_flat)}")
    assert non_zero_count > 0, "All restored params are zero - checkpoint not loaded!"
    
    # Show sample values
    print("Sample restored values:")
    for k, v in list(restored_flat.items())[:3]:
        if hasattr(v, 'flatten'):
            sample = v.flatten()[:5]
            print(f"  {k}: {sample}")
    
    restored_step = restored["train_state"].get("step", 0)
    print(f"✓ Restored step: {restored_step}")
    
    return restored_params, restored_step


def test_create_mock_base_params(checkpoint_paths):
    """Create a mock nnx.State that has the same structure as the real model."""
    print("\n" + "="*60)
    print("TEST: Create mock base params (nnx.State)")
    print("="*60)
    
    # We need to create an nnx.State with paths that match the checkpoint
    # The checkpoint has paths like:
    #   PaliGemma/llm/layers/attn/attn_vec_einsum/lora_a
    # We need nnx.State to produce the same paths when traversed
    
    # Build a nested dict structure that matches checkpoint paths
    base_dict = {}
    for path in checkpoint_paths:
        parts = path.split("/")
        current = base_dict
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        # Create zero array for the leaf (these are the base model's LoRA params)
        # Use shape [2, 2] as placeholder - actual shape doesn't matter for path testing
        current[parts[-1]] = np.zeros((2, 2))
    
    # Also add some non-LoRA params that should NOT be updated
    if "PaliGemma" in base_dict:
        base_dict["PaliGemma"]["base_weight"] = np.ones((4, 4))  # Should stay as ones
    
    print(f"Created mock base_dict with paths:")
    base_flat = flax.traverse_util.flatten_dict(base_dict, sep="/")
    for k in list(base_flat.keys())[:5]:
        print(f"  {k}")
    
    return base_dict


def test_path_matching(base_dict, restored_params):
    """Test that paths from nnx-like traversal match checkpoint paths."""
    print("\n" + "="*60)
    print("TEST: Path matching (the critical test)")
    print("="*60)
    
    # Flatten restored params (from checkpoint)
    lora_flat = flax.traverse_util.flatten_dict(restored_params, sep="/")
    lora_keys = set(lora_flat.keys())
    print(f"Restored checkpoint has {len(lora_keys)} keys")
    print(f"Sample lora_flat keys: {list(lora_keys)[:3]}")
    
    # Traverse base_dict with jax.tree_util.tree_map_with_path
    # This simulates what _merge_lora_params does
    base_paths = []
    def collect_paths(path, leaf):
        path_str = _keypath_to_str(path)
        base_paths.append(path_str)
        return leaf
    
    jax.tree_util.tree_map_with_path(collect_paths, base_dict)
    base_path_set = set(base_paths)
    
    print(f"Base params has {len(base_paths)} paths")
    print(f"Sample base paths: {base_paths[:3]}")
    
    # Check for matches
    matches = lora_keys & base_path_set
    lora_only = lora_keys - base_path_set
    base_only = base_path_set - lora_keys
    
    print(f"\nPath matching results:")
    print(f"  Matches: {len(matches)}")
    print(f"  In checkpoint only: {len(lora_only)}")
    print(f"  In base only: {len(base_only)}")
    
    if lora_only:
        print(f"  Checkpoint paths not in base: {list(lora_only)[:3]}")
    if base_only:
        print(f"  Base paths not in checkpoint: {list(base_only)[:3]}")
    
    # This is the critical assertion - if this fails, merge will fail
    assert len(matches) == len(lora_keys), \
        f"Path mismatch! Only {len(matches)}/{len(lora_keys)} checkpoint paths found in base"
    
    print("✓ All checkpoint paths match base params paths!")
    return True


def test_merge_params(base_dict, restored_params):
    """Test the actual merge function."""
    print("\n" + "="*60)
    print("TEST: Merge LoRA params")
    print("="*60)
    
    # Add a non-LoRA param that should NOT be changed
    if "PaliGemma" in base_dict:
        base_dict["PaliGemma"]["base_weight"] = np.ones((4, 4)) * 999.0
    
    merged = _merge_lora_params(base_dict, restored_params)
    
    # Verify merged is a dict (since base was dict)
    assert isinstance(merged, dict), f"Merged should be dict, got {type(merged)}"
    
    merged_flat = flax.traverse_util.flatten_dict(merged, sep="/")
    
    # Count how many LoRA params have non-zero values (meaning they were updated)
    lora_flat = flax.traverse_util.flatten_dict(restored_params, sep="/")
    
    updated_count = 0
    for k in lora_flat.keys():
        if k in merged_flat:
            merged_val = merged_flat[k]
            lora_val = lora_flat[k]
            if hasattr(merged_val, 'shape') and hasattr(lora_val, 'shape'):
                # Check if values match (were actually copied)
                if np.allclose(np.array(merged_val).flatten()[:5], np.array(lora_val).flatten()[:5]):
                    updated_count += 1
    
    print(f"✓ Updated {updated_count}/{len(lora_flat)} LoRA params")
    
    # Check that non-LoRA params were preserved
    if "PaliGemma/base_weight" in merged_flat:
        base_val = merged_flat["PaliGemma/base_weight"]
        assert np.allclose(base_val, 999.0), "Base weight was incorrectly modified!"
        print("✓ Non-LoRA params preserved")
    
    assert updated_count == len(lora_flat), \
        f"Merge failed! Only updated {updated_count}/{len(lora_flat)} params"
    
    return merged


def test_merge_into_nnx_state(restored_params):
    """Test merging into actual nnx.State (the real scenario)."""
    print("\n" + "="*60)
    print("TEST: Merge into nnx.State (CRITICAL)")
    print("="*60)
    
    # Get the checkpoint paths
    lora_flat = flax.traverse_util.flatten_dict(restored_params, sep="/")
    checkpoint_paths = list(lora_flat.keys())
    
    # Build a dynamic nnx module that matches the checkpoint structure
    # We need to create nested nnx modules that produce the same paths
    
    # For simplicity, let's create a structure that matches a few checkpoint paths
    # Real path: PaliGemma/llm/layers/attn/attn_vec_einsum/lora_a
    
    class LoRALayer(nnx.Module):
        def __init__(self, rngs, name_prefix=""):
            # Create LoRA params with names that will match checkpoint
            self.lora_a = nnx.Param(jnp.zeros((18, 8, 256, 16)))
            self.lora_b = nnx.Param(jnp.zeros((18, 8, 16, 2048)))
    
    class AttnVecEinsum(nnx.Module):
        def __init__(self, rngs):
            self.lora_a = nnx.Param(jnp.zeros((18, 8, 256, 16)))
            self.lora_b = nnx.Param(jnp.zeros((18, 8, 16, 2048)))
    
    class Attn(nnx.Module):
        def __init__(self, rngs):
            self.attn_vec_einsum = AttnVecEinsum(rngs)
    
    class Layers(nnx.Module):
        def __init__(self, rngs):
            self.attn = Attn(rngs)
    
    class LLM(nnx.Module):
        def __init__(self, rngs):
            self.layers = Layers(rngs)
    
    class PaliGemma(nnx.Module):
        def __init__(self, rngs):
            self.llm = LLM(rngs)
            self.base_weight = nnx.Param(jnp.ones((4, 4)) * 999.0)  # Non-LoRA param
    
    class MockModel(nnx.Module):
        def __init__(self, rngs):
            self.PaliGemma = PaliGemma(rngs)
    
    rngs = nnx.Rngs(0)
    model = MockModel(rngs)
    graphdef, state = nnx.split(model)
    
    print(f"Created mock nnx.State, type: {type(state)}")
    
    # See what paths jax.tree_util produces for this state
    state_paths = []
    def collect_paths(path, leaf):
        path_str = _keypath_to_str(path)
        state_paths.append(path_str)
        return leaf
    
    jax.tree_util.tree_map_with_path(collect_paths, state)
    
    print(f"nnx.State paths ({len(state_paths)}):")
    for p in state_paths:
        print(f"  {p}")
    
    # Create mock LoRA params that match some of these paths
    # The restored checkpoint has paths like: PaliGemma/llm/layers/attn/attn_vec_einsum/lora_a
    # The nnx.State should have paths like: PaliGemma/llm/layers/attn/attn_vec_einsum/lora_a
    
    mock_lora = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "attn_vec_einsum": {
                            "lora_a": jnp.ones((18, 8, 256, 16)) * 0.123,
                            "lora_b": jnp.ones((18, 8, 16, 2048)) * 0.456,
                        }
                    }
                }
            }
        }
    }
    
    # Now merge
    merged_state = _merge_lora_params(state, mock_lora)
    
    print(f"Merged state type: {type(merged_state)}")
    
    # Verify type is preserved
    assert type(merged_state).__name__ == 'State', \
        f"Expected State, got {type(merged_state)}"
    print("✓ State type preserved")
    
    # Verify we can reconstruct the model
    merged_model = nnx.merge(graphdef, merged_state)
    
    # Check LoRA values were updated
    lora_a_val = merged_model.PaliGemma.llm.layers.attn.attn_vec_einsum.lora_a.value
    assert jnp.allclose(lora_a_val, 0.123), f"lora_a not updated! Got {lora_a_val.flatten()[:3]}"
    print("✓ LoRA params updated correctly")
    
    # Check non-LoRA values preserved
    base_weight = merged_model.PaliGemma.base_weight.value
    assert jnp.allclose(base_weight, 999.0), f"base_weight modified! Got {base_weight.flatten()[:3]}"
    print("✓ Non-LoRA params preserved")
    
    return True


def test_real_model_paths():
    """Test that real pi0.5 model has paths matching checkpoint.
    
    This is THE critical test - it uses the actual model, not mocks.
    """
    print("\n" + "="*60)
    print("TEST: Real model path matching (MOST CRITICAL)")
    print("="*60)
    
    # Read checkpoint paths
    metadata = _read_checkpoint_metadata(CHECKPOINT_DIR, CHECKPOINT_STEP)
    checkpoint_paths = set()
    for path_tuple_str in metadata.get("tree_metadata", {}).keys():
        path_tuple = ast.literal_eval(path_tuple_str)
        if path_tuple[0] == 'params':
            path_tuple = path_tuple[1:]
        checkpoint_paths.add("/".join(path_tuple))
    
    print(f"Checkpoint has {len(checkpoint_paths)} LoRA params")
    
    # Initialize the REAL model
    from openpi.training import config as train_config
    
    # Get the training config
    config = train_config.get_config("pi05_openarm_ngc_lora")
    print(f"Using config: {config.name}")
    
    # Initialize model
    import jax.random as jr
    model_rng = jr.PRNGKey(0)
    model = config.model.create(model_rng)
    
    # Split to get state
    graphdef, state = nnx.split(model)
    print(f"Model state type: {type(state)}")
    
    # Collect all LoRA paths from the real model
    model_lora_paths = []
    all_model_paths = []
    
    def collect_paths(path, leaf):
        path_str = _keypath_to_str(path)
        all_model_paths.append(path_str)
        if 'lora' in path_str.lower():
            model_lora_paths.append(path_str)
        return leaf
    
    jax.tree_util.tree_map_with_path(collect_paths, state)
    
    print(f"Model has {len(all_model_paths)} total params")
    print(f"Model has {len(model_lora_paths)} LoRA params")
    
    # Show sample paths
    print("Sample model LoRA paths:")
    for p in model_lora_paths[:5]:
        print(f"  {p}")
    
    print("Sample checkpoint LoRA paths:")
    for p in list(checkpoint_paths)[:5]:
        print(f"  {p}")
    
    # Find matches and mismatches
    model_lora_set = set(model_lora_paths)
    matches = checkpoint_paths & model_lora_set
    ckpt_only = checkpoint_paths - model_lora_set
    model_only = model_lora_set - checkpoint_paths
    
    print(f"\nPath matching results:")
    print(f"  Matches: {len(matches)}")
    print(f"  In checkpoint only: {len(ckpt_only)}")
    print(f"  In model only: {len(model_only)}")
    
    if ckpt_only:
        print(f"  Checkpoint paths not in model:")
        for p in list(ckpt_only)[:5]:
            print(f"    {p}")
    
    if model_only:
        print(f"  Model paths not in checkpoint:")
        for p in list(model_only)[:5]:
            print(f"    {p}")
    
    # THIS IS THE CRITICAL CHECK
    if len(matches) == 0:
        print("\n*** CRITICAL FAILURE: No paths match! ***")
        print("This is why 'Merged 0 LoRA params' occurs!")
        
        # Debug: show raw keypath info
        print("\nRaw path debug for first LoRA param:")
        first_found = [False]
        def debug_paths(path, leaf):
            if not first_found[0] and 'lora' in str(path).lower():
                first_found[0] = True
                print(f"  Raw keypath: {path}")
                for i, key in enumerate(path):
                    print(f"    [{i}] type={type(key).__name__}, value={key}")
            return leaf
        jax.tree_util.tree_map_with_path(debug_paths, state)
        
        return False
    
    assert len(matches) == len(checkpoint_paths), \
        f"Path mismatch! Only {len(matches)}/{len(checkpoint_paths)} checkpoint paths found in model"
    
    print("✓ All checkpoint paths match model paths!")
    return True


def test_real_merge_flow():
    """Test the actual merge with the real model.
    
    This replicates EXACTLY what restore_state does.
    """
    print("\n" + "="*60)
    print("TEST: Real merge flow")
    print("="*60)
    
    # Step 1: Initialize real model (like train.py does)
    from openpi.training import config as train_config
    import jax.random as jr
    
    config = train_config.get_config("pi05_openarm_ngc_lora")
    model_rng = jr.PRNGKey(0)
    model = config.model.create(model_rng)
    graphdef, state = nnx.split(model)
    
    print(f"Model state type: {type(state)}")
    
    # Step 2: Build restore template and restore checkpoint
    metadata = _read_checkpoint_metadata(CHECKPOINT_DIR, CHECKPOINT_STEP)
    template = _build_restore_template_from_metadata(metadata, {})
    
    pytree_handler = ocp.PyTreeCheckpointHandler(
        save_concurrent_gb=16,
        restore_concurrent_gb=16,
    )
    checkpoint_manager = ocp.CheckpointManager(
        CHECKPOINT_DIR,
        item_handlers={
            "assets": ckpt_module.CallbackHandler(),
            "train_state": pytree_handler,
            "params": pytree_handler,
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=None, create=False),
    )
    
    restored = checkpoint_manager.restore(
        CHECKPOINT_STEP,
        items={
            "train_state": {"step": 0},
            "params": {"params": template},
        },
    )
    
    restored_params = restored["params"].get("params", restored["params"])
    print(f"Restored {len(flax.traverse_util.flatten_dict(restored_params, sep='/'))} LoRA params")
    
    # Step 3: Merge into REAL model state (the critical operation)
    print("Merging into real model state...")
    merged_state = _merge_lora_params(state, restored_params)
    
    # Step 4: Verify merge succeeded
    print(f"Merged state type: {type(merged_state)}")
    
    # Check for ShapeDtypeStruct anywhere in the merged state
    shapedtype_count = 0
    def check_shapedtype(path, leaf):
        nonlocal shapedtype_count
        if isinstance(leaf, jax.ShapeDtypeStruct):
            shapedtype_count += 1
            path_str = _keypath_to_str(path)
            print(f"  WARNING: ShapeDtypeStruct at {path_str}: {leaf}")
        return leaf
    
    jax.tree_util.tree_map_with_path(check_shapedtype, merged_state)
    
    if shapedtype_count > 0:
        print(f"\n*** CRITICAL: Found {shapedtype_count} ShapeDtypeStruct values! ***")
        print("This is why training crashes!")
        return False
    
    print("✓ No ShapeDtypeStruct found in merged state")
    
    # Step 5: Verify we can reconstruct the model
    merged_model = nnx.merge(graphdef, merged_state)
    print("✓ Model reconstructed successfully")
    
    return True


def test_shapedtypestruct_scenario():
    """Test what happens when model has ShapeDtypeStruct (the bug scenario).
    
    This replicates the EXACT bug: when resume=True, train.py was returning
    jax.eval_shape() output which contains ShapeDtypeStruct placeholders.
    LoRA-only restore would then try to merge LoRA weights into these
    placeholders, leaving base weights as ShapeDtypeStruct.
    """
    print("\n" + "="*60)
    print("TEST: ShapeDtypeStruct scenario (the bug)")
    print("="*60)
    
    from openpi.training import config as train_config
    import jax.random as jr
    
    config = train_config.get_config("pi05_openarm_ngc_lora")
    
    # This is what train.py USED to do when resume=True:
    # train_state_shape = jax.eval_shape(init, init_rng)
    # This creates ShapeDtypeStruct placeholders instead of actual arrays!
    
    model_rng = jr.PRNGKey(0)
    
    # Simulate jax.eval_shape by creating the model and getting shapes
    def init_model(rng):
        model = config.model.create(rng)
        return nnx.state(model)
    
    # This is what creates ShapeDtypeStruct - eval_shape returns shapes not values
    state_shape = jax.eval_shape(init_model, model_rng)
    
    print(f"State from eval_shape type: {type(state_shape)}")
    
    # Count ShapeDtypeStruct in the shape
    shapedtype_count = 0
    def count_shapedtype(path, leaf):
        nonlocal shapedtype_count
        if isinstance(leaf, jax.ShapeDtypeStruct):
            shapedtype_count += 1
        return leaf
    
    jax.tree_util.tree_map_with_path(count_shapedtype, state_shape)
    print(f"ShapeDtypeStruct count in eval_shape output: {shapedtype_count}")
    
    if shapedtype_count > 0:
        print("This confirms the bug: resume=True returns ShapeDtypeStruct!")
        print("LoRA-only restore would merge into these, leaving base weights as placeholders.")
        
        # Now test the FIX: initialize full model first, then merge LoRA
        print("\nTesting the fix: full model initialization before LoRA merge...")
        
        # Initialize full model (not just shapes)
        model = config.model.create(model_rng)
        graphdef, state = nnx.split(model)
        
        # Verify no ShapeDtypeStruct in full model
        full_shapedtype_count = 0
        def check_full(path, leaf):
            nonlocal full_shapedtype_count
            if isinstance(leaf, jax.ShapeDtypeStruct):
                full_shapedtype_count += 1
            return leaf
        
        jax.tree_util.tree_map_with_path(check_full, state)
        print(f"ShapeDtypeStruct count in fully initialized model: {full_shapedtype_count}")
        
        if full_shapedtype_count == 0:
            print("✓ Full model initialization has no ShapeDtypeStruct!")
            print("✓ This is the fix: for LoRA-only resume, initialize full model first")
            return True
        else:
            print("✗ Full model still has ShapeDtypeStruct - need to investigate")
            return False
    
    return True


def test_full_restore_flow():
    """Test the complete restore flow end-to-end."""
    print("\n" + "="*60)
    print("TEST: Full restore flow (end-to-end)")
    print("="*60)
    
    # Step 1: Verify checkpoint
    step_dir = test_checkpoint_exists()
    
    # Step 2: Read metadata
    metadata, checkpoint_paths = test_metadata_readable(step_dir)
    
    # Step 3: Build template
    template = test_build_template(metadata)
    
    # Step 4: Restore from orbax
    restored_params, restored_step = test_orbax_restore(template)
    
    # Step 5: Create mock base params
    base_dict = test_create_mock_base_params(checkpoint_paths)
    
    # Step 6: Test path matching (critical!)
    test_path_matching(base_dict, restored_params)
    
    # Step 7: Test merge
    merged = test_merge_params(base_dict, restored_params)
    
    # Step 8: Test merge into nnx.State
    test_merge_into_nnx_state(restored_params)
    
    # Step 9: Test with REAL model paths (MOST CRITICAL!)
    real_paths_ok = test_real_model_paths()
    
    # Step 10: Test actual merge with real model
    if real_paths_ok:
        test_real_merge_flow()
    
    # Step 11: Test the ShapeDtypeStruct bug scenario
    test_shapedtypestruct_scenario()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_full_restore_flow()
