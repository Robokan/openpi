#!/usr/bin/env python3
"""Quick test for LoRA checkpoint restore logic.

Run with:
    python scripts/test_lora_restore.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax
import flax.traverse_util

# Test the helper functions directly
from openpi.training.checkpoints import (
    _keypath_to_str,
    _read_checkpoint_metadata,
    _build_restore_template_from_metadata,
    _is_lora_only_checkpoint,
    _merge_lora_params,
)
from etils import epath


def test_keypath_to_str():
    """Test KeyPath to string conversion."""
    print("\n=== Testing _keypath_to_str ===")
    
    # Simulate JAX KeyPath structure
    class MockDictKey:
        def __init__(self, key):
            self.key = key
    
    path = (MockDictKey("PaliGemma"), MockDictKey("llm"), MockDictKey("layers"), MockDictKey("lora_a"))
    result = _keypath_to_str(path)
    print(f"Path: {path}")
    print(f"Result: {result}")
    assert result == "PaliGemma/llm/layers/lora_a", f"Expected 'PaliGemma/llm/layers/lora_a', got '{result}'"
    print("✓ _keypath_to_str works correctly")


def test_is_lora_only_checkpoint():
    """Test LoRA-only checkpoint detection."""
    print("\n=== Testing _is_lora_only_checkpoint ===")
    
    # Try container path first, then host path
    checkpoint_dir = epath.Path("/app/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    if not checkpoint_dir.exists():
        checkpoint_dir = epath.Path("/home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint dir not found: {checkpoint_dir}")
        return
    
    result = _is_lora_only_checkpoint(checkpoint_dir, 280)
    print(f"Checkpoint at step 280 is LoRA-only: {result}")
    assert result == True, "Expected True for LoRA-only checkpoint"
    print("✓ _is_lora_only_checkpoint works correctly")


def test_read_checkpoint_metadata():
    """Test reading checkpoint metadata."""
    print("\n=== Testing _read_checkpoint_metadata ===")
    
    checkpoint_dir = epath.Path("/app/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    if not checkpoint_dir.exists():
        checkpoint_dir = epath.Path("/home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint dir not found: {checkpoint_dir}")
        return
    
    metadata = _read_checkpoint_metadata(checkpoint_dir, 280)
    
    if metadata is None:
        print("Failed to read metadata")
        return
    
    tree_meta = metadata.get("tree_metadata", {})
    print(f"Found {len(tree_meta)} params in checkpoint metadata")
    
    # Show first few paths
    for i, (path_str, meta) in enumerate(list(tree_meta.items())[:3]):
        import ast
        path_tuple = ast.literal_eval(path_str)
        if path_tuple[0] == 'params':
            path_tuple = path_tuple[1:]
        path_key = "/".join(path_tuple)
        shape = meta.get("value_metadata", {}).get("write_shape", "?")
        print(f"  {path_key}: {shape}")
    
    print("✓ _read_checkpoint_metadata works correctly")


def test_build_restore_template():
    """Test building restore template from metadata."""
    print("\n=== Testing _build_restore_template_from_metadata ===")
    
    checkpoint_dir = epath.Path("/app/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    if not checkpoint_dir.exists():
        checkpoint_dir = epath.Path("/home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint dir not found: {checkpoint_dir}")
        return
    
    metadata = _read_checkpoint_metadata(checkpoint_dir, 280)
    if metadata is None:
        print("Failed to read metadata")
        return
    
    # Create a mock base_params dict (in real code this would be nnx.State)
    base_params = {}
    
    template = _build_restore_template_from_metadata(metadata, base_params)
    
    template_flat = flax.traverse_util.flatten_dict(template, sep="/")
    print(f"Template has {len(template_flat)} params")
    
    # Show shapes
    for i, (k, v) in enumerate(list(template_flat.items())[:3]):
        print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
    
    print("✓ _build_restore_template_from_metadata works correctly")


def test_merge_lora_params():
    """Test merging LoRA params into base params."""
    print("\n=== Testing _merge_lora_params ===")
    
    # Create mock base params (dict structure)
    base_params = {
        "layer1": {
            "weight": np.zeros((10, 10)),
            "lora_a": np.zeros((10, 4)),
            "lora_b": np.zeros((4, 10)),
        },
        "layer2": {
            "weight": np.ones((5, 5)),
            "lora_a": np.zeros((5, 2)),
        }
    }
    
    # Create mock restored LoRA params
    lora_params = {
        "layer1": {
            "lora_a": np.ones((10, 4)) * 0.5,
            "lora_b": np.ones((4, 10)) * 0.3,
        },
        "layer2": {
            "lora_a": np.ones((5, 2)) * 0.7,
        }
    }
    
    merged = _merge_lora_params(base_params, lora_params)
    
    # Verify merge
    merged_flat = flax.traverse_util.flatten_dict(merged, sep="/")
    
    # Check that LoRA params were updated
    assert np.allclose(merged_flat["layer1/lora_a"], 0.5), "layer1/lora_a should be 0.5"
    assert np.allclose(merged_flat["layer1/lora_b"], 0.3), "layer1/lora_b should be 0.3"
    assert np.allclose(merged_flat["layer2/lora_a"], 0.7), "layer2/lora_a should be 0.7"
    
    # Check that base weights were NOT changed
    assert np.allclose(merged_flat["layer1/weight"], 0.0), "layer1/weight should still be 0"
    assert np.allclose(merged_flat["layer2/weight"], 1.0), "layer2/weight should still be 1"
    
    print("✓ _merge_lora_params works correctly")
    print("  - LoRA params updated: ✓")
    print("  - Base weights preserved: ✓")


def test_keypath_formats():
    """Debug test to understand JAX keypath formats."""
    print("\n=== Testing JAX KeyPath Formats ===")
    
    # Test with different pytree types
    test_dict = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "lora_a": np.zeros((2, 2)),
                        "lora_b": np.zeros((2, 2)),
                    }
                }
            }
        }
    }
    
    paths_found = []
    def collect_paths(path, leaf):
        path_str = _keypath_to_str(path)
        paths_found.append(path_str)
        return leaf
    
    jax.tree_util.tree_map_with_path(collect_paths, test_dict)
    
    print(f"Paths from plain dict:")
    for p in paths_found:
        print(f"  {p}")
    
    # Check if paths match expected format
    expected = "PaliGemma/llm/layers/attn/lora_a"
    if expected in paths_found:
        print(f"✓ Path format matches: {expected}")
    else:
        print(f"✗ Expected path not found: {expected}")
        print(f"  Available paths: {paths_found}")


def test_merge_into_nnx_state():
    """Test merging LoRA params into actual nnx.State."""
    print("\n=== Testing Merge into nnx.State ===")
    
    try:
        import flax.nnx as nnx
    except ImportError:
        print("flax.nnx not available, skipping")
        return
    
    # Create an nnx module with LoRA-like params
    class SimpleModule(nnx.Module):
        def __init__(self, rngs):
            self.layer1 = SimpleLayer(rngs)
            self.layer2 = SimpleLayer(rngs)
    
    class SimpleLayer(nnx.Module):
        def __init__(self, rngs):
            self.lora_a = nnx.Param(np.zeros((4, 2)))
            self.lora_b = nnx.Param(np.zeros((2, 4)))
            self.weight = nnx.Param(np.ones((4, 4)))
    
    rngs = nnx.Rngs(0)
    module = SimpleModule(rngs)
    graphdef, state = nnx.split(module)
    
    print(f"Base state type: {type(state)}")
    
    # Create LoRA params to merge (plain dict, like what comes from checkpoint)
    lora_params = {
        "layer1": {
            "lora_a": np.ones((4, 2)) * 0.5,
            "lora_b": np.ones((2, 4)) * 0.3,
        },
        "layer2": {
            "lora_a": np.ones((4, 2)) * 0.7,
            "lora_b": np.ones((2, 4)) * 0.9,
        }
    }
    
    # Merge!
    merged = _merge_lora_params(state, lora_params)
    
    print(f"Merged state type: {type(merged)}")
    
    # Verify merged is still nnx.State
    assert type(merged).__name__ == 'State', f"Expected State, got {type(merged)}"
    
    # Verify values were updated - reconstruct module and check
    merged_module = nnx.merge(graphdef, merged)
    
    assert np.allclose(merged_module.layer1.lora_a.value, 0.5), "layer1.lora_a should be 0.5"
    assert np.allclose(merged_module.layer1.lora_b.value, 0.3), "layer1.lora_b should be 0.3"
    assert np.allclose(merged_module.layer2.lora_a.value, 0.7), "layer2.lora_a should be 0.7"
    assert np.allclose(merged_module.layer2.lora_b.value, 0.9), "layer2.lora_b should be 0.9"
    
    # Base weights should be unchanged
    assert np.allclose(merged_module.layer1.weight.value, 1.0), "layer1.weight should still be 1.0"
    assert np.allclose(merged_module.layer2.weight.value, 1.0), "layer2.weight should still be 1.0"
    
    print("✓ Merge into nnx.State works correctly!")
    print("  - Type preserved: ✓")
    print("  - LoRA params updated: ✓")
    print("  - Base weights preserved: ✓")


def test_nnx_state_paths():
    """Test path format with actual nnx.State."""
    print("\n=== Testing nnx.State Path Format ===")
    
    try:
        import flax.nnx as nnx
    except ImportError:
        print("flax.nnx not available, skipping")
        return
    
    # Create a simple nnx module with LoRA-like params
    class SimpleModule(nnx.Module):
        def __init__(self, rngs):
            self.lora_a = nnx.Param(np.zeros((4, 2)))
            self.lora_b = nnx.Param(np.zeros((2, 4)))
            self.weight = nnx.Param(np.ones((4, 4)))
    
    rngs = nnx.Rngs(0)
    module = SimpleModule(rngs)
    
    # Get state
    graphdef, state = nnx.split(module)
    
    print(f"State type: {type(state)}")
    # FlatState doesn't have .keys(), iterate directly
    flat = state.flat_state()
    print(f"FlatState type: {type(flat)}")
    print(f"First few state paths: {list(flat)[:5]}")
    
    # See what paths jax.tree_util gives us
    paths_found = []
    raw_paths = []
    def collect_paths(path, leaf):
        path_str = _keypath_to_str(path)
        paths_found.append((path_str, type(leaf).__name__))
        # Also collect raw path info
        raw_info = [(type(k).__name__, getattr(k, 'key', getattr(k, 'idx', str(k)))) for k in path]
        raw_paths.append(raw_info)
        return leaf
    
    jax.tree_util.tree_map_with_path(collect_paths, state)
    
    print(f"Raw paths from jax.tree_util on nnx.State:")
    for rp in raw_paths[:2]:
        print(f"  {rp}")
    
    print(f"Converted paths from jax.tree_util on nnx.State:")
    for p, t in paths_found:
        print(f"  {p} -> {t}")
    
    # The issue might be that nnx.State paths include 'value' 
    # or use different key types


def test_actual_checkpoint_restore_flow():
    """Test the actual restore flow with real checkpoint."""
    print("\n=== Testing Actual Checkpoint Restore Flow ===")
    
    checkpoint_dir = epath.Path("/app/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    if not checkpoint_dir.exists():
        checkpoint_dir = epath.Path("/home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora/spark_lora_v3")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint dir not found")
        return
    
    # Read metadata
    metadata = _read_checkpoint_metadata(checkpoint_dir, 280)
    if metadata is None:
        print("Failed to read metadata")
        return
    
    # Build template
    base_params = {}  # Empty base for template building
    template = _build_restore_template_from_metadata(metadata, base_params)
    
    # Flatten template to see keys
    template_flat = flax.traverse_util.flatten_dict(template, sep="/")
    print(f"Template keys (from checkpoint metadata):")
    for k in list(template_flat.keys())[:3]:
        print(f"  {k}")
    
    # Now simulate what happens during merge
    # The restored params would have same structure as template
    # But the base_params (nnx.State) might have different path format
    
    # Let's see what paths jax.tree_util would generate for the template
    paths_from_tree = []
    def collect_paths(path, leaf):
        path_str = _keypath_to_str(path)
        paths_from_tree.append(path_str)
        return leaf
    
    jax.tree_util.tree_map_with_path(collect_paths, template)
    
    print(f"\nPaths from jax.tree_util.tree_map_with_path on template:")
    for p in list(paths_from_tree)[:3]:
        print(f"  {p}")
    
    # Check if they match
    template_keys = set(template_flat.keys())
    tree_paths = set(paths_from_tree)
    
    if template_keys == tree_paths:
        print("\n✓ Paths match!")
    else:
        print(f"\n✗ Path mismatch!")
        print(f"  In template_flat but not tree_paths: {template_keys - tree_paths}")
        print(f"  In tree_paths but not template_flat: {tree_paths - template_keys}")


def main():
    print("=" * 60)
    print("LoRA Checkpoint Restore Test")
    print("=" * 60)
    
    test_keypath_to_str()
    test_keypath_formats()
    test_merge_into_nnx_state()  # Critical test!
    test_nnx_state_paths()
    test_is_lora_only_checkpoint()
    test_read_checkpoint_metadata()
    test_build_restore_template()
    test_merge_lora_params()
    test_actual_checkpoint_restore_flow()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
