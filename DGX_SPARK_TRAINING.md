# Training π₀.₅ on DGX Spark

This guide covers how to fine-tune the π₀.₅ (Pi-0.5) Vision-Language-Action model on NVIDIA DGX Spark with a custom robot dataset.

## Hardware Context

DGX Spark has unique characteristics that require special configuration:
- **Grace CPU (ARM64/aarch64)** - Not x86_64, so some wheels need to be built from source
- **Blackwell GB10 GPU** - Unified memory architecture (CPU/GPU share 128GB)
- **Unified Memory** - CPU and GPU share the same 128GB memory pool, which creates challenges for checkpoint saving

### Understanding Unified Memory

Unlike traditional systems where GPU VRAM and CPU RAM are separate:

```
Traditional System (e.g., 3x4090):
┌─────────────────────┐  ┌─────────────────────┐
│  GPU VRAM (72GB)    │  │  System RAM (256GB) │
│  Model + training   │  │  Checkpoint saves   │
│  No conflict        │  │  No conflict        │
└─────────────────────┘  └─────────────────────┘

DGX Spark (Unified Memory):
┌─────────────────────────────────────┐
│   128GB Shared Pool                 │
│   ├── GPU uses: ~100-105GB          │
│   ├── OS/System: ~5GB               │
│   └── Checkpoint save: needs ~20GB  │  ← CONFLICT!
└─────────────────────────────────────┘
```

This means checkpoint saves compete with GPU memory, requiring careful tuning.

## Prerequisites

1. **Docker with NVIDIA Container Toolkit** installed
2. **Dataset in LeRobot format** at `~/datasets/your_dataset_lerobot`
3. **Computed normalization statistics** for your dataset

## Quick Start

### 1. Build the NGC Container

```bash
cd /home/evaughan/sparkpack/openpi
docker compose -f scripts/docker/compose_ngc.yml build openpi_server_ngc
```

This uses NVIDIA's optimized JAX container (NGC JAX 25.04) which has proper ARM64 and Blackwell support.

### 2. Create JAX Cache Directory

```bash
mkdir -p .jax_cache && chmod 777 .jax_cache
```

### 3. Run Training with LoRA

**Important**: Full fine-tuning requires too much memory for DGX Spark. Use LoRA (Low-Rank Adaptation) instead:

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm openpi_server_ngc \
    python scripts/train.py pi05_openarm_ngc_lora \
    --exp-name my_experiment \
    --no-wandb-enabled \
    --overwrite
```

### 4. Monitor Training

Training saves checkpoints every 1000 steps to:
```
checkpoints/pi05_openarm_ngc_lora/<exp_name>/<step>/
```

Monitor memory during training:
```bash
watch -n 1 'nvidia-smi; free -h'
```

## Key Configuration Details

### Memory Settings (compose_ngc.yml)

```yaml
environment:
  # CRITICAL: Lower fraction to leave headroom for checkpoint saves
  # On unified memory, we need ~25-30GB free for serialization
  - XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
  - XLA_PYTHON_CLIENT_PREALLOCATE=true
  # Cache compiled XLA kernels to speed up subsequent runs
  - JAX_COMPILATION_CACHE_DIR=/app/.jax_cache
```

#### Understanding XLA_PYTHON_CLIENT_MEM_FRACTION

This controls how much memory JAX **pre-allocates** at startup:
- **0.9** = Pre-allocate ~115GB → Only ~13GB headroom → **OOM on checkpoint save**
- **0.7** = Pre-allocate ~90GB → ~38GB headroom → **Works for checkpoints**

Note: JAX can allocate beyond the pre-allocated amount if needed. During training:
- Initial XLA compilation spikes to ~114GB
- Settles to ~100-105GB during training steps
- Checkpoint saves need additional headroom (~20GB)

### LoRA Config (config.py)

The working LoRA configuration uses:
- `paligemma_variant="gemma_2b_lora"` - LoRA adapters on the vision-language model
- `action_expert_variant="gemma_300m_lora"` - LoRA adapters on the action expert
- `freeze_filter` - Freezes all parameters except LoRA adapters
- `ema_decay=None` - Disables EMA to save memory
- `batch_size=2` - Minimal batch size for memory headroom

### Why LoRA?

| Approach | Trainable Params | Memory Usage | Batch Size |
|----------|------------------|--------------|------------|
| Full Fine-tune | ~3B | ~120GB+ | OOM |
| LoRA | ~20-50M | ~100-105GB | 2 |

LoRA trains only small adapter layers (~1-2% of parameters) while keeping the rest frozen.

### LoRA-Only Checkpoint Saving

The codebase is configured to save only LoRA adapter weights during checkpoints:
- Full checkpoint: ~12GB (entire model)
- LoRA-only checkpoint: ~100MB (just adapters)

This dramatically reduces memory spikes during checkpoint saves.

## Training Time Estimates

With LoRA on DGX Spark (batch_size=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.7):
- **~1 second per step** (after initial compilation)
- 10k steps: **~2.8 hours**
- 20k steps: **~5.5 hours**
- 30k steps: **~8.3 hours**

First run includes ~15-20 minutes of XLA compilation before training starts.

## Troubleshooting

### OOM During Checkpoint Save (Most Common Issue)

If the system freezes/crashes when saving checkpoints:

1. **Lower `XLA_PYTHON_CLIENT_MEM_FRACTION`** to 0.7 or lower in `compose_ngc.yml`
2. **Ensure batch_size=2** in your config
3. **Ensure LoRA-only saving is enabled** in `src/openpi/training/checkpoints.py`
4. **Check memory before save**: Should have 20-30GB free headroom

To verify current memory usage during training:
```bash
nvidia-smi
```

### OOM During Training Steps

If you see `RESOURCE_EXHAUSTED: Out of memory` during training (not saving):
1. Reduce `batch_size` to 2 in the config
2. Lower `XLA_PYTHON_CLIENT_MEM_FRACTION` to 0.6
3. Ensure you're using LoRA, not full fine-tuning

### Slow First Run (~15-20 minutes to start training)

This is normal. The first run:
1. Downloads base model weights (~12GB, cached after first run)
2. Compiles XLA kernels for Blackwell GPU

The XLA compilation cache (`JAX_COMPILATION_CACHE_DIR`) should speed up subsequent runs, though cache population can be inconsistent.

### Memory Spike During Compilation

During XLA compilation, memory usage spikes (observed: 114GB), then settles to ~100-105GB during training. This is why we need the 0.7 fraction - to leave room for these spikes.

### `torch.stack()` Column Errors

The NGC container has a newer `datasets` library that returns `Column` objects. The Dockerfile patches lerobot to handle this. If you see these errors, rebuild the container.

### bf16/f16 Conversion Errors

Blackwell GPUs may have issues with certain bf16 operations in older JAX versions. The NGC container (JAX 25.04) has fixes for this.

### Known Issue: Memory Leak During Saves

There's a known issue in OpenPI where checkpoint saves can leak memory over time (see [GitHub Issue #721](https://github.com/Physical-Intelligence/openpi/issues/721)). The LoRA-only saving and cache clearing mitigations help reduce this.

## Dataset Format

Your dataset should be in LeRobot format with:
- `observation.state` - Robot joint positions (16 DOF for bimanual: 7+1 per arm)
- `observation.images.cam_high` - Overhead camera
- `observation.images.cam_left_wrist` - Left wrist camera  
- `observation.images.cam_right_wrist` - Right wrist camera
- `action` - Target joint positions (16 DOF)
- Task annotations for prompts

### Normalization Stats

Compute norm stats before training:
```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm openpi_server_ngc \
    python scripts/compute_norm_stats.py pi05_openarm_ngc_lora
```

Stats are saved to `assets/<config_name>/<asset_id>/norm_stats.json`.

## Serving the Trained Model

After training, serve the policy:

```bash
docker compose -f scripts/docker/compose_ngc.yml run --rm -p 8000:8000 openpi_server_ngc \
    python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_openarm_ngc_lora \
    --policy.dir=checkpoints/pi05_openarm_ngc_lora/<exp_name>/<step>
```

Then connect with the OpenPI client from your robot or simulator.

## Comparison: DGX Spark vs Multi-GPU Systems

| Aspect | DGX Spark (1×GB10) | 3×RTX 4090 |
|--------|-------------------|------------|
| GPU Memory | 128GB unified | 72GB dedicated VRAM |
| System RAM | (same pool) | 256GB separate |
| Checkpoint Saves | Competes with GPU ⚠️ | Uses separate RAM ✓ |
| Training Speed | ~1 step/sec | ~3 steps/sec (parallel) |
| Batch Size | 2 (memory limited) | 8-16 (per GPU) |
| Setup Complexity | Single device | Multi-GPU coordination |

For production training, a system with separate CPU/GPU memory (like 3×4090 with 256GB RAM) avoids the unified memory challenges entirely.

## Files Modified for DGX Spark

1. `scripts/docker/serve_policy_ngc.Dockerfile` - NGC-based container with dependency fixes
2. `scripts/docker/compose_ngc.yml` - Memory settings (0.7 fraction) and cache config
3. `src/openpi/training/config.py` - LoRA config with batch_size=2
4. `src/openpi/training/checkpoints.py` - LoRA-only saving, cache clearing before save
5. `src/openpi/training/utils.py` - Disabled typecheck on TrainState (jaxtyping compatibility)

## References

- [OpenPI Repository](https://github.com/Physical-Intelligence/openpi)
- [π₀ Paper](https://www.physicalintelligence.company/blog/pi0)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx/)
- [JAX Memory Allocation](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)
