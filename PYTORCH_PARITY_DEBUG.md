# π0.5 PyTorch vs JAX Parity Debug Log

**Goal:** Get the PyTorch policy server (port 8002) to produce actions
indistinguishable from the JAX policy server (port 8001) for the
`pi05_openarm_ngc_lora_v4 / chocolate_bars_pi05` checkpoint, so we can
deploy PT on DGX Spark / AGX Thor with FP4/FP8 quantization.

**Symptom:** Robot drifts arms upward when using PT server. JAX server works
correctly. Single-step model parity is excellent (`cos=0.995`) but
end-to-end actions after 10 diffusion steps + Unnormalize show consistent
**+0.135 rad bias on joint 3 (shoulder pitch)**.

---

## ★ ★ ★ 2026-05-19 RESOLVED: Bug #2 fixed via runtime LoRA ★ ★ ★

After ruling out bf16 LoRA-intermediate rounding as a sufficient
explanation (it gives perpendicular noise, not magnitude bias), we
implemented **runtime LoRA in PyTorch** — never pre-merge LoRA into
base weights; instead apply `base_out + scaling * (x @ la) @ lb` on
every forward pass, matching JAX's two-matmul order exactly.

### Files added/modified
- NEW: `src/openpi/models_pytorch/lora_runtime.py` — `install_runtime_lora()` patches each q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj's forward to add LoRA via einsum.
- `examples/convert_jax_model_to_pytorch.py` — `OPENPI_PT_RUNTIME_LORA=1` skips merging, emits `lora.safetensors` alongside `model.safetensors` with PT-named tensors.
- `src/openpi/models/model.py:load_pytorch` — auto-loads `lora.safetensors` if present, calls `install_runtime_lora()`.

### Results (10-step diffusion, `pi05_openarm_ngc_lora_v4`)

| Variant | raw cos | raw ratio | post-unnorm cos | post-unnorm ratio |
|---|---|---|---|---|
| Pre-merge bf16 (broken) | 0.996 | **0.918** | — | **0.918** (8% bias) |
| Pre-merge fp32 (FIXED ckpt) | 0.996 | **0.918** | — | **0.918** |
| Runtime LoRA, bf16 inference | +0.999992 | 0.9995 | +0.999972 | 0.9928 (0.7%) |
| **Runtime LoRA, fp32 inference** | **+0.999997** | **1.0001** | **+0.999988** | **0.9974 (0.26%)** |

The 8% magnitude bias dropped to **0.26%** post-unnormalization in fp32
mode (well within bf16 quantization noise from JAX-side). The remaining
residual is fundamentally bf16 quantization in JAX's 10-step Euler loop
and cannot be reduced further without running JAX in fp32 too.

### Deployed
- `/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch` now points at the runtime-LoRA checkpoint (old one renamed to `..._PRE_RUNTIME_LORA`).
- Server restarted with `OPENPI_PYTORCH_PRECISION=float32` (fp32 inference, even tighter parity).
- 252 LoRA modules patched (18 layers × 7 modules × 2 experts).

---

## ★ KEY 2026-05-18 FINDINGS

### Confirmed: bug is LoRA-related, not base model

`scripts/diag_no_lora.py` runs JAX with `lora_a/lora_b` zeroed AND PT
loaded from a no-LoRA checkpoint (converted with `OPENPI_CONV_LORA_SCALING=0`).

| Setting | cos(JAX,PT) | ratio (raw) | post-unnorm ratio | joint 3 diff |
|---------|-------------|-------------|--------------------|---------------|
| **With LoRA** (default)            | 0.996 | 0.918 | 1.084 | **+0.135 rad** |
| **No LoRA** (zeroed both sides)    | 0.99996 | 0.999 | 1.0016 | **+0.003 rad** |

**Conclusion:** Base PaliGemma + base gemma_300m forward is perfectly
correct in PT. The 8% bias only appears once LoRA is applied. The bug
is in LoRA pathway, not base model code.

### Bug #1 (FIXED 2026-05-18): bf16 truncation during LoRA merge in converter

`examples/convert_jax_model_to_pytorch.py:convert_pi0_checkpoint` used to
instantiate `PI0Pytorch(model_config)` with `config.dtype="bfloat16"`,
which internally calls `to_bfloat16_for_selected_params("bfloat16")`
during PaliGemmaWithExpertModel init. The freshly-built model thus had
**bf16 parameters**.

Then `load_state_dict(all_params, strict=False)` (no `assign=True`)
loaded our **fp32 LoRA-merged weights** into those bf16 parameters,
silently casting them down. The per-element rounding step at bf16
(~0.4% of base magnitude) is the same order of magnitude as the LoRA
delta (~1-3% of base magnitude in norm, but per-element often smaller).
The bf16 truncation destroyed ~1% of the LoRA contribution per weight.

`scripts/diag_checkpoint_diff.py` shows the corruption:
- BEFORE fix: `cos(jax_delta, pt_diff)=0.99` (note: NOT 1.0) and `ratio=1.01`
  (the truncation is biased — the per-weight delta is systematically 1%
  larger in the stored PT checkpoint than the JAX-computed delta).
- AFTER fix: `cos=1.0000000` and `ratio=1.000000` for ALL LoRA pairs.

**Fix**: Instantiate the conversion-time model with `dtype='float32'`
(via `dataclasses.replace(model_config, dtype='float32')`), then standard
`load_state_dict(strict=False)` is fp32→fp32 = no-op for dtype, preserving
the LoRA precision. Tried `assign=True` first but that broke the HF
PaliGemma `lm_head ↔ embed_tokens` weight tying — produced garbage
inference. The fp32-init approach is clean.

This fix is REAL: per-weight `cos(jax_delta, pt_diff)` went 0.99 → 1.0,
and `ratio` went 1.01 → 1.00 across all 10×18 LoRA pairs.

### Bug #2 (STILL OPEN): 8% magnitude bias persists despite exact LoRA merge

Even with the FIXED checkpoint (LoRA delta now stored to fp32 exactly):
- End-to-end `cos(JAX, PT)` = 0.996 (slightly better than 0.962 before)
- End-to-end `ratio (raw)` = **0.918** (PT ~8% smaller in raw chunk norm)
- End-to-end `ratio (post-unnorm)` = **1.084** (PT ~8% larger in real space)

So Bug #1's fix removed the ~1% per-weight precision noise but the
fundamental **PT's LoRA contribution is ~80% of JAX's LoRA contribution**
remains.

  - JAX-with-LoRA action norm: 17.93
  - JAX-no-LoRA action norm:   13.54
  - "LoRA effect" (JAX):       ~11.7 (Pythagorean)
  - PT-with-LoRA action norm:  16.46
  - PT-no-LoRA action norm:    13.52
  - "LoRA effect" (PT):        ~9.4
  - PT effect / JAX effect:    ~80%

### ★ 2026-05-19 FINDINGS — bias localized to PaliGemma 18-layer forward

#### Layer-by-layer KV cache divergence

`scripts/diag_per_step_lora.py` compares JAX and PT KV cache values for
prefix tokens across paligemma layers. The bias accumulates monotonically:

| Layer | K cos | K ratio | V cos | V ratio |
|-------|-------|---------|-------|---------|
| 0     | 0.99996 | 1.0005 | 0.99996 | 1.0009 |
| 1     | 0.99991 | 0.9989 | 0.99978 | 0.9949 |
| 5     | 0.99945 | 0.9967 | 0.99822 | 0.9952 |
| 9     | 0.99806 | 1.0080 | 0.99252 | 1.0046 |
| 13    | 0.99375 | 1.0154 | 0.97535 | 1.0271 |
| 17    | 0.98362 | 1.0163 | 0.98408 | 1.0507 |

PT V at the final paligemma layer is **+5.0% LARGER** than JAX V. The
divergence is ~0.3% per layer, consistent with a small bias amplified
through 18 residual stacks.

`suffix_out` (final gemma_expert output BEFORE action_out_proj):
- cos = 0.9946, ratio = 0.9924 (PT 0.8% SMALLER)

`v_t` (after action_out_proj):
- cos = 0.9991, ratio = 0.9842 (PT 1.6% smaller per step → 8% per chunk)

#### Ruled out (with diagnostic scripts)

- **embed_prefix differences**: injecting JAX's prefix tokens into PT's
  `compute_prefix_kv_cache` produces THE SAME bias pattern. Confirmed
  the bug is in the 18-layer paligemma forward, not embedding.
- **Custom KV cache code**: fast path (compute_prefix_kv_cache +
  denoise_step_with_cache) vs slow joint-forward path (`OPENPI_PT_NO_KVCACHE=1`)
  give IDENTICAL results (cos=1.0). So the bug is not in our manual
  layer walking — it's in the layer modules themselves.
- **RoPE**: `scripts/diag_rope_parity.py` — direct JAX `_apply_rope` vs
  PT `apply_rotary_pos_emb` on identical Q, positions: cos=1.0000000,
  max|diff|=2e-6.
- **Single matmul (q_proj LoRA-merged vs runtime-applied)**: cos=1.0,
  ratio=1.0 in fp32 and ~1.0 in bf16. The merge math is exact.
- **Runtime weights**: `scripts/diag_runtime_weights.py` confirms PT
  loaded weights equal JAX `base + scaling*(la@lb)` to bf16 precision
  (cos=1.0, ratio=1.0) for all checked attention and MLP weights.
- **Precision (bf16 vs fp32)**: `scripts/diag_jax_fp32_vs_pt_fp32.py`
  forces PT to fp32 (`OPENPI_PYTORCH_PRECISION=float32`). The bias is
  IDENTICAL: cos=0.9957, ratio=0.9182. So precision is definitively
  ruled out.
- **action_out_proj**: the bias is already present in `suffix_out`
  BEFORE this projection (cos=0.9946, ratio=0.9924).

#### Reviewed code in `src/openpi/models_pytorch/`

Found no obvious algorithmic bug in:
- `pi0_pytorch.py` — embed_prefix, embed_suffix, compute_prefix_kv_cache,
  denoise_step_with_cache all match JAX semantics on paper
- `gemma_pytorch.py` — PaliGemmaWithExpertModel, fp32_attention all match
  JAX semantics on paper
- `preprocessing_pytorch.py` — image preprocessing same in inference mode
- `transformers_replace/models/gemma/modeling_gemma.py` — customized
  GemmaRMSNorm, GemmaDecoderLayer, apply_rotary_pos_emb match JAX exactly
- `transformers_replace/models/paligemma/modeling_paligemma.py` —
  multi_modal_projector is just a linear layer, matches JAX

#### Open hypothesis

The bias is purely structural in how the 18-layer PaliGemma forward
accumulates contributions. Even with **identical fp32 weights, identical
fp32 inputs, identical RoPE, identical attention**, PT produces a
progressively larger residual stream than JAX. The amplification is
roughly geometric (~0.3% per layer).

Most likely culprits remaining (in order):
1. The `_gated_residual(x, y, None) = x + y` for paligemma vs JAX's
   `x + 1.0 * y` — should be mathematically identical, but possible
   subtle PT broadcasting difference
2. Some module's `.float()` cast leading to subtle precision creep
3. A subtle bug in the **prefix attention mask shape/format** that
   causes PT to attend slightly differently than JAX

Next step suggestion: instrument JAX side to dump per-layer hidden
state output (need to bypass `nn.scan` somehow), then directly compare
PT layer 0 output to JAX layer 0 output for the same input. This would
localize whether the bias starts at layer 0 (and accumulates) or
emerges from a specific deeper layer.

Tried (no effect):
  - `OPENPI_PYTORCH_PRECISION=float32` (run inference in fp32 throughout)
  - `OPENPI_PT_MATMUL_PRECISION=highest` (no TF32)
  - `OPENPI_PT_FP32_ATTN=1` (fp32 attention accumulator) — already on
  - `OPENPI_PT_DISABLE_COMPILE=1` (eager mode)
  - `OPENPI_PT_ADARMS_BF16=1` (force bf16 adarms modulation, match JAX)

None of these change the 8% bias. So it's not precision-related.

Open hypotheses:
  - LoRA-affected weights act differently in attention vs my mental model
    (e.g., maybe the einsum equations include some `jnp.where`-based
    masking that interacts with LoRA non-linearly)
  - The FFN linear LoRA might be applied differently (look at
    `gating_einsum_lora` vs `linear_lora` tuple structure in lora.py)
  - There's another set of LoRA-affected weights I haven't accounted for
  - JAX's `nn.remat` / `nn.scan` may modify the LoRA application semantics

To investigate next:
  - Pull a single attention layer's bf16 Q/K/V output for SAME x in both
    JAX (with LoRA) and PT (with merged FIXED weight). If the layer
    outputs differ by 20%, the bug is in that layer's compute. If they
    match, the bug is downstream/cumulative.
  - Layer-by-layer comparison through the expert (or PaliGemma)
  - Check JAX's actual `Einsum.__call__` vs the lora.py code (maybe
    the runtime path differs from the `lora.py` definition).

---

## Setup

| Component | Value |
|-----------|-------|
| Robot | OpenArm 16-DOF dual-arm |
| Policy | `pi05_openarm_ngc_lora_v4` config |
| Checkpoint | `chocolate_bars_pi05` step 29999 (LoRA-trained) |
| JAX ckpt | `/app/checkpoints/.../chocolate_bars_pi05/29999/` |
| PT ckpt | `/app/checkpoints/.../chocolate_bars_pi05_pytorch/` (converted) |
| JAX server | `openpi-jax-server` container, port 8001 ✅ |
| PT server | `openpi-pt-server` container, port 8002 ❌ |
| Hardware | DGX Spark (NVIDIA GB10, ARM64) |
| Model arch | PaliGemma 2B + gemma_300m_lora expert (18 layers, AdaRMS) |
| Diffusion | 10 Euler steps, dt=-0.1, t=1.0→0.0 |
| Action | 50 horizon × 32 dim (16 dim used by robot, padded to 32) |

Run JAX server:
```bash
docker run -d --name openpi-jax-server --gpus all --network host \
  openpi-jax-server:latest \
  python scripts/serve_policy.py --port 8001 policy:checkpoint \
  --policy.config pi05_openarm_ngc_lora_v4 \
  --policy.dir checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999
```

Run PT server (currently `openpi-thor:latest` for CUDA):
```bash
docker run -d --name openpi-pt-server --gpus all --network host \
  -v /home/evaughan/sparkpack/openpi:/app \
  -v /home/evaughan/.cache/openpi:/openpi_assets \
  -v /home/evaughan/.cache/huggingface:/root/.cache/huggingface \
  -w /app \
  -e PYTHONPATH=/app/src:/app/packages/openpi-client/src \
  openpi-thor:latest \
  python scripts/serve_policy.py --port 8002 policy:checkpoint \
  --policy.config pi05_openarm_ngc_lora_v4 \
  --policy.dir checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch
```

---

## Bug summary (CURRENT)

Running `diag_full_loop.py` with **identical fixed noise (seed=42)** and
**identical real observation**:

| Layer | cos(JAX, PT) | norm(PT)/norm(JAX) | max\|diff\| |
|-------|------|--------|------|
| `embed_prefix` image tokens | 0.9999 | ~1.0 | small |
| `embed_prefix` lang tokens (after fix) | 1.0 | 1.0 | 0 |
| `action_in_proj` output | 0.999999 | 1.0000 | 0.01 |
| `adarms_cond` (time MLP) | 0.999998 | 1.0001 | 0.001 |
| paligemma after 18 layers (prefix) | 0.97 | ~1.0 | – |
| `suffix_out` after 18 expert layers | 0.991 | 1.001 | 0.75 |
| `v_t` single step (action_out_proj) | 0.995 | 0.993 | 0.80 |
| Final actions (10 Euler steps, raw) | 0.9958 | **0.917** (PT 8% smaller) | 0.16 |
| Post-Unnormalize actions | 0.963 | **1.087** (PT 8.7% LARGER in real space) | 0.18 |
| **Joint 3 (shoulder pitch) diff** | – | – | **+0.135 rad** |

The per-step v_t parity is excellent (cos=0.995, ratio=0.993). The 0.7%
per-step magnitude bias compounds over 10 diffusion steps to ~8%
chunk-level bias. After Unnormalize (which multiplies by std), high-std
joints get amplified disproportionately, and the sign flips because the
state values are positive (joint 3 state = 2.008 rad).

**Net effect:** PT consistently overshoots upward on shoulder pitch joints
(3 and 11) by ~0.13 rad per chunk, causing the arms to slowly drift up.

---

## CRITICAL: `transformers_replace/` requires manual `docker cp` to apply

The files in `src/openpi/models_pytorch/transformers_replace/` are NOT
automatically imported. They must be **manually copied** into the
container's transformers package:

```bash
docker cp src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py \
  openpi-pt-server:/usr/local/lib/python3.12/dist-packages/transformers/models/gemma/modeling_gemma.py
```

This was a debugging gotcha — edits to those files appeared to do nothing.

## Fixes ALREADY applied (do not redo)

These are committed; their effects on parity are below.

1. **LoRA merge fix** (commit pre-history)
   - `examples/convert_jax_model_to_pytorch.py`: `_merge_lora_into_base()`
     correctly handles both `attn/X/lora_a + .../w` and FFN-style
     `gating_einsum_lora_a + gating_einsum`. Without this the converted PT
     checkpoint was the un-fine-tuned base model.
   - Verified: cos=1.0 on every weight tensor (`diag_weight_match.py`).

2. **`load_pytorch` `strict=False, assign=True`**
   - `src/openpi/models/model.py`: handles bf16 safetensors loaded into
     fp32-initialized model parameters without dtype-mismatch errors.

3. **Re-tie `embed_tokens.weight = lm_head.weight`** AFTER load
   - `src/openpi/models/model.py`: `assign=True` breaks HuggingFace's
     constructor-time weight tying. The safetensors file only contains
     `lm_head.weight` (default safetensors behavior for tied weights). After
     `assign=True`, `embed_tokens.weight` was left at random init, making
     ALL language token embeddings random.
   - Effect: lang token cos went from **0.0016 → 1.0** at `embed_prefix`.
     Massive improvement: motion delta cos went from ~0 → 0.48-0.74.

4. **`fp32_attention` helper** for attention logits + softmax
   - `src/openpi/models_pytorch/gemma_pytorch.py`: matches JAX's
     `jnp.einsum(..., preferred_element_type=jnp.float32)` semantics.
   - Effect on PaliGemma prefix forward: lang token cos went from
     **0.65 → 0.97**, lang norm ratio 1.279 → 1.011.
   - **However**: did not fix end-to-end action parity (still 8% bias).

5. **`make_att_2d_masks` ordering matches JAX `make_attn_mask`**
   - Verified both produce identical masks on the same inputs.

6. **KV cache** ported from TurboPi
   - `src/openpi/models_pytorch/pi0_pytorch.py:compute_prefix_kv_cache` +
     `denoise_step_with_cache`. ~7× inference speedup vs joint forward.
   - Verified: joint-forward vs KV-cache give same result.

7. **`time_emb` precision**: kept as fp32 (JAX semantics) — `posemb_sincos`
   in JAX returns fp32.

8. **GPU container**: switched from CPU-only `openpi-jax-server` image
   to `openpi-thor:latest` with `PYTHONPATH=/app/src:...` override so the
   container picks up the live source from the mounted workspace.

9. **🆕 `rotary_emb.inv_freq` precision (MAY 2026)**
   - `src/openpi/models_pytorch/gemma_pytorch.py:to_bfloat16_for_selected_params`
   - The `self.to(dtype=torch.bfloat16)` call casts both PARAMETERS and
     BUFFERS. The `GemmaRotaryEmbedding.inv_freq` buffer is fp32 by
     default but got truncated to bf16, e.g. `inv_freq[1]` was
     **0.9296875** instead of **0.9305720**. This compounded through
     `cos/sin` at high positions, e.g. PT cos at pos 968 dim 1 was
     **+0.129** vs JAX's **-0.665** — completely wrong RoPE rotation.
   - Fix: walk submodules, re-run `rope_init_fn` to get fp32 inv_freq,
     and overwrite the buffer.
   - **Effect on actions**: ~1e-4 per-dimension change (i.e., very tiny;
     the bf16 rounding of cos/sin AFTER the matmul cancels most of the
     gain, and the model was likely trained to be robust to this).
   - **Still a real bug** — keep the fix in. But not the source of the
     8% chunk-level bias.

---

## Things ruled out (with evidence)

For each, the diagnostic that proved it's not the bug:

### Weights
- All weights match cos=1.0 between JAX and PT (`diag_weight_match.py`).
  Includes AdaRMS dense kernel/bias, action_in_proj, action_out_proj,
  time_mlp_in/out, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
  down_proj, input_layernorm, post_attention_layernorm, final_norm.

### Input pipeline
- Image preprocessing: cos=0.9999 on SigLIP output.
- Tokenization: identical tokens (verified character-by-character).
- `embed_prefix`: cos=1.0 on language tokens (after weight-tying fix),
  cos=0.9999 on image tokens.
- `_input_transform`: state values identical between JAX and PT through
  the entire input pipeline.

### Normalization stats
- `norm_stats.json` md5 matches between JAX and PT checkpoint dirs.
  Confirmed via `md5sum`.

### Output transforms
- `_output_transform` is the same code path for both JAX and PT
  (constructed from the same `data_config`).
- Unnormalize uses the same norm_stats.

### Precision
- bf16 vs fp32 throughout: gives **identical** end-to-end actions
  (tested with `OPENPI_PYTORCH_PRECISION=float32`). The 8% bias is
  preserved either way.
- TF32 vs strict fp32 matmul (`OPENPI_PT_MATMUL_PRECISION=highest`):
  gives **identical** actions to default `"high"`. TF32 is not the bug.
- `fp32_attention` enabled/disabled (`OPENPI_PT_FP32_ATTN`): improves
  per-layer prefix parity but does not change end-to-end action bias.
- **🆕 torch.compile vs eager** (`OPENPI_PT_DISABLE_COMPILE=1`): gives
  **identical** end-to-end actions. The earlier "compile gives cos=0.85
  but eager gives cos=-0.96" comment in code was about pi05_libero
  with ZERO inputs — does NOT apply to pi05_openarm with real obs.
- **🆕 AdaRMS modulation in bf16 vs fp32**: identical results. The
  AdaRMS dense weight is fp32 (kept by `params_to_keep_float32`), and
  forcing bf16 matmul there changes nothing.
- **🆕 RoPE precision fp32 vs bf16-truncated inv_freq**: changes
  per-element output by ~1e-4. Trivial relative to 8% chunk bias.
- **Conclusion**: The bug is NOT precision-related. NVIDIA reports FP4
  quantized pi05 works, which is consistent — if FP4 works, then
  bf16-vs-fp32 mismatches on the order of 0.4% per element cannot
  produce an 8% systematic chunk bias. The bug must be ALGORITHMIC.

### Diffusion loop
- JAX and PT loops are byte-equivalent:
  - dt = -1.0 / num_steps = -0.1
  - while time >= -dt/2: x_t = x_t + dt * v_t; time += dt
  - both start with x_t = noise, time = 1.0
- Initial noise: when `noise` arg is passed to both, they use the same
  values.
- num_steps defaults to 10 on both sides.

### KV cache vs joint forward
- `OPENPI_PT_NO_KVCACHE=1` (slow joint-forward path through
  PaliGemmaWithExpert.forward) produces same result as KV-cache path
  (cos=0.9999 between the two PT paths).

### Time embedding
- `posemb_sincos` (JAX) vs `create_sinusoidal_pos_embedding` (PT):
  produce identical `time_emb` vectors at fp32. Verified via direct
  comparison.

### AdaRMS / final_norm
- Dense kernel/bias for AdaRMS match cos=1.0.
- The PT path correctly applies `gemma_expert.norm(x, cond=adarms_cond)`
  after the 18 layers.

### Attention mask
- `make_att_2d_masks(pad_masks, att_masks)` (PT) and `make_attn_mask`
  (JAX) produce identical bool masks for the same inputs (verified by
  direct equality check on bool tensors).
- Suffix self-attention is full bidirectional (all True). Cross-attention
  to prefix masks padding correctly.

### Output dimensionality
- Both produce (1, 50, 32) action tensors. Robot uses only first 16 dims.
- Slicing [:, :, :16] vs full [:, :, :] shows same cos and ratio.

### gemma_expert lm_head
- `gemma_expert.model.embed_tokens` is set to `None` (expert doesn't
  embed tokens; gets pre-embedded suffix input).
- `gemma_expert.lm_head` is loaded but never called (we use
  `action_out_proj` instead).
- The "Could not re-tie gemma_expert embed_tokens to lm_head" warning
  during load is BENIGN — embed_tokens is intentionally None.

### Vision tower
- SigLIP vision tower: image token cos=0.9999 vs JAX. Not the bug.

---

## TurboPi comparison (`/home/evaughan/sparkpack/TurboPi/openpi`)

TurboPi achieves 100% LIBERO accuracy with their PT impl. Compared
their code to ours:

| Item | TurboPi | Ours |
|------|---------|------|
| Attention in `compute_prefix_kv_cache` | SDPA, bf16, with mask | fp32_attention with mask |
| Attention in `denoise_step_with_cache` | SDPA, bf16, **NO mask, NO scale** | fp32_attention with mask |
| LoRA merging in converter | **NONE** (libero ckpt has no LoRA) | Full LoRA merge |
| `time_emb` dtype | cast to model_dtype (bf16) | kept as fp32 |
| `embed_suffix` action cast | `noisy_actions.to(model_dtype)` | direct |
| Final norm with adarms | identical | identical |
| Per-layer adarms application | `cond=adarms_cond` | `cond=adarms_cond` |

**Key observations:**
- TurboPi's success on LIBERO does **not** prove their impl is correct —
  LIBERO might be tolerant to bf16 drift / padding-unmasked attention.
- TurboPi has NO LoRA handling. Their working result is for a
  no-LoRA checkpoint. Our bug could be specific to LoRA-merged inference.

---

## Diagnostic scripts (use these to test)

All in `scripts/`. Run inside `openpi-pt-server` container:
```bash
docker exec openpi-pt-server bash -lc \
  "cd /app && PYTHONPATH=/app/src:/app/packages/openpi-client/src \
   python scripts/<diag_name>.py"
```

| Script | Purpose | Current result |
|--------|---------|----------------|
| `compare_jax_pytorch.py` | Compare server outputs on real obs (random noise) | cos=0.04-0.7, ratio=1.5-2.6 (noisy) |
| `compare_servers_live.py` | Same but via running servers (HTTP) | varies |
| `diag_weight_match.py` | Per-tensor weight comparison | cos=1.0 everywhere ✅ |
| `diag_embed_prefix.py` | `embed_prefix` token-by-token | image cos=0.9999, lang cos=1.0 ✅ |
| `diag_per_layer.py` | Per-layer hidden state diff in PaliGemma prefix | layer 0–17 norms+cos |
| `diag_model_prefix.py` | Final prefix output via `compute_prefix_kv_cache` | cos=0.97 (lang fixed) |
| `diag_one_step.py` | Single denoise step v_t comparison | cos=0.995, ratio=0.99 |
| **`diag_suffix.py`** | Single denoise step: action_in_proj → suffix_out → v_t | **see breakdown above** |
| **`diag_full_loop.py`** | Full 10-step diffusion + post-Unnormalize, fixed noise | **cos=0.996 raw, 0.963 post-xfm, +0.135 on joint 3** |

---

## Env vars for debugging

| Env var | Default | Effect |
|---------|---------|--------|
| `OPENPI_PYTORCH_PRECISION` | `bfloat16` | `float32` runs everything in fp32 |
| `OPENPI_PT_NO_KVCACHE` | `0` | `1` uses slow joint-forward path |
| `OPENPI_PT_FP32_ATTN` | (on) | `0` would disable fp32 attention if we add the flag |
| `OPENPI_PT_MATMUL_PRECISION` | `high` | `highest` for strict fp32, `medium` for bf16 |

---

## Algorithmic checks done (verified identical between JAX and PT)

These were CONFIRMED to be structurally / mathematically the same:

- `embed_prefix`: image features cos=0.9999, lang `* sqrt(2048)` scaling
  applied in both JAX (`Embedder.encode`) and PT (`embed_prefix`).
- `embed_suffix`: for pi05, both:
  - action_tokens = action_in_proj(noisy_actions)
  - time_emb = posemb_sincos(t) ... -> time_mlp_in -> swish -> time_mlp_out -> swish
  - adarms_cond = time_emb
  - tokens = action_tokens (NO state token, NO time token in suffix)
  - ar_mask = [True] + [False] * (action_horizon - 1)
- `make_attn_mask`: `cumsum_ar <= cumsum_ar` AND `pad_i * pad_j`. Same.
- KV cache reuse: both compute prefix-only forward to fill cache, then
  expert forward on suffix attending to cached prefix K, V.
- `_gated_residual`: `x + y * gate` (or `x + y` if gate is None). Same.
- `_apply_rope` (JAX) vs `apply_rotary_pos_emb` (PT): mathematically
  equivalent (different but congruent formulations).
- LoRA merge math: `merged = base + scaling * (la @ lb)` for both attn
  einsum and FFN. scaling=1.0 confirmed for our checkpoints (alpha=rank).

## Still-open investigation paths

These haven't been ruled out yet:

1. **Per-layer suffix hidden state divergence**
   - We've done per-layer for the PREFIX. Haven't done it for the SUFFIX
     (gemma_expert layers). The cos=0.991 on suffix_out means each of the
     18 expert layers loses ~0.05% cosine + ~0.04% norm.
   - Need: monkey-patch JAX `Block.__call__` to capture per-layer
     gemma_expert outputs, then compare to PT layer-by-layer.
   - Initial attempt with `jax.disable_jit()` failed due to
     `TracerArrayConversionError`. Need a different approach (e.g.,
     `jax.experimental.callback` or rebuild the model without
     `nn.scan`).

2. **AdaRMS Dense modulation dtype**
   - JAX: `nn.Dense(...)` with `dtype=x.dtype` (= bf16) → modulation in bf16
   - PT: `nn.Linear` with weight kept in fp32 → modulation in fp32
   - PT modulation is MORE precise. Probably HELPS PT, not hurts. But
     worth verifying by forcing bf16 modulation in PT (one-line cast).

3. **LoRA merge math precision**
   - JAX: `out = w @ x + scaling * (lora_a @ lora_b) @ x` (3 matmuls, bf16)
   - PT: `out = (w + scaling * lora_a @ lora_b) @ x` (1 matmul, bf16)
   - Different rounding. JAX has MORE rounding error per LoRA application.
     PT has less. But this should mean PT is MORE accurate, not less.
   - To rule out: implement runtime LoRA on PT side (store lora_a/lora_b
     separately, apply JAX-style at every forward). Slow but proves it.

4. **HuggingFace transformers Gemma layer details**
   - We use `transformers_replace/models/gemma/modeling_gemma.py` which
     overrides the standard HF code. Audit this for any deviation from
     JAX semantics, especially in:
     - `_gated_residual`
     - `apply_rotary_pos_emb`
     - `eager_attention_forward`
     - `GemmaMLP.forward`

5. **RoPE precision**
   - JAX `_apply_rope` uses fp32 cos/sin.
   - PT `rotary_emb` precision: needs verification.
   - max_wavelength = 10000 on both. Confirmed.

6. **Suffix attention mask**
   - JAX masks padding tokens in prefix when suffix attends to them.
   - TurboPi does NOT mask (works on LIBERO).
   - We match JAX (mask padding). But the JAX TRAINING-time code might
     differ from sample-time. Worth double-checking JAX training mask.

7. **Numerical differences in suffix self-attention vs prefix forward**
   - In sample_actions, suffix tokens have positions
     `prefix_len + cumsum(suffix_mask) - 1`. Positions for suffix go
     768+200=968 through 1017.
   - Cross-check RoPE values at these high positions between JAX and PT.

8. **action_out_proj bias values**
   - PT bias values are small (~0.01–0.02 range). Verified non-zero.
   - Need: directly compare `m.action_out_proj.bias` values between JAX
     `_model.action_out_proj.bias` and PT counterpart (cos + max diff).

9. **gemma_expert specifically vs paligemma**
   - The paligemma prefix forward has good parity (cos=0.97 lang, 0.97
     image). The gemma_expert suffix forward also has cos=0.991.
   - Question: is the BIAS in gemma_expert systematically different from
     paligemma? E.g., do certain Q/K/V projections diverge more?

10. **Try Triton-based JAX-equivalent matmul kernels**
    - JAX (XLA) might use slightly different matmul algorithms than
      PyTorch (cuBLAS / CUTLASS). Same precision, different reduction
      order. Could account for the 0.05% per-layer drift.
    - Probably impossible to eliminate without a custom kernel.

11. **Compile vs eager**
    - Current code: `self.sample_actions = torch.compile(..., mode='max-autotune')`.
    - Comment in `pi0_pytorch.py` notes that EAGER gives cos=-0.96 (!)
      on pi05_libero and compile fixes it to 0.85. That's a HUGE swing
      from compile. Suggests compile is silently doing some
      precision/algorithmic substitution.
    - Worth testing: run with `OPENPI_PT_DISABLE_COMPILE=1` (if we add
      this) and see what eager gives on chocolate_bars.

---

## Action items (next session)

In priority order:

- [ ] Per-layer **suffix** (gemma_expert) hidden state diff (analog of
      `diag_per_layer.py` but for the expert). Will pinpoint which of
      the 18 expert layers introduces the most drift.
- [ ] Verify `action_out_proj.bias` matches between JAX and PT.
- [ ] Try forcing bf16 modulation in PT AdaRMS to match JAX exactly.
- [ ] Audit `transformers_replace/models/gemma/modeling_gemma.py` for
      any deviation from JAX `Block.__call__` (line by line).
- [ ] Test eager (no torch.compile) on chocolate_bars to see if compile
      is masking a bigger bug.
- [ ] Implement runtime LoRA on PT side (parallel base + lora paths) to
      rule out LoRA merge precision as the source.
- [ ] If all else fails: a **calibration hack** — apply a per-joint
      scale factor learned from JAX vs PT comparison on a held-out
      validation set. Pragmatic last resort to get the robot working
      while we keep debugging.

---

## Commit history (relevant)

```
197a88d pi0_pytorch: make matmul precision env-var configurable (default 'high')
b89bada diag: add suffix + full-loop comparisons that isolate residual PT bias
621d5c1 [previous: fp32 attention fix for prefix lang tokens]
[older: weight-tying re-tie fix in model.py]
[older: LoRA merge in convert_jax_model_to_pytorch.py]
[older: KV cache port from TurboPi]
```
