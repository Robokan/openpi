# JAX → PyTorch Conversion of LoRA-Finetuned π0.5

Engineering record of how the OpenArm `pi05_openarm_ngc_lora_v4` checkpoint
gets from a JAX-trained orbax checkpoint to a working PyTorch inference
server, with the gotchas that took 4 days of debug to find.

**TL;DR:** Don't pre-merge LoRA into base weights and quantize. Keep
`lora_a`/`lora_b` separate at runtime and apply them as two matmuls,
matching JAX's flow exactly.

---

## 1. The architecture being converted

π0.5 is a vision-language-action diffusion policy with:

```
PaliGemma (frozen base, 2B params, LoRA r=16/α=16)
  ├─ SigLIP vision tower (frozen)
  ├─ multi-modal projector
  └─ Gemma 2B language model (LoRA-tuned)
        layers[18] × { self_attn(q,k,v,o), mlp(gate,up,down) }

Action Expert (gemma_300m, LoRA r=32/α=32)
  └─ Gemma 300M variant (LoRA-tuned)
        layers[18] × { self_attn(q,k,v,o), mlp(gate,up,down) }

+ action_in_proj, action_out_proj, time_mlp_in, time_mlp_out (small dense heads)
```

The two Gemma stacks share KV via joint attention (paligemma keys/values feed
the action expert). The action expert reads diffusion-step + noisy-action
tokens and outputs a velocity field for Euler integration over 10 steps.

**LoRA scaling** = `alpha / rank` = 1.0 for both variants. Stored in JAX as
two extra tensors (`lora_a`, `lora_b`) next to every base weight `w`.

---

## 2. The conversion pipeline at a glance

`examples/convert_jax_model_to_pytorch.py` walks:

```
JAX orbax checkpoint (fp32 on disk)
    │
    ├─ restore_params(dtype=fp32)        # full-precision restore
    │
    ├─ slice_initial_orbax_checkpoint
    │     ├─ flatten params (PaliGemma + projections)
    │     └─ either:
    │           a) _merge_lora_into_base   ← OLD/broken path
    │           b) _extract_lora_pt        ← runtime-LoRA (current)
    │
    ├─ slice_paligemma_state_dict
    │     └─ reshape einsum weights → HF Linear shapes
    │           q_einsum:        (N,D,H) → (N*H, D)   .transpose(0,2,1).reshape
    │           kv_einsum:       (2,1,D,H) → split, .transpose
    │           attn_vec_einsum: (N,H,D) → (D, N*H)  .transpose(2,0,1).reshape
    │           gating_einsum:   (2,D,I) → split into gate_proj / up_proj
    │           linear:          (I,D)
    │
    ├─ slice_gemma_state_dict (same, but for action expert under _1 suffix)
    │
    ├─ Instantiate PI0Pytorch with dtype='float32'   ← critical (see §4.B)
    ├─ load_state_dict(strict=False)                 ← default cast, no assign
    ├─ .to(precision)                                ← fp32 or bf16 cast
    │
    ├─ safetensors.save_model(...)        → model.safetensors
    └─ if runtime LoRA: safetensors.save_file(lora_dict) → lora.safetensors
```

And at inference time, `BaseModel.load_pytorch`:

```
safetensors.load_file(model.safetensors)
  → model.load_state_dict(strict=False, assign=True)   # see §4.A
  → re-tie lm_head ↔ embed_tokens (§4.A)
  → if lora.safetensors exists: install_runtime_lora() (§3)
  → optional .to(bf16 / fp32) cast per OPENPI_PYTORCH_PRECISION
```

---

## 3. Runtime LoRA — the core insight

JAX's `lora.Einsum.__call__` does this on every forward:

```python
base   = einsum(eqn,   x, w)        # bf16 input, bf16 weight, fp32 accum, bf16 out
lora_a = einsum(eqn_a, x, la)       # bf16, output bf16  ← rank-r "neck"
lora_b = einsum(eqn_b, lora_a, lb)  # bf16, output bf16
result = base + scaling * lora_b    # bf16
```

The naive PyTorch conversion did this *once* at conversion:

```python
w_merged = (w_fp32 + scaling * la_fp32 @ lb_fp32).astype(bf16)
# then at runtime:
result = einsum(eqn, x_bf16, w_merged_bf16)
```

These two are mathematically identical **in fp32**. They differ **in bf16**
because the JAX path has an extra bf16 rounding on the rank-r intermediate
`lora_a` output. With r=16 and the base hidden-dim of 2048, that intermediate
has a much narrower dynamic range than the full weight, so quantizing it
introduces relatively larger error.

Per-matmul, the difference is small (cos≈0.999993, rel_err≈0.4%) and looks
like perpendicular noise. But compounded over 18 transformer layers × 10
diffusion-step Euler integrations, the actions ended up with a systematic
**8% magnitude shrinkage** and cos=0.996 vs JAX, enough to make the robot
arms drift upward.

### The fix: `src/openpi/models_pytorch/lora_runtime.py`

After base weights load, walk every LoRA-targeted `nn.Linear`, register
`lora_a` / `lora_b` as buffers, and monkey-patch the forward to add the
LoRA contribution via einsum. Three forward patches based on the
projection's shape semantics:

| Projection | JAX einsum | PT base | LoRA application |
|---|---|---|---|
| `q_proj` | `BTD,NDH→BTNH` | Linear(D, N·H) | `lora_b(lora_a(x))` with per-head einsum, then flatten |
| `k_proj` / `v_proj` | `BTD,DH→BTH` (per kv-head) | Linear(D, H) | standard `(x @ la) @ lb` |
| `o_proj` | `BTNH,NHD→BTD` | Linear(N·H, D) | flatten lora_a's (N,H,L) → (N·H, L); **pre-sum lora_b over N** at attach time, then standard matmul |
| `gate_proj` / `up_proj` | `BTD,DI→BTI` | Linear(D, I) | standard |
| `down_proj` | `BTI,ID→BTD` | Linear(I, D) | standard |

**Why the o_proj N-sum is correct:** JAX writes `eqn_b = "BTL,NLD→BTD"`. The
`N` axis appears only in the lora_b weight, not the input or the output, so
einsum implicitly sums over it. Pre-computing `lb.sum(axis=N)` in fp32 once
at load is mathematically identical, simpler at runtime, and avoids an
extra bf16 rounding.

### The patched forward

```python
def forward(x):
    out = base_forward(x)
    la = module.lora_a.to(x.dtype)   # match input dtype each call
    lb = module.lora_b.to(x.dtype)
    lora_int = einsum_or_matmul_a(x, la)   # rank-r intermediate
    lora_out = einsum_or_matmul_b(lora_int, lb)
    return out + module.lora_scaling * lora_out
```

The `.to(x.dtype)` cast handles whichever precision the model is running
in (fp32 / bf16 / future fp8) without storing duplicate copies.

### Result

| Setup | post-unnorm magnitude ratio |
|---|---|
| Pre-merge bf16 | **0.918** (8% bias — robot drifts) |
| Runtime LoRA, bf16 | 0.9928 (0.7% bias) |
| Runtime LoRA, fp32 | **0.9974** (0.26% bias — robot works) |

---

## 4. Conversion-time landmines

Things that silently broke inference even when "the converter ran fine."

### A. Weight tying (HuggingFace PaliGemma)

HF ties `lm_head.weight` ↔ `embed_tokens.weight` at construction. When you
load with `safetensors.torch.load_file(...) + load_state_dict(assign=True)`,
the saved tensor *replaces* `lm_head.weight`, **breaking the aliasing**.
The other half (`embed_tokens.weight`) stays at random init → all language
token embeddings random → cosine on the language slice of the prefix is ~0.

The safetensors file only contains one of the tied weights by default
(deduplication in `save_model`). So after load you must explicitly re-tie:

```python
pg.model.language_model.embed_tokens.weight = pg.lm_head.weight
ge.model.embed_tokens.weight = ge.lm_head.weight   # gemma expert
```

This is done in `BaseModel.load_pytorch` after every `load_state_dict`.
Without it, the LM is incoherent and parity is impossible.

### B. Dtype during conversion

Original converter instantiated `PI0Pytorch(model_config)` with
`model_config.dtype="bfloat16"`. The freshly-built model had **bf16
parameters**. Then `load_state_dict(all_params, strict=False)` (no
`assign=True`) loaded our **fp32 LoRA-merged weights** into those bf16
parameters, silently casting them down. The per-element bf16 rounding
step at the base-weight magnitude is the same order as the LoRA delta
(~1% of base norm) — so the LoRA contribution gets partially **thrown
away in the cast**.

**Fix:** instantiate at fp32 during conversion, do the load, then
`.to(precision)` at the very end:

```python
fp32_model_config = dataclasses.replace(model_config, dtype="float32")
pi0_model = PI0Pytorch(fp32_model_config)
pi0_model.load_state_dict(all_params, strict=False)   # fp32 → fp32 = no-op
pi0_model = pi0_model.to(precision)                   # explicit final cast
```

We considered `assign=True` to avoid the dtype cast, but it broke the
weight tying above. The fp32-instantiate approach is the clean one.

### C. The bf16 noise in LoRA merge

Even with the fp32-instantiate fix, the converter's `_merge_lora_into_base`
did the merge in fp32 but then stored the result in whatever the safetensors
file's precision is. With `--precision bfloat16` that's one bf16 rounding
on the merged weight. JAX's runtime equivalent involves a *different*
sequence of bf16 roundings (two matmuls + an add). Hence runtime LoRA in §3.

### D. Action projections must stay aligned with inputs

`action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out` are tiny
dense heads that sit outside `paligemma_with_expert`. The runtime function
`to_bfloat16_for_selected_params` is called on `paligemma_with_expert` only,
so these projections keep whatever dtype the safetensors file had. Mixing
dtypes (fp32 input × bf16 weight) errors out in eager mode.

**Two reliable patterns:**
- Convert with `--precision float32` and let `OPENPI_PYTORCH_PRECISION=float32`
  control inference precision uniformly (what's deployed now).
- Or, in `sample_actions`, cast `noisy_actions` / `timestep` to match the
  projection's weight dtype.

`OPENPI_PYTORCH_PRECISION=bfloat16` works because `torch.compile`'s inductor
inserts type-promotions. Eager mode (used for debug) does not.

### E. RoPE `inv_freq` precision

`GemmaRotaryEmbedding` keeps `inv_freq` as an fp32 buffer (matches JAX). A
naive `model.to(bfloat16)` casts buffers too, dropping ~3 decimal digits
on `inv_freq`. The cos/sin tables then drift by ~1e-3 per position, which
compounds to ~8% chunk-level bias entirely on its own (we found this
*before* the LoRA bug, separately).

**Fix in `to_bfloat16_for_selected_params`:** after `self.to(bf16)`, walk
named_buffers and promote any `rotary_emb.inv_freq` back to fp32.

---

## 5. Validation strategy that actually worked

Many fancy diagnostics; only a few were decisive. Listed in order of
diagnostic value:

1. **End-to-end action parity** (`diag_runtime_lora.py`):
   `JAX vs PT` raw + post-unnormalize cos/ratio. **The number we cared about.**
2. **LoRA-off / LoRA-on toggle** (`diag_no_lora.py`):
   Zero JAX's `lora_a`/`lora_b`, compare to PT-without-LoRA-merged. Match → bug
   is LoRA-related. Mismatch → bug is in base. *This isolated the problem.*
3. **Per-step diffusion diff** (`diag_per_step_lora.py`):
   Compare `v_t`, KV cache, suffix_out at every Euler step. Showed the bias
   grew per-layer.
4. **Single-matmul LoRA test** (`diag_q_proj_bf16_proper.py`):
   Synthetic input → JAX-style vs PT-style with proper intermediate rounding.
   First test that produced a measurable per-matmul mismatch (0.4%).
5. **Checkpoint diff** (`diag_checkpoint_diff.py`):
   `JAX-computed merged weight` vs `PT safetensors merged weight` per layer.
   Caught the bf16 truncation bug (§4.B) before we got to the runtime bug.

Don't trust:
- **Random-input forward parity.** Real inputs are needed; the bias depends
  on signal structure.
- **Cos alone.** Magnitude (`ratio`) was the discriminating metric. cos was
  0.996 for both broken and working states.
- **fp32 mode "should be identical."** True only if both frameworks are
  *actually* in fp32; flax linen `nn.scan` parameters aren't reachable via
  `nnx.iter_graph`, so naive `nn.Variable.value.astype(fp32)` only catches a
  fraction of the params. JAX-side fp32 is hard.

---

## 6. Why the design ended up this way

A few decisions worth defending:

**Why monkey-patch `Linear.forward` instead of subclassing?**
The HuggingFace `GemmaAttention` / `GemmaMLP` are instantiated by config
and used unmodified. Subclassing would require touching the transformers
override files (`transformers_replace/...`) for every Linear, plus
threading LoRA presence through the config. The patch is purely additive,
gated on `lora.safetensors` existing, and zero risk for non-LoRA models.

**Why a separate `lora.safetensors` file?**
`safetensors.torch.save_model` deduplicates tied weights. Mixing LoRA
buffers (registered with `register_buffer(persistent=False)`) with tied
parameters in one save_model call would either lose dedup or lose the
LoRA. A second file is two lines of code, makes loading explicit, and
lets us A/B test by deleting the file.

**Why pre-sum o_proj's lora_b over N?**
JAX writes the einsum that way, so semantically the N axis is summed.
Doing it once at fp32 attach time is exact; doing it at every bf16
forward would add an unnecessary rounding.

**Why store lora_a as `(N,D,L)` for q_proj instead of flattening?**
The per-head structure is meaningful — quantization (FP8/FP4) wants to
calibrate per-head scales. Keeping the head axis exposed makes future
work easier.

---

## 7. Map for future quantization work

The LoRA application now lives in one file: `lora_runtime.py`. For FP8/FP4
the natural pattern is:

- **Base weights → low precision (FP8 / FP4)** via Transformer Engine,
  TensorRT-LLM, or bitsandbytes 4-bit Linear.
- **LoRA `lora_a` / `lora_b` stay in fp16 or bf16.**
  They're tiny (rank ≤ 32) and the LoRA contribution is small enough that
  quantizing it loses accuracy fast (this is the standard QLoRA recipe).
- **Patched forward stays the same.** `base_forward(x)` calls into the
  quantized backend; `+ scaling * (x @ la) @ lb` is fp16/bf16. The two
  results add at the higher precision.

Concretely we'd modify `install_runtime_lora` to:
1. Replace each LoRA-targeted `nn.Linear` with a `te.Linear` / `bnb.Linear4bit`
   pre-conversion, OR quantize in-place after load.
2. Keep the existing einsum-based LoRA forward unchanged.
3. Add an explicit dtype promotion at the add: `base_out.float() + lora_out.float()`
   to avoid catastrophic mantissa loss at fp8.

If we need TensorRT engine builds, we'd export the *base* model only (no
LoRA) to ONNX/TRT, then apply LoRA in a thin PyTorch wrapper around the
TRT engine. Same separation; the merge stays out of the engine.

---

## 8. File-by-file map

| File | Role |
|---|---|
| `examples/convert_jax_model_to_pytorch.py` | JAX → PT conversion; `OPENPI_PT_RUNTIME_LORA=1` emits separate `lora.safetensors`. |
| `src/openpi/models_pytorch/lora_runtime.py` | Runtime LoRA application: shape conversion + forward patches. |
| `src/openpi/models/model.py:load_pytorch` | Loads `model.safetensors`, re-ties weights, calls `install_runtime_lora` if LoRA file present. |
| `src/openpi/models_pytorch/pi0_pytorch.py` | The diffusion sampling loop; embed_prefix/suffix; KV cache. |
| `src/openpi/models_pytorch/gemma_pytorch.py` | `PaliGemmaWithExpertModel` wrapper, `to_bfloat16_for_selected_params`, fp32 attention accumulator, RoPE fp32 fix. |
| `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py` | Pi05-specific overrides: AdaRMS, gated residual, no internal `hidden_states * normalizer`. |
| `src/openpi/models/lora.py` | JAX-side reference: `Einsum`, `FeedForward` with LoRA. |
| `scripts/diag_runtime_lora.py` | Definitive parity test. |
| `scripts/diag_no_lora.py` | LoRA-on/off isolation. |
| `scripts/diag_q_proj_bf16_proper.py` | Per-matmul bf16 rounding comparison. |
| `PYTORCH_PARITY_DEBUG.md` | Verbose debug log (this is the polished summary). |
