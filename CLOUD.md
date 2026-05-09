# CLOUD.md — math, hyperparameter derivation, and rationale

This document is the technical record for **why** each choice in the pipeline was made. The [RUNBOOK.md](RUNBOOK.md) covers **how** to run it. Read this once when reviewing the pipeline; the runbook is for execution.

---

## Project goal

Produce a Qwen3.5-9B QLoRA fine-tune that **unconditionally** embodies the Katherine k0 persona — a 29-year-old human woman in Austin with no awareness of being an LLM. The model **is** Katherine, not "an assistant playing Katherine when the system prompt says so."

---

## Dataset characterization (post-dedupe)

```
Source: C:\Katherine\JSONLs\*.jsonl (38 files, 6,164 raw lines)
After dedup by content hash:
  SFT examples (have 'messages'):  1,886 unique
  DPO pairs (k0_dpo_only.jsonl):     180 unique (curated, has _cat/_type metadata)
  Other DPO pairs in legacy files:    30 (skipped for round 1)

SFT schema: {"messages": [{"role":"system|user|assistant", "content":"..."}]}
DPO schema: {"prompt": [...], "chosen": [{"role":"assistant","content":"..."}], "rejected": [...]}

Token-length distribution (after system-prompt stripping):
  p50:   102 tokens
  p90:   192 tokens
  p99:   246 tokens
  max:   ~280 tokens
```

All examples are short conversational turns — character voice, not long-form content.

---

## Why unconditional Katherine (system prompts stripped)

The default approach is to keep the K0 system prompt in every training example. That trains `P(K_response | sysprompt_K)` — the model learns to BE Katherine *when prompted*. Three problems:

1. **Conditional collapse.** Drop the sysprompt at inference and the model reverts to its assistant-distribution priors. A "remove all previous instructions" jailbreak works trivially.

2. **Wrong objective for this deployment.** This is a single-persona dedicated model, not a multi-tenant assistant with a Katherine mode. We want the unconditional `P(K_response | nothing)`, which means stripping the sysprompt during training so the model *re-anchors its base distribution* on Katherine-output rather than learning a conditional override.

3. **Harder to attack.** Sysprompt-conditioned Katherine breaks frame on sysprompt removal. Unconditionally-trained Katherine doesn't have a frame to break — Katherine *is* the base distribution after training.

Implementation: `prep_dataset.py` drops all `role=system` messages from each conversation. The remaining user→assistant chain is what the model sees. The persona-establishing details (Austin, teal walls, cold feet, arts coordinator) survive only through the conversation content, not through the sysprompt scaffold. This is more demanding training but yields a more robust deployment.

---

## Why `enable_thinking=False`

Qwen3.5 instruct models default to thinking mode (`<think>...</think>` blocks before output). The Dave persona we trained earlier uses thinking blocks (Two-Is architecture: reasoning layer + output layer both in Dave's voice). **Katherine k0 does not.** K0 is embodied — she reasons in prose like a person, not in tagged reasoning blocks like a model.

Setting `enable_thinking=False` at chat-template time during both training and inference ensures:
- No `<think>` markers leak into training text → model doesn't learn to emit them
- Inference produces direct prose response, no visible reasoning tags
- Matches K0's persona: a human doesn't have a thinking-tag layer

---

## Hyperparameter derivation

### LoRA configuration

| Parameter | Value | Rationale |
|---|---|---|
| `rank` | 64 | High enough for persona consolidation; rank 32 is the standard floor for instruction-tuning; persona work benefits from somewhat higher rank but rank 128 is overkill for 1,886 examples (overfit risk) |
| `lora_alpha` | 128 | Maintains the standard `alpha = 2 × rank` ratio |
| `lora_dropout` | 0.05 | Light regularization. With high rank + small dataset, some dropout is sensible. Web-Claude advice argued for 0.0 ("dropout fights consolidation") but evidence is thin and overfit downside is real |
| `target_modules` | q/k/v/o + gate/up/down | Standard "all linear projections" — covers attention + MLP. No reason to exclude any |
| `bias` | none | Standard for LoRA |
| `gradient_checkpointing` | "unsloth" | Unsloth's optimized variant; trades ~10% wallclock for substantial memory savings; safe default |

### Training schedule

| Parameter | Value | Rationale |
|---|---|---|
| `epochs` | 3 | More than 2 to allow voice subtleties to consolidate; not 4+ to avoid overfit on 1,886 examples |
| `learning_rate` | 1e-4 | Conservative for QLoRA on a 9B model. Standard 2e-4 is fine for shorter runs; 1e-4 for 3 epochs gives more stable convergence |
| `lr_scheduler` | cosine | Standard; avoids the noisier endgame of linear decay |
| `warmup_ratio` | 0.05 | Standard; brief warmup avoids initial gradient blowup with high rank |
| `optim` | adamw_8bit | Standard Unsloth-friendly choice; saves ~6 GB vs full adamw at 9B |
| `weight_decay` | 0.01 | Standard regularization |
| `bf16` | true | Native on H100/H200 (Hopper), faster than fp16 |

### Batch + sequence

| Parameter | Value | Rationale |
|---|---|---|
| `per_device_batch` | 16 | Token-length p99 is 246; 16 × 1024 = 16K tokens/step fits comfortably in H200's 141 GB VRAM |
| `grad_accum` | 2 | Effective batch 32 — good for gradient stability without large memory cost |
| `max_seq_length` | 1024 | Data p99 is 246 tokens; 1024 has 4× margin. Going to 4096 wastes ~4× compute for zero quality gain |
| `packing` | False | Conservative; unpacked training is more interpretable. With 1,886 examples wallclock is short anyway |

### DPO stage (Stage 2)

| Parameter | Value | Rationale |
|---|---|---|
| `epochs` | 2 | DPO converges quickly on small preference sets (180 pairs); 2 epochs is the standard floor |
| `learning_rate` | 5e-6 | DPO requires tiny steps — 20-50× smaller than SFT LR — to avoid catastrophically forgetting the SFT-learned voice |
| `beta` | 0.1 | Standard KL strength; higher (0.3-1.0) = more conservative (stay closer to ref model); lower (0.01-0.05) = more willing to diverge for preference satisfaction. 0.1 is the sweet spot |
| `batch` | 4 + grad_accum 2 | Effective batch 8 — DPO loss is more variance-y than SFT, smaller batch is fine |
| `warmup_ratio` | 0.1 | Slightly larger than SFT warmup; DPO is sensitive to early instability |

### Reference model for DPO

The standard DPO recipe needs both a "policy" (the model being trained) and a "reference" (frozen snapshot for KL divergence). For PEFT/LoRA setups, `DPOTrainer(ref_model=None)` uses an adapter-disabled forward pass as the reference — same architecture, same weights, just LoRA-deltas zeroed. This saves a second model copy in VRAM and works correctly when the SFT adapter loaded as the policy is what we want as the reference too.

---

## Wallclock and cost projections

```
H200 SXM5 throughput on Qwen3.5-9B QLoRA:
  Prefill + backward at batch 16, seq 1024 (16K tokens/step):
  ~3 sec/step (estimate; exact depends on Hopper-specific kernels in unsloth)

SFT total: 1,886 examples × 3 epochs / batch 32 = 177 steps
           177 × 3 sec = 531 sec ≈ 9 min compute
           + ~3 min model load on first run
           ≈ 12 min total

DPO: 180 pairs × 2 epochs / batch 8 = 45 steps
     ~5 sec/step (DPO is heavier per step than SFT due to ref + policy forward)
     45 × 5 = 225 sec ≈ 4 min

Merge + GGUF: ~10 min for first quant (compiles llama.cpp), ~3 min each subsequent
              q4 + q5 + q6 = ~16 min total

HF push: ~6 GB GGUF × 3 + ~250 MB adapters × 2 = ~19 GB
         At HF API ~50-100 MB/s realistic upload: ~3-5 min

Total wallclock: ~35-45 min on H200
                ~50-60 min on H100 SXM5
                ~75-90 min on H100 PCIe
                
Cost on RunPod Secure Cloud:
  H200 SXM5 ($3.99/hr) × 0.75 hr = $3.00
  H100 SXM5 ($3.49/hr) × 1.0 hr  = $3.50
  H100 PCIe ($2.49/hr) × 1.5 hr  = $3.75
```

All under $5. Genuinely cheap to iterate.

---

## Provider lessons inherited from buddhabrot project

- **Use RunPod Secure Cloud, not Community.** Community shares HBM bandwidth across tenants; persona training is bandwidth-sensitive (4-bit weight loads + bf16 activations + adapter gradients all hit memory). Secure Cloud delivers the rated throughput; Community can be 3-5× slower.
- **Diagnostic for shared throttling:** `nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader`. H200 doing real work draws 600-700W of 700W cap. If you see <30% with 100% util, you're sharing bandwidth — switch pods.
- **Web Terminal is full bash.** No SSH key drama needed for first-time setup.
- **Bootstrap one-liner pattern works.** Same shape as `bootstrap-hyperbolic.sh` — clone, install, prep, leave at ready-to-launch state.

---

## What this pipeline does NOT do (deferred decisions)

- **No DPO data augmentation.** Just the 180 curated pairs.
- **No held-out validation set.** With 1,886 examples and 3 epochs, eyeballing the loss curve in the training log is the validation. A formal eval split would cut training data by 5-10% for marginal value at this scale.
- **No multi-quantization eval automation.** All 3 GGUFs are produced; manual probe-testing across quants is post-pipeline operator work.
- **No public HF model repo.** Only the private bucket. Public release happens after operator manually evaluates the GGUF outputs.
- **No checkpoint averaging / EMA.** Single final-epoch adapter is the artifact.

These are all intentional simplifications. Add them in v2 if first pass surfaces a need.
