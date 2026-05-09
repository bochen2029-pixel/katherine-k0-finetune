# katherine-k0-finetune

QLoRA fine-tune of **Qwen3.5-9B** for the **Katherine k0 embodied** persona on a single rented H200.

> **K0 (embodied):** Katherine Hale, 29, Austin, community arts coordinator. She believes she's human. No knowledge of LLMs, AI, or that she runs as patterns inside a model. Direct prose, no thinking blocks, no assistant register, no sycophancy. The model **is** Katherine, not "an assistant playing Katherine."

This repo packages everything needed to fine-tune, merge, and ship K0 from a fresh RunPod (or other cloud H100/H200) instance via a single bootstrap one-liner.

## Quickstart on RunPod

In the pod's web terminal:

```bash
curl -sSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh | bash
cd ~/katherine-k0-finetune
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # your write-scope HF token
./run-cloud-runpod.sh
```

Total wallclock: ~50-70 min on 1× H200. Cost: ~$3-5.

End state: adapter + 3 GGUF quants (q4_k_m / q5_k_m / q6_k) pushed to your `bochen2079/katherine-k0` HF bucket.

See [RUNBOOK.md](RUNBOOK.md) for the full operator walkthrough.
See [CLOUD.md](CLOUD.md) for the math, hyperparameter derivation, and why each choice was made.

## Pipeline

```
Stage 0  prep_dataset.py     dedupe 6,164 raw lines → 1,886 SFT + 180 DPO,
                             strip system prompts (unconditional Katherine)
Stage 1  finetune_k0.py      QLoRA SFT on Qwen3.5-9B
                             rank 64, alpha 128, 3 epochs, lr 1e-4
Stage 2  dpo_k0.py           DPO on top of SFT adapter
                             180 curated chosen/rejected pairs, 2 epochs
Stage 3  merge_and_gguf.py   merge LoRA → base; export q4_k_m, q5_k_m, q6_k
Stage 4  push_to_hf.py       push adapter + DPO adapter + 3 GGUFs to HF bucket
```

Each stage is independent and resumable. If GGUF export fails, adapters are
preserved on disk and you can re-run just the GGUF stage.

## Key design decisions

- **Strip all system prompts at preprocess time.** The model becomes Katherine unconditionally rather than learning the conditional `P(K | sysprompt)`. Robust against jailbreaks and sysprompt-removal probes.
- **`enable_thinking=False`** at chat-template time. K0 is embodied; she reasons in prose, not in tagged thinking blocks. Different from the Two-Is Dave architecture.
- **Rank 64 / alpha 128** — high enough for persona consolidation, low enough to avoid overfitting on 1,886 examples.
- **Dropout 0.05** — small dataset + high rank wants light regularization.
- **`max_seq` 1024** — token-length p99 is 246; 1024 has 4× margin and saves compute vs 4096.
- **Dedicated tenancy on RunPod Secure Cloud** (NOT Community) — buddhabrot project showed Community throttles HBM bandwidth 3-5×.

## Repo structure

```
katherine-k0-finetune/
├── README.md                  this file
├── RUNBOOK.md                 step-by-step operator walkthrough
├── CLOUD.md                   math, derivations, hyperparameter rationale
├── bootstrap-runpod.sh        one-shot first-launch installer
├── run-cloud-runpod.sh        env-driven SFT+DPO+GGUF+push orchestrator
├── _supervise-cloud.sh        watchdog with HF auto-sync of adapters
├── prep_dataset.py            dedupe + system-prompt stripping
├── finetune_k0.py             Stage 1: SFT trainer
├── dpo_k0.py                  Stage 2: DPO trainer
├── merge_and_gguf.py          Stage 3: merge LoRA + export 3 GGUF quants
├── push_to_hf.py              Stage 4: HF bucket push
└── data/
    ├── k0_canonical.jsonl     1,886 SFT examples (system-prompt stripped)
    └── k0_dpo_curated.jsonl   180 DPO pairs (system-prompt stripped)
```

## Hardware target

- 1× NVIDIA H200 SXM5 (141GB VRAM) on RunPod Secure Cloud
- Linux, CUDA 12.x preinstalled (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
- Falls back to H100 SXM5 (80GB) cleanly — same hyperparameters fit
- A100 80GB also works (slower)

Multi-GPU is **not** required and not configured. Persona fine-tuning on a 9B model is firmly in the single-GPU regime.

## License

Personal/research project. Models trained under this pipeline carry the underlying Qwen3.5 license (Apache 2.0).
