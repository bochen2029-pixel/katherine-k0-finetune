# K0 v2 Dataset: Pending Updates Tracker

Tentative placeholders for things that need to happen before specific generation steps. Don't lose these.

## Resolved (kept for reference)

### inc_002 + tier_1000 — COMPLETE 2026-05-10

inc_002 generated end-to-end: 501 traces (401 SFT across A-I + 100 DPO across 9 subtypes). Cumulative tier_1000 built (1003 traces) and validated by both per-domain validator and the new cumulative validator. K0_EXEMPLARS.jsonl created (86 calibration anchors, one per fine sub-cat + 6 long-form variants). Domain I inc_001 vision-honesty audited: 6 brackets corrected to vision-tower-style generic descriptions. New tooling: scripts/validate_cumulative.py + scripts/audit_consistency.py + dataset/v2/dataset_schedule.csv. Next: inc_003 (1500 traces, adds Domain P audio-register at 5%).

### Domain I (Vision) gate — SATISFIED 2026-05-10

Trace generation prompt and validator both updated before Domain I generation. Domain I (32 traces) and DPO-VISION-REGISTER (10 pairs) generated and validated 100% pass at inc_001. Vision section of trace_generation_prompt.md is at the file's "VISION HANDLING" section. Validator has VISION_REGISTER_RX with persona-aware activation on Domain I and DPO-VISION-REGISTER traces.

### Three v2 soul doc date contradictions — FIXED 2026-05-10

Caught by fork-instance audit. Same shape as the original 37-vs-20-year James-duration catch. All harmonized:

- §2.11 "three years ago" → "four years ago, in early 2022" (matches §5.4)
- §1.6 "thirty years of teaching" → "thirty-two years of teaching" (matches §5.4 Eleanor retired spring 2022 specific)
- §5.13 "in 2022 to see Daniel and Hannah's wedding" → "in 2023" (matches §5.4 "married since 2023")

### v2 DPO trainer — BUILT 2026-05-10

`scripts/dpo_k0_v2.py` exists. FastVisionModel-based, vision-preserving, points at v2 DPO data path, handles v2 schema. Hyperparams: epochs=2, lr=5e-6, beta=0.1. Not yet test-trained on hardware — verify with a dry run before T2500+ production. Spec is in CHANGELOG.md.

### Held-out eval set — GENERATED 2026-05-10

`dataset/v2/held_out_eval/eval_traces.jsonl` exists. 100 traces, balanced inc_001/002 distribution (A:18 B:10 C:14 D:19 E:11 F:9 G:5 H:6 I:8). Never to be included in any training tier; `build_cumulative.py` does not touch the directory. Use at T2500+ training to measure overfit/underfit.

### Domain P (Audio) pre-flight — DONE 2026-05-10

Trace gen prompt + validator + exemplars all updated. AUDIO HANDLING section added to `data/trace_generation_prompt.md`. P-domain hard checks (TTS-broken patterns, audio-modality ack, bracket vocab, identity-in-bracket) wired into `scripts/validate_k0_v2.py` with P-trace activation. 8 P-cat exemplars (P1-P8) appended to `K0_EXEMPLARS.jsonl` (now 94 anchors). Sanity check: existing tier_1000 SFT/DPO re-validated, no regressions.

**Convention chosen** (vision-tower-honesty principle applied to audio):
- Bracket = harness-detectable: `[voice]`, `[voicemail]`, `[in-person]`, `[recorded message]` + paralinguistic state combos (`[voice, sounds tired]`, etc.)
- Identity NOT in bracket — operator self-identifies in spoken transcript or comes from conversational context
- TTS-friendly output rules are HARD fails on P traces: no markdown, no URLs, no code blocks, no listicles, no all-caps emphasis, no abbreviations TTS mangles
- K0 substrate-naive about audio path: does NOT say "switching to voice mode", "thanks for calling", "hello this is Katherine"

**The `[Bo on the phone]` style from prior PLAN.md notes is DROPPED** as the audio analog of the vision-honesty failure (vision tower can't see "Bo" from pixels; ASR/diarization can't identify "Bo" from voiceprint). Validator now hard-fails it via AUDIO_IDENTITY_IN_BRACKET check.

## Before Domain P trace generation begins (inc_003)

- Smoke-test 3-5 P traces against the new validator before generating the full P batch (60 SFT + 15 DPO)
- If voice register or bracket format is off, fix BEFORE bulk generation to keep blast radius limited
- Generate full inc_003 P batch when smoke-test passes

## Before tier 2500+ training run

- Confirm `finetune_k0_v2.py` uses `FastVisionModel` (already done) so vision tower is preserved through training
- Verify `merge_and_gguf.py` no longer filters mmproj files (already done)
- Verify `push_to_hf.py` uploads mmproj alongside LLM GGUFs (already done)
- Consider Path B vs A for vision: at tier 2500+ we want real image substitution for Domain I traces. Plan a script that scans Domain I traces, reads the bracketed description, finds a matching image from a folder, and converts to multimodal JSON format

## Before merging tier_500 to a finetune run

- Review increment_001 cumulative once all domains are done
- Verify validator passes corpus-level (no single trope >5%, no single roster >10%)
- Confirm SFT/DPO ratio is roughly 80/20
- Run `build_cumulative.py` to produce `cumulative/tier_500/sft_train.jsonl` and `dpo_train.jsonl`

## When real images get sourced

- Decide convention for image filename mapping (image_001.jpg etc paired with traces by index, or filename baked into trace metadata)
- Build the substitution script that converts `[image: <description>]` placeholder traces to real multimodal format
- Test that Unsloth's FastVisionModel correctly tokenizes the substituted images

## When TTS/ASR harness is built

- Lock the operator-side scene-setting convention (the harness needs to know how to inject `[Bo on the phone]` style prefixes)
- Confirm K0 v2 trained on Domain P responds correctly when harness sends voice-flagged input
- Test paralinguistic cue handling end-to-end
