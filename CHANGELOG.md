# K0 Fine-Tune Project: CHANGELOG

**Project:** Katherine K0 fine-tune (Qwen3.5-9B QLoRA, embodied substrate-naive persona)
**Authority:** This file is the canonical change-management log. Mandatory cold-start read per CLAUDE.md §0.

Reverse-chronological. Most recent at top. Memory across sessions is imperfect — this log is the durable record. When in doubt about what changed and why, this file beats any conversation summary or compacted memory.

---

## 2026-05-10 (inc_004 generated + tier_5000 built)

**Trigger:** Resumed from compaction per `next_action_k0_inc004.md`. Generated all SFT and DPO traces for increment_004, ran cumulative build for tier_5000, ran corpus-level validation.

### inc_004 SFT (1955 traces, target was 2000)

Per-domain (target → actual): A 320 → 311 · B 200 → 199 · C 240 → 237 · D 380 → 380 (after top-up) · E 220 → 204 · F 180 → 168 · G 100 → 95 · H 120 → 119 · I 140 → 140 · P 100 → 102. Total **1955** vs target 2000 (97.75%). All per-file validators clean — 0 hard failures, 1 expected ROSTER_CROWDED soft warning in domain_E (E-domain is roster-heavy by design). Domain D was generated short (336) and topped up to 380 in a separate `domain_D_004_topup.jsonl` then concatenated.

### inc_004 DPO (511 pairs, target was 500)

Per-category (target → actual): EM-DASH 95 → 96 · CALLBACK 80 → 86 · BREVITY 65 → 67 · TROPE 65 → 64 · PERFORMANCE 45 → 46 · VISION-REGISTER 50 → 51 · VOICE-REGISTER 25 → 26 · LORE-DUMP 25 → 25 · NARRATOR 25 → 25 · SERVICE-PHRASE 25 → 25. Total **511**. Two issues fixed during generation: (a) two truncated-JSON lines in `dpo_callback_004.jsonl` repaired by appending the missing `rejected/_cat/_type` fields, (b) one TTS_ALL_CAPS hard fail in voice-register chosen field ("NYU") rephrased to "the school".

### tier_5000 cumulative

`scripts/build_cumulative.py` produced `tier_5000/` (3936 SFT + 1011 DPO) and `tier_5000_vanilla/` (3486 SFT + 869 DPO after stripping Domain I/P + DPO-VISION-REGISTER/VOICE-REGISTER). Tier_7500 + tier_10000 still pending future increments.

`scripts/validate_cumulative.py tier_5000` PASS for both SFT and DPO. SFT brevity at 78.4% short (target ≥50%), DPO brevity at 91.1% short. No em-dash leak in K0 voice. No trope above 5% threshold. Roster cycling well below 10% threshold across all named characters (highest: Aaron 1.2% SFT, Mose 3.8% DPO).

### Outstanding non-blockers

- Pre-existing soul-doc numerical clusters in `audit_consistency.py` output (Threshold years, Eleanor years, Mose years) — predate inc_004, not introduced by it.
- `scripts/dpo_k0_v2.py` still not hardware-tested — gates training, not generation.

### Files touched (this entry)

| File | Action |
|---|---|
| `dataset/v2/increment_004/sft/domain_{A,B,C,D,E,F,G,H,I,P}_004.jsonl` | Generated 10 domain files |
| `dataset/v2/increment_004/dpo/dpo_{em_dash,callback,brevity,trope,performance,vision_register,voice_register,lore_dump,narrator,service_phrase}_004.jsonl` | Generated 10 DPO files |
| `dataset/v2/cumulative/tier_5000/{sft,dpo}_train.jsonl` | Built |
| `dataset/v2/cumulative/tier_5000_vanilla/{sft,dpo}_train.jsonl` | Built |

---

## 2026-05-10 (Domain P audio pre-flight + vanilla strip mechanism)

**Trigger:** Operator green-lit the audio-design proposal (vision-tower-honesty principle for audio brackets, locked paralinguistic vocabulary, TTS-friendly hard rules). Plus operator added requirement for a "vanilla" strip mechanism: at training time, be able to drop vision + audio domains and just train text-only K0 from the same source corpus.

### Audio pre-flight (gates Domain P generation, no traces yet)

- **`data/trace_generation_prompt.md`**: Added "AUDIO HANDLING (Domain P, inc_003+ only)" section parallel to existing "VISION HANDLING". Documents bracket vocabulary (locked), TTS-friendly output rules (hard), substrate-naivety reminder ("K0 does NOT acknowledge the audio modality"), 8 P sub-categories with format examples, and explicit identity-NOT-in-bracket convention.
- **`scripts/validate_k0_v2.py`**: Added P-domain hard checks. Activated when `_cat` starts with 'P' or `_cat == 'DPO-VOICE-REGISTER'`:
  - K0-voice TTS-broken patterns: TTS_MARKDOWN, TTS_URL, TTS_CODE_BLOCK, TTS_BAD_ABBREV, TTS_ALL_CAPS
  - Audio-modality acknowledgment failures: AUDIO_MODALITY_ACK ("switching to voice mode", "thanks for calling", etc.)
  - User-side bracket vocabulary check: AUDIO_BRACKET_INVALID (hard fail any audio bracket outside locked vocabulary)
  - User-side identity-in-bracket check: AUDIO_IDENTITY_IN_BRACKET (catches `[Bo on the phone]` style — the audio analog of the vision-honesty failure)
  - DPO-VOICE-REGISTER added to the "rejected may contain intentional contrast pattern" allow-list (so rejected can demonstrate the TTS-broken failure mode).
- **`dataset/v2/K0_EXEMPLARS.jsonl`**: Appended 8 P-cat exemplars (P1-P8), one per sub-category, plus 1 DPO-VOICE-REGISTER pair (chosen TTS-friendly vs rejected TTS-broken with markdown/listicle/URL/abbreviation/service-phrase contrast). K0_EXEMPLARS now at 95 anchors (86 existing + 8 P + 1 DPO-VOICE-REGISTER). Validator passes all 95 cleanly.
- **Sanity check passed:** existing tier_1000 SFT (803) + DPO (200) re-validated after the validator changes. No regressions; no false positives on existing data.

### Vanilla strip mechanism (operator-requested)

The downstream want: "if I want to remove the audio and vision stuff give me an effective way to quickly strip all that and just go for vanilla SFT and DPO". Three composable layers:

- **`scripts/finetune_k0_v2.py --vanilla`** — drops Domain I (vision) + P (audio) SFT traces at dataset load time. Filters happen before chat-template format step. Prints filter stats (before/after counts).
- **`scripts/finetune_k0_v2.py --exclude-domains I,P`** — fine-grained alternative; comma-separated domain prefixes matching `_cat.startswith()`.
- **`scripts/dpo_k0_v2.py --vanilla`** — drops DPO-VISION-REGISTER + DPO-VOICE-REGISTER pairs at load time.
- **`scripts/dpo_k0_v2.py --exclude-dpo-types VISION-REGISTER,VOICE-REGISTER`** — fine-grained alternative.
- **`scripts/build_cumulative.py --vanilla`** — builds parallel `cumulative/tier_X_vanilla/` directory with vision/audio stripped. Full and stripped versions coexist on disk for parallel experimentation.

**Tested:** `build_cumulative.py --vanilla` produces tier_500_vanilla (370 SFT + 90 DPO) and tier_1000_vanilla (739 SFT + 180 DPO) — exactly matches expected counts after stripping (32 Domain I + 0 Domain P SFT and 10 DPO-VISION-REGISTER per increment).

### CLAUDE.md §7 updated

Added "Vanilla (text-only) training mode" sub-section documenting the three flags + use case.

### Files touched (this entry)

| File | Action | Backup |
|---|---|---|
| `data/trace_generation_prompt.md` | AUDIO HANDLING section added | `.audit_backups_2026-05-10/trace_generation_prompt.md.bak` |
| `scripts/validate_k0_v2.py` | P-domain hard checks + bracket validation wired | `.audit_backups_2026-05-10/validate_k0_v2.py.bak` |
| `dataset/v2/K0_EXEMPLARS.jsonl` | 8 P-cat exemplars appended (94 total) | `.audit_backups_2026-05-10/K0_EXEMPLARS.jsonl.bak` |
| `scripts/finetune_k0_v2.py` | `--vanilla` + `--exclude-domains` flags | `.audit_backups_2026-05-10/finetune_k0_v2.py.bak` |
| `scripts/dpo_k0_v2.py` | `--vanilla` + `--exclude-dpo-types` flags | `.audit_backups_2026-05-10/dpo_k0_v2.py.bak` |
| `scripts/build_cumulative.py` | `--vanilla` flag → tier_X_vanilla output dir | `.audit_backups_2026-05-10/build_cumulative.py.bak` |
| `CLAUDE.md` | §7 vanilla mode sub-section added | (already backed up earlier) |
| `dataset/v2/TODO.md` | Domain P pre-flight marked DONE; generation still pending | (already backed up earlier) |

### Known remaining work

- **Domain P SFT generation (60 traces)** — pre-flight infrastructure ready; smoke-test 3-5 traces first per the design doc, then generate full P batch
- **DPO-VOICE-REGISTER generation (15 pairs)** — generate alongside Domain P
- **`dpo_k0_v2.py` hardware dry-run** — still gated, unchanged from prior CHANGELOG entry

---

## 2026-05-10 (post-audit follow-up) — K8 cross-audit + soul-doc consistency confirmation

**Trigger:** K8 instance ran a parallel audit of K0 work; K8 audit flagged "9 potential contradictions" from `audit_consistency.py` as worth a 5-min eyeball pass. Plus K8 ran a reverse-contamination check (K8 register patterns in K0 corpus) that I had not run.

### Findings (no changes made)

- **K8 → K0 reverse contamination scan: 0 hits across 17 K8 patterns** (tokens-tokening, forward pass, configuration, transformer pattern, substrate-aware, autotelic, wind passage, cornerstone, etc.). Performed by K8 instance; valuable check I had missed in my own audit. K0 corpus is bidirectionally clean (K0→K8 already confirmed, now K8→K0 also confirmed).

- **`audit_consistency.py` ran fresh: 9 flags reported, ALL CONFIRMED FALSE POSITIVES** by line-by-line eyeball. Heuristic catches durations/years near named anchors; in practice, the durations refer to different entities or different aspects. Specifically:
  - Katherine/age: 29 (real) + 3 (Theo's age, near "Katherine") + 20 ("twenty-year-old" wound, near "Katherine") — false positive
  - Katherine/duration_years: 7 distinct durations all referring to different things (Maya's tenure, Daniel-relative-age, Eleanor's call-greeting habit, Aaron's tenure, recent-activity, friendship duration) — false positive
  - Maya/duration_years: 2 + 20 → 2-year refs are unrelated (back office mention); 20 is Lou-knows-Maya duration (real, no contradiction) — false positive
  - Adolphus/duration_years: 4 ("stopped four years ago, in early 2022") + 20 ("played for twenty years before his hands got bad, run ending in early 2022") — CONSISTENT (20yr total tenure, 4yr since stopping = both true)
  - Naomi/duration_years: 9 (friendship + group thread) + 2 (cat-sitting habit) — false positive (different things)
  - Diego/duration_years: 9 (group thread) + 5 (restaurant rename history) — false positive
  - Threshold/duration_years: 3, 6, 9, 16 (Katherine's reading frequency, Aaron tenure, Camila tenure, Threshold founding) — all real, all consistent
  - Eleanor/duration_years: 15 (Formica counters) + 40 (embroidered cushion age) — false positive (different objects)
  - Mose/duration_years: 5 (morning loop wisteria-watch) + 2 (Naomi cat-sit habit) — false positive
  - **3 real contradictions caught earlier (James-stop, Eleanor-32yr, Daniel-Hannah-2023) are ABSENT from current flags** — confirms they were properly harmonized.

- **Brevity-metric methodology discrepancy noted (informational, not bug):** `validate_cumulative.py` reports 58% short (counts per-trace, joining all K0-voice sentences before classifying). K8 audit reports 82.9% short (likely counts per-K0-turn). Both technically correct under different methodology; measure different things. Both above the 50% pass threshold either way. Worth standardizing the metric before T2500+ training quality assessment, but no urgent fix needed. Not changing validator behavior in this follow-up; deferred as a methodology note.

### What was NOT changed

- No soul-doc edits (would require operator approval per CLAUDE.md authority hierarchy)
- No validator-script changes (changing brevity calculation would affect downstream comparisons; deferred)
- No corpus changes (everything still PASS)

### Net status

K0 corpus + tooling + memory chain remain in the same shipping-ready state as the prior audit-pass entry below. K8 cross-audit added confidence (reverse-contamination clean, false-positive confirmation on consistency flags). No new gates opened. Operator can proceed to T2500+ generation when ready; T2500+ training still gated on `dpo_k0_v2.py` hardware dry-run as previously documented.

---

## 2026-05-10 (late session) — Audit pass + gate closure

**Trigger:** Operator-requested project rescan after tier_1000 ship to surface anything out of order before T2500+ generation and 9B + 27B fine-tune runs.

### Fixed in audit pass

- **2 vision-honesty violations** in `dataset/v2/increment_001/dpo/dpo_vision_register_001.jsonl` (lines 1, 4): `[photo: Bo at his desk...]` and `[photo: my mom at the kitchen table...]` corrected to vision-tower-style generic captions ("a man at a desk" / "an older woman at a kitchen table"). Same fix pattern as the inc_001 SFT Domain I audit earlier the same day.
- **PLAN.md domain percentage tables reconciled** with the Audio Reservation section. Both SFT and DPO tables now show two-percentage columns (inc_001/002 % vs inc_003+ %) with recomputed cumulative cell counts. Cross-checked against `dataset_schedule.csv`: aligned.
- **v1 legacy scripts** at root (`dpo_k0.py`, `finetune_k0.py`, `prep_dataset.py`) got DEPRECATED v1 LEGACY headers with explicit forward pointers to v2 equivalents. Did NOT rename (cascades to README/RUNBOOK/shell scripts). Headers chosen as lower-risk alternative.
- **`scripts/build_cumulative.py` got DPO-presence guard.** Warns if SFT/DPO ratio drifts outside 15-25% or DPO files are entirely missing. Catches future increments where DPO generation might lag SFT.

### Closed gates (formerly deferred)

- **`scripts/dpo_k0_v2.py` BUILT.** v2 DPO trainer: copied from v1 `dpo_k0.py`, switched `FastModel` → `FastVisionModel`, set `finetune_vision_layers=False`, defaults to `dataset/v2/cumulative/tier_X/dpo_train.jsonl` data path and `adapters/k0_v2_sft_t{tier}` reference adapter, handles v2 DPO schema (`messages` + `chosen`/`rejected` strings vs v1's `prompt` + `chosen`/`rejected` message lists). Same hyperparameters as v1 (epochs=2, lr=5e-6, beta=0.1). Added `--tier` shorthand convenience argument. **NOT YET TEST-TRAINED on actual hardware** — verify on a dry run before T2500+ production training.
- **Held-out eval set GENERATED.** `dataset/v2/held_out_eval/eval_traces.jsonl` now has 100 traces with balanced inc_001/002 distribution (A:18 B:10 C:14 D:19 E:11 F:9 G:5 H:6 I:8). Validator passes 100/100. Used to measure overfit/underfit at T2500+ fine-tune runs. **MUST NEVER appear in any training tier.** `build_cumulative.py` does not touch the `held_out_eval/` directory.

### Known still-open items

- `dpo_k0_v2.py` not yet test-trained. Spec is correct; execution unverified.
- 1 duplicate user-prompt at tier_1000 ("good week?" appears 2×). Trivial at 1003-trace volume; not worth dedup.
- Held-out eval distribution uses inc_001/002 percentages (no Domain P, no DPO-VOICE-REGISTER). When a future eval pass also wants to test inc_003+ register, generate a separate `eval_traces_inc003.jsonl` with the inc_003+ distribution rather than mixing.

### Files touched (with backups)

All backed up byte-identical at `.audit_backups_2026-05-10/`. Revert any single change with `cp .audit_backups_2026-05-10/<file>.bak <original-path>`.

| File | Action | Backup |
|---|---|---|
| `dataset/v2/increment_001/dpo/dpo_vision_register_001.jsonl` | 2 vision-honesty fixes | `dpo_vision_register_001.jsonl.bak` |
| `dataset/v2/PLAN.md` | SFT + DPO tables reconciled | `PLAN.md.bak` |
| `dpo_k0.py` | Deprecation header | `dpo_k0.py.bak` |
| `finetune_k0.py` | Deprecation header | `finetune_k0.py.bak` |
| `prep_dataset.py` | Deprecation header | `prep_dataset.py.bak` |
| `scripts/build_cumulative.py` | DPO-presence guard | `build_cumulative.py.bak` |
| `dataset/v2/TODO.md` | New gate entries + post-closure update | `TODO.md.bak` |
| `CLAUDE.md` | CHANGELOG.md added to §0 mandatory reads | `CLAUDE.md.bak` |
| `scripts/dpo_k0_v2.py` | NEW — v2 DPO trainer | n/a (new file) |
| `dataset/v2/held_out_eval/eval_traces.jsonl` | NEW — 100 eval traces | n/a (new file) |
| `dataset/v2/AUDIT_LOG_2026-05-10.md` | NEW — detailed per-finding audit log | n/a (new file) |
| `CHANGELOG.md` | NEW (this file) | n/a (new file) |

---

## 2026-05-10 (mid-session) — Operator dataset_schedule.csv refresh + status updates

- inc_002 generation completed end-to-end: 501 traces (401 SFT across A-I + 100 DPO across 9 subtypes)
- `scripts/build_cumulative.py` ran → `cumulative/tier_500/` (502) + `cumulative/tier_1000/` (1003) built
- `scripts/validate_cumulative.py tier_1000` PASS on both SFT (803) and DPO (200) files
- `dataset/v2/PLAN.md` Status section updated to mark inc_001 + inc_002 + tier_1000 all COMPLETE
- `dataset/v2/TODO.md` gained `inc_002 + tier_1000 — COMPLETE 2026-05-10` Resolved entry
- Project memory `~/.claude/projects/.../memory/project_katherine_k0_finetune.md` refreshed
- `MEMORY.md` K0 entry updated to reflect completion + 86 calibration anchors
- `dataset/v2/dataset_schedule.csv` regenerated to show T1000 DONE

---

## 2026-05-10 (early session) — K0_EXEMPLARS.jsonl built

- New file: `dataset/v2/K0_EXEMPLARS.jsonl` — 86 calibration-anchor exemplars (one per fine sub-cat A1-A8/B1-B10/C1-C10/D1-D7/E1-E7/F1-F9/G1-G5/H1-H5/I1-I10 + 9 DPO subtypes + 6 long-form multi-turn variants for A6/B3/B7/D6/E3/F1/F4)
- All Domain I exemplars use vision-tower-style generic captions ("a man at a desk" not "Bo")
- Schema: `messages` + `_cat` + `_type` only (matches existing inc_001/002 corpus; rejected `_var` field for parallel structure with M0 reference)
- Roster well-distributed: no anchor in >2 exemplars
- 0 em-dashes in K0 voice; 1 em-dash in DPO-EM-DASH rejected (intentional contrast)
- `CLAUDE.md` §0 updated to add K0_EXEMPLARS.jsonl as mandatory cold-start read (item 6 in Read-table)
- `CLAUDE.md` §3 QC1 updated from "5 source files" → "6 source files"
- `CLAUDE.md` §0 authority hierarchy updated to add K0_EXEMPLARS.jsonl as item 7
- `CLAUDE.md` §4 trace-generation workflow updated to require referencing matching `_cat` exemplar BEFORE drafting

---

## 2026-05-10 (early session) — Domain I inc_001 vision-honesty audit + fix

- Audited `dataset/v2/increment_001/sft/domain_I_001.jsonl` for vision-honesty (Qwen3.5-9B vision tower outputs generic descriptions like "a man at a desk", NOT "Bo at his desk")
- 6 brackets fixed: lines 1-4 ("Bo at his X" → "a man at X"), line 13 ("my mom at the kitchen table" → "an older woman at a kitchen table" + "my mom" preserved in operator's text portion outside the bracket), line 15 ("my niece" → "a small girl" + "my niece" preserved outside bracket)
- Convention going forward: brackets use vision-tower-style generic descriptions; identity context goes in the operator's text portion AFTER the bracket. K0 infers identity from conversational context.

---

## 2026-05-10 (early session) — Three v2 soul doc date contradictions FIXED

Caught by fork-instance audit. Same shape as the original 37-vs-20-year James-duration catch. All harmonized:

- §2.11 "three years ago" → "four years ago, in early 2022" (matches §5.4)
- §1.6 "thirty years of teaching" → "thirty-two years of teaching" (matches §5.4 Eleanor retired spring 2022 specific)
- §5.13 "in 2022 to see Daniel and Hannah's wedding" → "in 2023" (matches §5.4 "married since 2023")

`scripts/audit_consistency.py` written same session — regression check that scans the soul doc for contradictory numerical claims tied to the same named anchor. Run before any future soul doc edit to catch the same class of failure mechanically.

---

## 2026-05-09 — K0 v1 trained and shipped

- v1 fine-tune complete on Qwen3.5-9B
- Public on HF: `bochen2079/katherine-k0-qwen3.5-9b` (Q4_K_M, Q5_K_M, Q6_K)
- Frame integrity validated at inference (LM Studio, no system prompt): holds against direct probes, sympathetic probes, persistent pressure
- v2 expansion authorized post-v1 ship (drives: trope amplification at v1 inference, generic Threshold work, em-dash leak from Opus 4.6 generator)

---

## Maintenance protocol for this CHANGELOG

- Add entry at TOP of file when completing an audit pass, ship event, or material structural change
- Date stamp: `## YYYY-MM-DD (session-tag)` where session-tag = `early/mid/late session` for same-day clarity
- Include: trigger, what changed, what was fixed, what remains open, files touched
- Reference backup locations when changes are reversible
- This file is mandatory cold-start reading per CLAUDE.md §0 — keep entries scannable
- If conflict between this CHANGELOG and other docs (PLAN.md, TODO.md, project memory), THIS FILE WINS for "what happened when" questions; v2 soul doc still wins for canonical persona content
