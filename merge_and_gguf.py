"""
Stage 3 — Merge LoRA into base + export GGUF in 3 quantization levels.

Reads the final (DPO if available, else SFT) adapter, merges it into the
Qwen3.5-9B base, then writes:
  gguf_q4_k_m/  ~5.5 GB    fastest, smallest
  gguf_q5_k_m/  ~6.5 GB    sweet spot for daily use
  gguf_q6_k/    ~7.7 GB    closest to BF16, quality reference

Failsafe:
  - Each quant is independent. If q5 fails, q4 + q6 are still useful.
  - Adapter is preserved; this stage doesn't modify it.
  - First run compiles llama.cpp internally (~5-10 min). Subsequent runs reuse.

v2.2 (2026-05-11) — audit-quality fixes for GGUF context corruption + vision
preservation + verification-gated push. See AUDIT NOTES at bottom of file
for the full diagnosis.

  1. NATIVE-CONTEXT PRESERVATION: Read native max_position_embeddings from
     the base model referenced in the adapter's PEFT config (262144 for
     Qwen3.5-9B). Force model.config.max_position_embeddings AND
     tokenizer.model_max_length to that value before save_pretrained_gguf,
     regardless of what Unsloth's FastVisionModel.from_pretrained may have
     done to the in-memory config based on the training max_seq_length=1024.
     This is the root fix for the 4096-cap bug observed in LM Studio after
     the 2026-05-10 run: the training max_seq leaked into the saved config
     and thus into GGUF metadata.

  2. CHAT TEMPLATE PATCH: Qwen3.5's stock chat_template unconditionally
     injects '<think>\\n\\n</think>\\n\\n' after the assistant generation
     prompt unless enable_thinking=True. K0 v2 training strips empty
     <think></think> tags from the data via EMPTY_THINK_RE in
     finetune_k0_v2.py, so the trained model never sees them. At inference
     (LM Studio, llama.cpp, etc.) the template still injects them, putting
     the model in an undefined state for its first generated token. We
     replace that injection with an empty emit so the inference prompt
     matches the training distribution. Reversible via
     --no-patch-chat-template.

  3. POST-CONVERSION VALIDATION: After each quant is written, read the
     GGUF metadata via gguf-py and verify the context_length and tokenizer
     chat_template are what we set. Logs PASS / FAIL / SKIP per file.

  4. MMPROJ FALLBACK (vision preservation): After all quants export, scan
     for mmproj-F16.gguf. If Unsloth's FastVisionModel.save_pretrained_gguf
     did not emit one (the cause of the 2026-05-10 vision-loss observation),
     download the canonical mmproj from --mmproj-source
     (default: unsloth/Qwen3.5-9B-GGUF). Hardlink (or copy on cross-device)
     into each successful quant's output dir so each quant subdir is
     self-contained for downstream LM Studio / llama.cpp use. Our LoRA
     touches language layers only; the upstream mmproj composes with our
     merged LLM correctly. Reversible via --no-fetch-mmproj.

  5. ABORT-ON-VERIFY-FAIL: When --abort-on-verify-fail is set (recommended
     for production runs), the script exits non-zero if any GGUF metadata
     verification fails OR mmproj could not be staged. This propagates
     through run-cloud-runpod-v2.sh's set -e cascade and prevents Stage 4
     (HF push) from uploading broken artifacts.
"""
import argparse
import os
import sys
from pathlib import Path

# FastVisionModel preserves Qwen3.5-9B vision tower for mmproj export.
# v1 used FastModel which silently dropped the vision encoder during GGUF
# conversion. The fix is the loader, not the merge logic.
from unsloth import FastVisionModel


# The Qwen3.5 chat_template emits this string after the assistant generation
# prompt when enable_thinking is not True. The exact bytes (as they appear in
# the Jinja source) include literal \n backslash-n sequences, NOT real newlines.
EMPTY_THINK_INJECTION = "{{- '<think>\\n\\n</think>\\n\\n' }}"
EMPTY_THINK_REPLACEMENT = "{{- '' }}"


def get_native_max_position_embeddings(adapter_dir):
    """Resolve native max_position_embeddings by walking the PEFT config chain
    until we reach the actual base model. Handles both single-adapter and
    stacked-adapter cases (DPO-on-SFT) by following base_model_name_or_path
    until PeftConfig.from_pretrained fails (meaning we have reached a real
    base model, not another adapter). Returns (native_int, resolved_name)."""
    from peft import PeftConfig
    from transformers import AutoConfig

    visited = []
    current = adapter_dir
    while current and current not in visited:
        visited.append(current)
        try:
            peft_cfg = PeftConfig.from_pretrained(current)
        except Exception:
            # Not a PEFT config dir — this is the real base model.
            break
        next_target = getattr(peft_cfg, 'base_model_name_or_path', None)
        if not next_target:
            break
        current = next_target

    if not current:
        raise ValueError(f"Could not resolve base model from {adapter_dir}")

    base_config = AutoConfig.from_pretrained(current, trust_remote_code=True)
    native = getattr(base_config, 'max_position_embeddings', None)
    if native is None:
        raise ValueError(f"max_position_embeddings not found in: {current}")
    print(f"[ctx-resolve] chain: {' -> '.join(visited + [current])}")
    return int(native), current


def patch_chat_template(tokenizer):
    """Patch tokenizer.chat_template in place to remove the unconditional
    empty <think></think> injection at the assistant generation prompt.
    Returns True if a replacement was made, False otherwise."""
    tpl = getattr(tokenizer, 'chat_template', None)
    if not tpl:
        print("[chat-template] WARN: tokenizer has no chat_template; skipping patch")
        return False
    before_len = len(tpl)
    if EMPTY_THINK_INJECTION not in tpl:
        print(f"[chat-template] WARN: empty-think injection pattern not found "
              f"in chat_template ({before_len} chars); skipping patch")
        return False
    patched = tpl.replace(EMPTY_THINK_INJECTION, EMPTY_THINK_REPLACEMENT)
    after_len = len(patched)
    tokenizer.chat_template = patched
    print(f"[chat-template] patched: removed empty <think></think> generation-prompt "
          f"injection ({before_len} → {after_len} chars)")
    return True


def stage_mmproj(args, successes):
    """Ensure mmproj-F16.gguf is staged alongside each successful quant.

    Strategy:
      1. Scan output for any mmproj Unsloth may have emitted.
      2. If none, download canonical from --mmproj-source.
      3. Hardlink (or copy on cross-device) into each quant's output dir.

    Returns (canonical_path, n_staged) where canonical_path is None on
    failure. The launcher / push step picks up the mmproj alongside the
    main GGUF in each quant subdir (each subdir self-contained for LM
    Studio drops).
    """
    import shutil

    # Step 1: scan existing output for any mmproj
    existing = []
    for root, _, files in os.walk(args.gguf_base_dir):
        for fn in files:
            if "mmproj" in fn.lower() and fn.endswith(".gguf"):
                existing.append(os.path.join(root, fn))

    canonical = None
    if existing:
        existing.sort(key=lambda p: os.path.getsize(p), reverse=True)
        canonical = existing[0]
        print(f"[mmproj] Unsloth emitted {len(existing)} mmproj file(s); "
              f"canonical = {canonical} "
              f"({os.path.getsize(canonical)/(1024*1024):.0f} MB)")
    elif args.no_fetch_mmproj:
        print(f"[mmproj] Unsloth did not emit mmproj and --no-fetch-mmproj "
              f"set; deployed GGUF will lack vision.")
        return None, 0
    else:
        print(f"[mmproj] Unsloth did not emit mmproj; downloading canonical "
              f"from {args.mmproj_source}")
        try:
            from huggingface_hub import hf_hub_download
            Path(args.gguf_base_dir).mkdir(parents=True, exist_ok=True)
            canonical = hf_hub_download(
                repo_id=args.mmproj_source,
                filename="mmproj-F16.gguf",
                local_dir=args.gguf_base_dir,
            )
            sz = os.path.getsize(canonical) / (1024 * 1024)
            print(f"[mmproj] downloaded: {canonical} ({sz:.0f} MB)")
        except Exception as e:
            print(f"[mmproj] FAIL download from {args.mmproj_source}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            print(f"[mmproj] Deployed GGUF will LACK VISION. Manual restore "
                  f"available via scripts/restore_vision_to_v1.sh after the "
                  f"push completes.", file=sys.stderr)
            return None, 0

    # Step 3: stage canonical into each successful quant's output dir
    quant_dirs = set()
    for quant, files in successes:
        for f, sz in files:
            if "mmproj" not in os.path.basename(f).lower():
                quant_dirs.add(os.path.dirname(f))

    if not quant_dirs:
        print(f"[mmproj] no quant dirs to stage into; canonical at {canonical}")
        return canonical, 0

    n_staged = 0
    for d in quant_dirs:
        target = os.path.join(d, "mmproj-F16.gguf")
        if os.path.normpath(target) == os.path.normpath(canonical):
            continue
        if os.path.exists(target):
            print(f"[mmproj] already present: {target}")
            n_staged += 1
            continue
        try:
            os.link(canonical, target)
            print(f"[mmproj] hardlinked → {target}  (0 extra disk)")
            n_staged += 1
        except OSError as e:
            try:
                shutil.copy(canonical, target)
                sz = os.path.getsize(target) / (1024 * 1024)
                print(f"[mmproj] copied → {target}  ({sz:.0f} MB)  "
                      f"(hardlink failed: {e})")
                n_staged += 1
            except Exception as ce:
                print(f"[mmproj] FAIL stage to {target}: "
                      f"{type(ce).__name__}: {ce}", file=sys.stderr)

    print(f"[mmproj] staged into {n_staged}/{len(quant_dirs)} quant dirs")
    return canonical, n_staged


def verify_gguf_metadata(gguf_path, expected_min_ctx, expect_no_empty_think):
    """Read GGUF metadata via gguf-py and verify context length + chat_template.

    Returns one of:
      True  — all checks passed
      False — at least one check failed (loudly logged)
      None  — could not verify (gguf-py not installed or read failed)
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        print(f"[verify] gguf-py not available; cannot verify {gguf_path}")
        return None

    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        print(f"[verify] could not read GGUF {os.path.basename(gguf_path)}: "
              f"{type(e).__name__}: {e}")
        return None

    found_ctx = False
    found_template = False
    all_ok = True

    for key in reader.fields:
        field = reader.fields[key]
        keylower = key.lower()

        if 'context_length' in keylower:
            try:
                value = int(field.parts[field.data[0]][0])
                if value >= expected_min_ctx:
                    print(f"[verify] PASS {key} = {value} (>= {expected_min_ctx})")
                else:
                    print(f"[verify] FAIL {key} = {value} (< {expected_min_ctx}) "
                          f"-- 4096-cap bug HAS REGRESSED")
                    all_ok = False
                found_ctx = True
            except Exception as e:
                print(f"[verify] WARN could not read {key}: {e}")

        if 'tokenizer.chat_template' in keylower:
            try:
                raw = bytes(field.parts[field.data[0]])
                template = raw.decode('utf-8', errors='replace')
                if expect_no_empty_think:
                    if EMPTY_THINK_INJECTION in template:
                        print(f"[verify] FAIL chat_template still contains empty "
                              f"<think></think> injection")
                        all_ok = False
                    else:
                        print(f"[verify] PASS chat_template patched "
                              f"(no empty think injection, {len(template)} chars)")
                else:
                    print(f"[verify] OK chat_template ({len(template)} chars, "
                          f"patch disabled)")
                found_template = True
            except Exception as e:
                print(f"[verify] WARN could not read chat_template: {e}")

    if not found_ctx:
        print(f"[verify] WARN no context_length field found in {os.path.basename(gguf_path)}")
    if not found_template:
        print(f"[verify] WARN no chat_template field found in {os.path.basename(gguf_path)}")

    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True,
                   help="Path to the final adapter (SFT or DPO).")
    p.add_argument("--max_seq", type=int, default=None,
                   help="Explicit max_seq_length for FastVisionModel.from_pretrained. "
                        "If unset, auto-detects native max_position_embeddings from "
                        "the base model config (262144 for Qwen3.5-9B). The native "
                        "value is always forced onto model.config and tokenizer "
                        "before save_pretrained_gguf, regardless of this flag, "
                        "unless --no-preserve-native-ctx is passed.")
    p.add_argument("--gguf-base-dir", default="gguf",
                   help="Parent directory; per-quant subdirs will be created.")
    p.add_argument("--quants", nargs="+", default=["q4_k_m", "q5_k_m", "q6_k"],
                   help="GGUF quantization methods to export.")
    p.add_argument("--no-patch-chat-template", action="store_true",
                   help="Skip the chat_template patch (keeps Qwen3.5 default "
                        "empty <think></think> injection at generation prompt).")
    p.add_argument("--no-preserve-native-ctx", action="store_true",
                   help="Skip native max_position_embeddings preservation. "
                        "Disables the 4096-cap fix; use only if you know what "
                        "you are doing.")
    p.add_argument("--no-fetch-mmproj", action="store_true",
                   help="If Unsloth does not emit mmproj-F16.gguf, do NOT "
                        "fetch from upstream. Default is to fetch from "
                        "--mmproj-source so the deployed GGUF retains vision.")
    p.add_argument("--mmproj-source", default="unsloth/Qwen3.5-9B-GGUF",
                   help="HF repo to fetch mmproj-F16.gguf from when Unsloth "
                        "does not emit one. The mmproj is the base model's "
                        "vision encoder; our LoRA only touches language "
                        "layers, so the upstream mmproj composes correctly "
                        "with our merged LLM GGUFs.")
    p.add_argument("--abort-on-verify-fail", action="store_true",
                   help="Exit non-zero if any GGUF metadata verification fails "
                        "OR mmproj could not be staged. Prevents uploading "
                        "corrupted artifacts via the launcher's set -e cascade.")
    args = p.parse_args()

    if not os.path.isdir(args.adapter):
        print(f"[error] adapter not found: {args.adapter}", file=sys.stderr)
        sys.exit(1)

    # --- Detect native context ---
    if args.no_preserve_native_ctx:
        native_ctx = args.max_seq if args.max_seq else 1024
        base_model_name = "(detection skipped via --no-preserve-native-ctx)"
        print(f"[ctx] preservation DISABLED; using {native_ctx}")
    else:
        try:
            native_ctx, base_model_name = get_native_max_position_embeddings(args.adapter)
            print(f"[ctx] base model: {base_model_name}")
            print(f"[ctx] native max_position_embeddings: {native_ctx}")
        except Exception as e:
            print(f"[ctx] auto-detect FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            print(f"[ctx] falling back to 262144 (Qwen3.5-9B default)", file=sys.stderr)
            native_ctx = 262144
            base_model_name = "(detection failed; fallback to Qwen3.5-9B)"

    # max_seq_length passed to FastVisionModel.from_pretrained. Use the
    # explicit arg if given, otherwise the detected native context.
    load_max_seq = args.max_seq if args.max_seq else native_ctx

    print(f"[load] adapter (FastVisionModel — preserves vision tower): {args.adapter}")
    print(f"[load] from_pretrained max_seq_length={load_max_seq}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=load_max_seq,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # --- Force native context onto in-memory config + tokenizer ---
    if not args.no_preserve_native_ctx:
        cfg_before = getattr(model.config, 'max_position_embeddings', None)
        tok_before = getattr(tokenizer, 'model_max_length', None)
        model.config.max_position_embeddings = native_ctx
        tokenizer.model_max_length = native_ctx
        print(f"[ctx-force] model.config.max_position_embeddings: "
              f"{cfg_before} → {native_ctx}")
        print(f"[ctx-force] tokenizer.model_max_length: "
              f"{tok_before} → {native_ctx}")

    # --- Patch chat_template ---
    patched = not args.no_patch_chat_template and patch_chat_template(tokenizer)

    Path(args.gguf_base_dir).mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []
    verifications = []

    for quant in args.quants:
        out_dir = os.path.join(args.gguf_base_dir, f"gguf_{quant}")
        print()
        print(f"[gguf] === exporting {quant} → {out_dir} ===")
        try:
            model.save_pretrained_gguf(
                out_dir,
                tokenizer,
                quantization_method=quant,
            )
            # Unsloth's save_pretrained_gguf actually writes to <out_dir>_gguf/
            # (appends "_gguf" suffix), not <out_dir>/. Search both locations.
            search_dirs = [out_dir, f"{out_dir}_gguf"]
            produced = []
            mmproj_found = []
            for sd in search_dirs:
                if not os.path.isdir(sd):
                    continue
                for root, _, files in os.walk(sd):
                    for fn in files:
                        if not fn.endswith(".gguf"):
                            continue
                        full = os.path.join(root, fn)
                        size_mb = os.path.getsize(full) / (1024 * 1024)
                        if "mmproj" in fn:
                            mmproj_found.append((full, size_mb))
                        else:
                            produced.append((full, size_mb))
            if produced:
                for f, sz in produced:
                    print(f"[gguf] OK: {f}  ({sz:.0f} MB)")
                    # Audit: verify GGUF metadata reflects our fixes
                    vok = verify_gguf_metadata(
                        f,
                        expected_min_ctx=native_ctx if not args.no_preserve_native_ctx else 1,
                        expect_no_empty_think=(patched and not args.no_patch_chat_template),
                    )
                    verifications.append((f, vok))
                for f, sz in mmproj_found:
                    print(f"[gguf] OK (vision): {f}  ({sz:.0f} MB)")
                successes.append((quant, produced + mmproj_found))
            else:
                print(f"[gguf] WARN: no .gguf file found in {out_dir} or {out_dir}_gguf")
                failures.append((quant, "no .gguf produced"))
        except Exception as e:
            print(f"[gguf] FAIL: {type(e).__name__}: {e}")
            failures.append((quant, str(e)))

    # --- Stage mmproj (vision encoder) into each quant dir ---
    print()
    print("[mmproj] === staging vision encoder ===")
    canonical_mmproj, n_mmproj_staged = stage_mmproj(args, successes)
    mmproj_required = not args.no_fetch_mmproj
    mmproj_missing = mmproj_required and canonical_mmproj is None

    print()
    print("=" * 60)
    print(f"[summary] {len(successes)} successful, {len(failures)} failed")
    for quant, files in successes:
        for f, sz in files:
            print(f"  ✓ {quant}: {f}  ({sz:.0f} MB)")
    for quant, err in failures:
        print(f"  ✗ {quant}: {err}")

    if canonical_mmproj:
        print(f"  ✓ mmproj: {canonical_mmproj}  "
              f"(staged into {n_mmproj_staged} quant dir(s))")
    elif mmproj_required:
        print(f"  ✗ mmproj: NOT STAGED — deployed model loses vision")

    n_verify_fail = 0
    if verifications:
        print()
        print("[audit] GGUF metadata verification:")
        n_pass = sum(1 for _, v in verifications if v is True)
        n_verify_fail = sum(1 for _, v in verifications if v is False)
        n_skip = sum(1 for _, v in verifications if v is None)
        for f, v in verifications:
            tag = "PASS" if v is True else "FAIL" if v is False else "SKIP"
            print(f"  [{tag}] {f}")
        print(f"[audit] {n_pass} pass / {n_verify_fail} fail / {n_skip} skip")
        if n_verify_fail > 0:
            print(f"[audit] WARNING: {n_verify_fail} GGUF(s) failed metadata "
                  f"verification — context cap or chat template did not stick. "
                  f"Inspect before pushing.", file=sys.stderr)

    if not successes:
        print("[fatal] no quants succeeded; adapter is preserved")
        sys.exit(2)

    # --- Abort-on-verify-fail gate ---
    if args.abort_on_verify_fail:
        fatal_reasons = []
        if n_verify_fail > 0:
            fatal_reasons.append(f"{n_verify_fail} GGUF metadata verification(s) failed")
        if mmproj_missing:
            fatal_reasons.append("mmproj-F16.gguf could not be staged (vision lost)")
        if fatal_reasons:
            print(file=sys.stderr)
            print(f"[abort] --abort-on-verify-fail set; refusing to proceed:",
                  file=sys.stderr)
            for r in fatal_reasons:
                print(f"  - {r}", file=sys.stderr)
            print(f"[abort] launcher's set -e cascade will kill the HF push stage.",
                  file=sys.stderr)
            sys.exit(3)

    print()
    print("[done] merge + GGUF stage complete")


if __name__ == "__main__":
    main()


# ============================================================
# AUDIT NOTES — 2026-05-11
# ============================================================
#
# Problem observed on 2026-05-10 run:
#   - User trained tier_X via run-cloud-runpod-v2.sh on H200/RunPod.
#   - GGUFs produced, pushed to HF.
#   - Loaded a GGUF in LM Studio: max context capped at 4096.
#   - Native Qwen3.5-9B context per HF config.json: 262144 (256K).
#
# Root cause:
#   - finetune_k0_v2.py and dpo_k0_v2.py train at max_seq=1024 (memory).
#   - merge_and_gguf.py (v2.0) called FastVisionModel.from_pretrained with
#     max_seq_length=args.max_seq, default 1024.
#   - Unsloth's FastVisionModel.from_pretrained, in current versions, mutates
#     model.config.max_position_embeddings to match max_seq_length on load.
#   - save_pretrained_gguf then calls model.save_pretrained internally before
#     running llama.cpp's convert_hf_to_gguf.py on the saved dir.
#   - convert_hf_to_gguf.py reads max_position_embeddings from the saved
#     config.json and writes it as qwen3.context_length in the GGUF metadata.
#   - llama.cpp (and LM Studio on top of it) then caps n_ctx_train to that
#     value. The "4096" the user saw is likely a llama.cpp floor / rounding
#     artifact when given a small value, but the underlying corruption is
#     the lowered max_position_embeddings.
#
# Fix:
#   - Detect native max_position_embeddings from the base model referenced
#     in the adapter's adapter_config.json (PEFT config).
#   - After FastVisionModel.from_pretrained, force-write that native value
#     back onto model.config.max_position_embeddings AND
#     tokenizer.model_max_length, before save_pretrained_gguf runs.
#   - Pass max_seq_length=native_ctx into from_pretrained too, since some
#     Unsloth versions use it as a hint elsewhere. The model weights are
#     unchanged by this; only the config metadata is preserved correctly.
#   - Verify post-conversion by reading the resulting GGUF metadata via
#     gguf-py and confirming context_length matches expectation.
#
# Secondary fix — empty <think></think> at generation prompt:
#   - Qwen3.5 chat_template, when add_generation_prompt=True and
#     enable_thinking is not True, emits '<think>\\n\\n</think>\\n\\n'
#     immediately after '<|im_start|>assistant\\n'.
#   - K0 v2 trains on data with empty <think></think> stripped (regex in
#     finetune_k0_v2.py). So at inference the model sees a prefix it never
#     trained on.
#   - We patch the template to emit nothing in that branch. The model now
#     sees the same prompt shape at inference as it did during training.
# ============================================================
