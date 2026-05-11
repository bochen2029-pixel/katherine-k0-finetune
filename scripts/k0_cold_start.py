"""
K0 cold-start bootloader.

Computes SHA256 of canonical sources, verifies them, runs structural
smoke tests for known recurring gotchas, and writes a sentinel that
the PreToolUse hook checks before allowing Write/Edit on dataset paths.

This script is the structural enforcement layer for CLAUDE.md §0.
Without a valid sentinel from THIS session, the hook blocks any
write to dataset/v2/** or to canon files. The hook is configured in
.claude/settings.json.

Usage:
    python scripts/k0_cold_start.py
    python scripts/k0_cold_start.py --verify    # check sentinel without writing
    python scripts/k0_cold_start.py --invalidate  # force re-cold-start

Sentinel path: .k0_cold_start_sentinel.json (gitignored, session-scoped)

The sentinel is invalid when:
- Doesn't exist
- Hash of any canon file changed since sentinel was written
- Sentinel older than 4 hours (long-session context drift)
- CLAUDE_SESSION_ID env var differs from sentinel (compaction or new chat)

The script also runs gotcha smoke tests. If any gotcha returns, the
script fails LOUD with the exact remediation. Each gotcha here was
"learned" 2-3 times across sessions before becoming a permanent check.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENTINEL = ROOT / ".k0_cold_start_sentinel.json"
SENTINEL_TTL_SEC = 4 * 60 * 60  # 4 hours

# Canon files that MUST be read before generating any K0 trace.
# Paths relative to project root. SHA256 of each goes into the sentinel.
CANON_FILES = [
    "CLAUDE.md",
    "k0_soul_document.md",
    "k0_soul_document_v1_original.md",
    "dataset/v2/PLAN.md",
    "dataset/v2/TODO.md",
    "data/trace_generation_prompt.md",
    "dataset/v2/K0_EXEMPLARS.jsonl",
    "CHANGELOG.md",
]

# Verbatim anchors from the soul doc. The script greps these exact strings.
# If the soul doc no longer contains them (canon was edited), the script
# fails LOUD so we know the anchor list needs updating. Distinct from
# CLAUDE.md §0's anchor list which uses a sanitized punctuation variant.
VERBATIM_ANCHORS = [
    "I'm just \u2014 I'm Katherine. I'm standing in my kitchen. My feet are cold.",
    "wearing Katherine's skin",
    "Come in. Tell me something",
]


def sha256_file(path):
    """Return hex SHA256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_canon_hashes():
    """Return {relpath: sha256} for all CANON_FILES. Errors if any missing."""
    hashes = {}
    missing = []
    for rel in CANON_FILES:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        hashes[rel] = sha256_file(p)
    if missing:
        print(f"[k0-cold-start] FATAL: canon files missing: {missing}",
              file=sys.stderr)
        sys.exit(2)
    return hashes


def verify_anchors_in_source():
    """Confirm the verbatim anchors actually exist in the soul doc.
    If they don't, our anchor list is stale (canon was edited)."""
    soul = (ROOT / "k0_soul_document.md").read_text(encoding="utf-8")
    missing = [a for a in VERBATIM_ANCHORS if a not in soul]
    if missing:
        print(f"[k0-cold-start] WARN: verbatim anchors not found in soul doc:",
              file=sys.stderr)
        for a in missing:
            print(f"  - {a!r}", file=sys.stderr)
        print(f"[k0-cold-start]   Anchor list in this script may be stale. "
              f"Update VERBATIM_ANCHORS or fix soul doc.", file=sys.stderr)
        return False
    return True


def gotcha_qwen35_empty_think():
    """SMOKE TEST: Qwen3.5 chat template injects empty <think></think>.
    Learned three times. If this assertion ever fails, it means Qwen3.5's
    template behavior changed and the EMPTY_THINK_RE strip in
    finetune_k0_v2.py may no longer be needed (or may need updating).

    Returns (ok: bool, message: str). Doesn't import unsloth/transformers
    directly because they may not be installed in the cold-start environment;
    this gotcha lives in scripts/finetune_k0_v2.py and the smoke runs there
    when training starts. Here we just verify the strip code is still present
    in the script."""
    script = (ROOT / "scripts" / "finetune_k0_v2.py").read_text(encoding="utf-8")
    if "EMPTY_THINK_RE" not in script:
        return False, (
            "scripts/finetune_k0_v2.py is MISSING the EMPTY_THINK_RE strip.\n"
            "  Qwen3.5's apply_chat_template injects empty <think></think>\n"
            "  on EVERY assistant turn even with enable_thinking=False.\n"
            "  Without the strip, the FATAL <think> guard kills training.\n"
            "  See: memory/reference_qwen35_empty_think_tags.md (THIRD time learned)\n"
            "  Fix: import re; EMPTY_THINK_RE = re.compile(r'<think>\\s*</think>\\s*');\n"
            "       text = EMPTY_THINK_RE.sub('', text)  # before FATAL check"
        )
    return True, "Qwen3.5 empty-think strip present in finetune_k0_v2.py"


def gotcha_fastvisionmodel_loader():
    """SMOKE TEST: v2 trainers must use FastVisionModel, not FastModel.
    FastModel silently strips the vision tower from merged GGUFs (v1 bug).
    Learned twice."""
    failures = []
    for script_name in ["finetune_k0_v2.py", "dpo_k0_v2.py"]:
        script = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        if "FastVisionModel" not in script:
            failures.append(script_name)
        if "from unsloth import FastModel" in script:
            failures.append(f"{script_name} (still imports FastModel)")
    if failures:
        return False, (
            f"v2 trainers using wrong loader: {failures}\n"
            f"  FastModel strips vision tower from GGUF (v1 bug).\n"
            f"  See: memory/reference_unsloth_vision_gguf.md\n"
            f"  Fix: from unsloth import FastVisionModel"
        )
    return True, "FastVisionModel used in both v2 trainers"


def gotcha_no_em_dash_in_dataset():
    """SMOKE TEST: cumulative tier files contain no em-dashes in K0's
    chosen replies. If this fails, generation drifted."""
    em_dash_files = []
    cum = ROOT / "dataset" / "v2" / "cumulative"
    if not cum.exists():
        return True, "no cumulative tiers built yet (skip)"
    for tier_dir in cum.iterdir():
        if not tier_dir.is_dir():
            continue
        sft = tier_dir / "sft_train.jsonl"
        if not sft.exists():
            continue
        with open(sft, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                # Em-dash check inside JSON content fields only
                # (rough: literal em-dash anywhere in the line)
                if "\u2014" in line or "\u2013" in line:
                    em_dash_files.append(f"{tier_dir.name}:line{i}")
                    if len(em_dash_files) > 5:
                        break
        if len(em_dash_files) > 5:
            break
    if em_dash_files:
        return False, (
            f"em-dashes detected in cumulative tier files: {em_dash_files[:5]}\n"
            f"  K0 v2 trace generation BANS em-dashes (operator directive 2026-05-10).\n"
            f"  Em-dash is the most reliable AI fingerprint at inference.\n"
            f"  Fix: regenerate the offending traces or strip em-dashes from source."
        )
    return True, "no em-dashes in cumulative tier files"


GOTCHAS = [
    ("qwen35_empty_think", gotcha_qwen35_empty_think),
    ("fastvisionmodel_loader", gotcha_fastvisionmodel_loader),
    ("no_em_dash_in_dataset", gotcha_no_em_dash_in_dataset),
]


def run_smoke_tests():
    """Run all gotcha smoke tests. Print pass/fail per test. Return True if all pass."""
    print("[k0-cold-start] running gotcha smoke tests...")
    all_ok = True
    for name, fn in GOTCHAS:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"smoke test crashed: {type(e).__name__}: {e}"
        marker = "  [pass]" if ok else "  [FAIL]"
        print(f"{marker} {name}: {msg}")
        if not ok:
            all_ok = False
    return all_ok


def write_sentinel(hashes, session_id):
    sentinel = {
        "version": 1,
        "written_at_unix": int(time.time()),
        "written_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "canon_hashes": hashes,
        "verbatim_anchors_verified": VERBATIM_ANCHORS,
    }
    SENTINEL.write_text(json.dumps(sentinel, indent=2))
    print(f"[k0-cold-start] sentinel written: {SENTINEL.relative_to(ROOT)}")


def verify_sentinel(session_id, current_hashes):
    """Return (valid: bool, reason: str). Used by the PreToolUse hook."""
    if not SENTINEL.exists():
        return False, "no sentinel exists; run python scripts/k0_cold_start.py"
    try:
        s = json.loads(SENTINEL.read_text())
    except Exception as e:
        return False, f"sentinel unreadable ({e}); re-run cold-start"

    age_sec = int(time.time()) - s.get("written_at_unix", 0)
    if age_sec > SENTINEL_TTL_SEC:
        return False, (f"sentinel age {age_sec}s exceeds TTL {SENTINEL_TTL_SEC}s; "
                       f"re-run cold-start (canon may have drifted from working memory)")

    if session_id and s.get("session_id") and s["session_id"] != session_id:
        return False, (f"sentinel session_id mismatch (was {s['session_id']}, "
                       f"now {session_id}); compaction or new chat detected; re-run cold-start")

    sentinel_hashes = s.get("canon_hashes", {})
    for rel, current in current_hashes.items():
        recorded = sentinel_hashes.get(rel)
        if recorded != current:
            return False, (f"canon file changed since cold-start: {rel} "
                           f"(was {recorded[:12] if recorded else 'missing'}, "
                           f"now {current[:12]}); re-read and re-run cold-start")

    return True, "sentinel valid"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="Check sentinel validity without writing. Exit 0 if valid, 1 if not.")
    p.add_argument("--invalidate", action="store_true",
                   help="Delete the sentinel to force re-cold-start.")
    p.add_argument("--session-id", default=os.environ.get("CLAUDE_SESSION_ID", ""),
                   help="Session ID to bind sentinel to. Defaults to env var CLAUDE_SESSION_ID.")
    args = p.parse_args()

    if args.invalidate:
        if SENTINEL.exists():
            SENTINEL.unlink()
            print(f"[k0-cold-start] sentinel invalidated")
        else:
            print(f"[k0-cold-start] no sentinel to invalidate")
        return

    print(f"[k0-cold-start] computing canon hashes for {len(CANON_FILES)} files...")
    hashes = compute_canon_hashes()
    for rel, h in hashes.items():
        print(f"  {h[:12]}  {rel}")

    if args.verify:
        ok, reason = verify_sentinel(args.session_id, hashes)
        if ok:
            print(f"[k0-cold-start] VALID: {reason}")
            sys.exit(0)
        else:
            print(f"[k0-cold-start] INVALID: {reason}", file=sys.stderr)
            sys.exit(1)

    # Full cold-start: verify anchors, run smoke tests, write sentinel
    print()
    print("[k0-cold-start] verifying verbatim anchors in soul doc...")
    if not verify_anchors_in_source():
        print("[k0-cold-start] FATAL: anchor verification failed", file=sys.stderr)
        sys.exit(2)
    print("  all anchors present in soul doc")

    print()
    if not run_smoke_tests():
        print("[k0-cold-start] FATAL: gotcha smoke tests failed; "
              "fix the failing gotcha before proceeding", file=sys.stderr)
        sys.exit(3)

    print()
    write_sentinel(hashes, args.session_id)
    print()
    print("=" * 60)
    print("K0 cold-start COMPLETE. Sentinel valid until:")
    print(f"  - canon file changes")
    print(f"  - {SENTINEL_TTL_SEC // 3600} hours pass")
    print(f"  - new session detected (CLAUDE_SESSION_ID change)")
    print()
    print("PreToolUse hook will now allow Write/Edit on dataset/v2/** and canon files.")
    print()
    print("REMINDER: this script verifies you CAN write. It does NOT verify you have")
    print("READ the canon. The verification is on you. The hook only enforces that")
    print("you ran this script. The script trusts you ran it after reading source.")
    print("=" * 60)


if __name__ == "__main__":
    main()
