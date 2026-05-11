#!/usr/bin/env bash
# PreToolUse hook for K0 fine-tune project.
# Configured via .claude/settings.json. Fires before Write/Edit tool calls.
#
# Behavior: if the target path is inside dataset/v2/ OR is a canon file,
# verify the K0 cold-start sentinel is valid for the current session.
# If not, BLOCK the tool call with an explicit remediation message.
#
# The hook receives JSON on stdin describing the tool call (per Claude
# Code hooks protocol). It outputs JSON on stdout indicating allow or deny.
#
# This is the structural enforcement that makes "read canon before
# generating" a hard requirement instead of a documentation suggestion.

set -uo pipefail

# Detect python interpreter. command -v python3 returns true on Windows
# even when there's only a Microsoft Store stub that doesn't actually
# work, so we test by actually running --version and checking output.
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version_out=$("$candidate" --version 2>&1 || true)
        # Real python prints "Python 3.x.y"; Microsoft Store stub prints
        # "Python was not found...". Match digit after "Python ".
        if [[ "$version_out" =~ ^Python\ [0-9] ]]; then
            PY="$candidate"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo '{"decision":"block","reason":"K0 hook FATAL: no working python interpreter. Install python3 or python and ensure it is in PATH (not the Microsoft Store stub)."}'
    exit 0
fi

# Read the tool call JSON from stdin
INPUT=$(cat)

# Extract the file path the tool wants to write to
# (Both Write and Edit tools have file_path field in tool_input)
FILE_PATH=$(echo "$INPUT" | "$PY" -c "
import json, sys
try:
    data = json.load(sys.stdin)
    tool_input = data.get('tool_input', {})
    print(tool_input.get('file_path', ''))
except Exception:
    print('')
")

# Paths that require canon to be loaded before writing
PROTECTED_PATTERNS=(
    "dataset/v2/"
    "k0_soul_document"
    "data/trace_generation_prompt.md"
    "dataset/v2/K0_EXEMPLARS.jsonl"
    "dataset/v2/PLAN.md"
    "CLAUDE.md"
)

is_protected=0
for pattern in "${PROTECTED_PATTERNS[@]}"; do
    if [[ "$FILE_PATH" == *"$pattern"* ]]; then
        is_protected=1
        break
    fi
done

if [ "$is_protected" = "0" ]; then
    # Not a protected path, allow
    echo '{"continue": true}'
    exit 0
fi

# Protected path. Check sentinel.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERIFY_OUTPUT=$(cd "$PROJECT_ROOT" && "$PY" scripts/k0_cold_start.py --verify 2>&1)
VERIFY_RC=$?

if [ "$VERIFY_RC" = "0" ]; then
    # Sentinel valid, allow the write
    echo '{"continue": true}'
    exit 0
fi

# Sentinel invalid. Block the tool call with a clear remediation message.
# The Claude Code hook protocol uses a "decision: block" + "reason" field
# to deny tool execution and surface a system-reminder to the model.
"$PY" -c "
import json
print(json.dumps({
    'decision': 'block',
    'reason': '''K0 COLD-START SENTINEL INVALID. Write to '$FILE_PATH' BLOCKED.

Reason: $VERIFY_OUTPUT

This path is canon-protected. Before writing to dataset/v2/** or any K0
canon file, you must:

  1. Read C:\\\\katherine-k0-finetune\\\\k0_soul_document.md IN FULL (200KB)
  2. Read CLAUDE.md, PLAN.md, TODO.md, trace_generation_prompt.md,
     K0_EXEMPLARS.jsonl, CHANGELOG.md (the 7 other CLAUDE.md §0 files)
  3. Run: python scripts/k0_cold_start.py
  4. Then retry the write.

The cold-start script does not verify YOU read the canon. It verifies
you ran the script. The reading is on you. But the bootloader catches
the case where you skipped both.

This block exists because the inc_004 generation incident on 2026-05-10
cost 2466 contaminated traces from skipping cold-start. Operator made
the structural enforcement explicit. Do not bypass.'''
}))
"
exit 0
