#!/usr/bin/env bash
# restore_vision_to_v1.sh: Path A fix for the existing K0 v1 deployment.
#
# The K0 v1 fine-tune used Unsloth's FastModel loader, which doesn't preserve
# the vision tower into the merged GGUF. The Unsloth-published vanilla
# mmproj-F16.gguf for Qwen3.5-9B is unmodified by our LoRA (we only touched
# language layers), so it composes with our K0 LLM GGUF and restores vision.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash scripts/restore_vision_to_v1.sh
#
# Result: bochen2079/katherine-k0-qwen3.5-9b gets mmproj-F16.gguf added,
# and LM Studio users can drop images again.

set -euo pipefail

REPO="bochen2079/katherine-k0-qwen3.5-9b"
SOURCE_REPO="unsloth/Qwen3.5-9B-GGUF"
MMPROJ_FILE="mmproj-F16.gguf"
WORK_DIR="${WORK_DIR:-/tmp/k0_vision_fix}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "[1/3] Downloading $MMPROJ_FILE from $SOURCE_REPO"
hf download "$SOURCE_REPO" "$MMPROJ_FILE" --local-dir . --local-dir-use-symlinks False

echo "[2/3] Uploading $MMPROJ_FILE to $REPO"
hf upload "$REPO" "$MMPROJ_FILE" "$MMPROJ_FILE" --repo-type model

echo "[3/3] Done. Verify at https://huggingface.co/$REPO/blob/main/$MMPROJ_FILE"
echo
echo "LM Studio users now load both:"
echo "  Qwen3.5-9B.Q5_K_M.gguf  (the K0 LLM)"
echo "  $MMPROJ_FILE             (the vision tower, native Qwen3.5-9B)"
echo
echo "Vision and OCR should now work."
