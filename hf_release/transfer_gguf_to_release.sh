#!/usr/bin/env bash
# Transfer GGUFs from your private bucket → public model repo.
#
# Source: hf://buckets/bochen2079/katherine-k0/gguf/{q4_k_m,q5_k_m,q6_k}/
# Destination: bochen2079/katherine-k0-qwen3.5-9b (model repo, public)
#
# Run on whichever machine has fastest internet. Total transfer ~36 GB
# (download from bucket + upload to model repo). On a $0.04/hr CPU pod with
# 1 Gbps it's <5 min. On home internet it's 30-60 min depending on bandwidth.

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

# Verify we're logged in
if ! hf auth whoami 2>&1 | grep -q "Logged in"; then
    hf auth login --token "$HF_TOKEN"
fi

WORK_DIR="${WORK_DIR:-./katherine_release_transfer}"
mkdir -p "$WORK_DIR"

BUCKET="bochen2079/katherine-k0"
MODEL="bochen2079/katherine-k0-qwen3.5-9b"

echo "=== Step 1: Download GGUFs from bucket → local ==="
for quant in q4_k_m q5_k_m q6_k; do
    Q_UPPER=$(echo "$quant" | tr '[:lower:]' '[:upper:]')
    LOCAL_DIR="$WORK_DIR/$quant"
    mkdir -p "$LOCAL_DIR"

    # The bucket file is named like Qwen3.5-9B.Q4_K_M.gguf
    REMOTE_FILE="gguf/gguf_${quant}/Qwen3.5-9B.${Q_UPPER}.gguf"
    LOCAL_FILE="$LOCAL_DIR/Qwen3.5-9B.${Q_UPPER}.gguf"

    if [ -f "$LOCAL_FILE" ]; then
        echo "  [$quant] already downloaded; skipping"
        continue
    fi

    echo "  [$quant] downloading $REMOTE_FILE..."
    hf buckets cp "hf://buckets/$BUCKET/$REMOTE_FILE" "$LOCAL_FILE" \
        || (echo "    download failed; trying alt sync"; \
            hf sync "hf://buckets/$BUCKET/gguf/gguf_${quant}/" "$LOCAL_DIR/" --include "*.${Q_UPPER}.gguf")
done

echo
echo "=== Step 2: Upload GGUFs → model repo ==="
for quant in q4_k_m q5_k_m q6_k; do
    Q_UPPER=$(echo "$quant" | tr '[:lower:]' '[:upper:]')
    LOCAL_FILE="$WORK_DIR/$quant/Qwen3.5-9B.${Q_UPPER}.gguf"

    if [ ! -f "$LOCAL_FILE" ]; then
        echo "  [$quant] local file missing; skipping"
        continue
    fi

    echo "  [$quant] uploading $LOCAL_FILE..."
    hf upload "$MODEL" "$LOCAL_FILE" "Qwen3.5-9B.${Q_UPPER}.gguf" \
        --repo-type model \
        --commit-message "Add ${Q_UPPER} GGUF" \
        || echo "    [$quant] upload FAILED"
done

echo
echo "=== Done ==="
echo "Model repo: https://huggingface.co/$MODEL"
echo
echo "Optional cleanup:"
echo "  rm -rf $WORK_DIR"
