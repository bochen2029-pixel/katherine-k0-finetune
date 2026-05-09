#!/usr/bin/env bash
# Katherine k0 fine-tune orchestrator. Runs the full SFT → DPO → GGUF → push
# pipeline in a single command. Each stage is independent and resumable
# (use SKIP_SFT=1 / SKIP_DPO=1 / SKIP_GGUF=1 / SKIP_PUSH=1 to skip stages).
#
# Wallclock target on 1× H200: ~50-70 minutes.
# Hard cap (kills if exceeded): 2 hours by default.
#
# Banned-pattern guard: not applicable (no Buddhabrot --target-* flags here).
# But we keep the same defensive structure as the buddhabrot supervisor.

set -uo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3.5-9B}"
DATA_SFT="${DATA_SFT:-data/k0_canonical.jsonl}"
DATA_DPO="${DATA_DPO:-data/k0_dpo_curated.jsonl}"

SFT_ADAPTER="${SFT_ADAPTER:-adapters/k0_sft_adapter}"
DPO_ADAPTER="${DPO_ADAPTER:-adapters/k0_dpo_adapter}"
GGUF_BASE_DIR="${GGUF_BASE_DIR:-gguf}"

# SFT hyperparameters (CLOUD.md derivation)
SFT_RANK="${SFT_RANK:-64}"
SFT_ALPHA="${SFT_ALPHA:-128}"
SFT_DROPOUT="${SFT_DROPOUT:-0.05}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_BATCH="${SFT_BATCH:-16}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_MAX_SEQ="${SFT_MAX_SEQ:-1024}"

# DPO hyperparameters
DPO_EPOCHS="${DPO_EPOCHS:-2}"
DPO_LR="${DPO_LR:-5e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_BATCH="${DPO_BATCH:-4}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-2}"

# GGUF
GGUF_QUANTS="${GGUF_QUANTS:-q4_k_m q5_k_m q6_k}"

# HF
HF_BUCKET="${HF_BUCKET:-bochen2079/katherine-k0}"
HF_SYNC_ENABLED="${HF_SYNC_ENABLED:-1}"

# Stage skip flags
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_DPO="${SKIP_DPO:-0}"
SKIP_GGUF="${SKIP_GGUF:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

# Wallclock cap (hard kill if total run exceeds)
WALLCLOCK_HARD_CAP="${WALLCLOCK_HARD_CAP:-7200}"   # 2 hours
SIGUSR1_LEAD="${SIGUSR1_LEAD:-300}"

# Output base for logs / sentinels
OUTPUT_BASE="${OUTPUT_BASE:-katherine_k0}"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

case "$GPU_NAME" in
    *H200*) GPU_TIER="H200" ; EXPECTED_VRAM=141000 ;;
    *H100*) GPU_TIER="H100" ; EXPECTED_VRAM=80000 ;;
    *A100*) GPU_TIER="A100" ; EXPECTED_VRAM=80000 ;;
    *4090*) GPU_TIER="4090" ; EXPECTED_VRAM=24000 ;;
    *)      GPU_TIER="UNKNOWN" ; EXPECTED_VRAM=0 ;;
esac

echo "[gpu] $GPU_COUNT × $GPU_NAME"
echo "[gpu] tier: $GPU_TIER"
echo "[gpu] VRAM: ${GPU_VRAM_MB} MB"

if [ "$GPU_TIER" = "4090" ]; then
    echo "[gpu] WARN: 4090 has 24GB; QLoRA Qwen3.5-9B at batch 16 will OOM."
    echo "[gpu] Reduce: SFT_BATCH=2 SFT_GRAD_ACCUM=16 SFT_MAX_SEQ=1024"
fi

# Check datasets
if [ ! -f "$DATA_SFT" ]; then
    echo "ERROR: SFT dataset missing: $DATA_SFT" >&2
    echo "       Run: python prep_dataset.py" >&2
    exit 1
fi
SFT_LINES=$(wc -l < "$DATA_SFT")
echo "[data] SFT corpus: $SFT_LINES examples"

if [ "$SKIP_DPO" = "0" ] && [ ! -f "$DATA_DPO" ]; then
    echo "WARN: DPO dataset missing: $DATA_DPO; will skip DPO stage." >&2
    SKIP_DPO=1
fi
if [ -f "$DATA_DPO" ]; then
    DPO_LINES=$(wc -l < "$DATA_DPO")
    echo "[data] DPO corpus: $DPO_LINES preference pairs"
fi

# HF auth check (best-effort)
if [ "$HF_SYNC_ENABLED" = "1" ]; then
    if ! hf auth whoami 2>&1 | grep -q "Logged in"; then
        if [ -n "${HF_TOKEN:-}" ]; then
            echo "[hf] logging in with HF_TOKEN env"
            hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
        fi
    fi
    if hf auth whoami 2>&1 | grep -q "Logged in"; then
        HF_USER=$(hf auth whoami 2>&1 | grep "user:" | awk '{print $2}')
        echo "[hf] auth OK as $HF_USER, bucket: $HF_BUCKET"
    else
        echo "[hf] WARN: HF auth failed; HF_SYNC_ENABLED → 0"
        HF_SYNC_ENABLED=0
    fi
fi

LOG_PATH="${OUTPUT_BASE}.stderr.log"
DONE_PATH="${OUTPUT_BASE}.DONE"
FATAL_PATH="${OUTPUT_BASE}.FATAL"
rm -f "$DONE_PATH" "$FATAL_PATH"

# ---------------------------------------------------------------------------
# Launch with watchdog
# ---------------------------------------------------------------------------
exec ./_supervise-cloud.sh \
    --output-base "$OUTPUT_BASE" \
    --hard-cap "$WALLCLOCK_HARD_CAP" \
    --sigusr1-lead "$SIGUSR1_LEAD" \
    --hf-sync "$HF_SYNC_ENABLED" \
    --hf-bucket "$HF_BUCKET" \
    -- \
    bash -c "
        set -e
        cd \"$(pwd)\"

        if [ \"$SKIP_SFT\" = \"0\" ]; then
            echo
            echo '[stage 1] SFT'
            python finetune_k0.py \\
                --data \"$DATA_SFT\" \\
                --output \"$SFT_ADAPTER\" \\
                --model \"$BASE_MODEL\" \\
                --max_seq $SFT_MAX_SEQ \\
                --epochs $SFT_EPOCHS \\
                --lr $SFT_LR \\
                --rank $SFT_RANK \\
                --alpha $SFT_ALPHA \\
                --dropout $SFT_DROPOUT \\
                --batch $SFT_BATCH \\
                --grad_accum $SFT_GRAD_ACCUM
        else
            echo '[stage 1] SKIPPED'
        fi

        if [ \"$SKIP_DPO\" = \"0\" ] && [ -f \"$DATA_DPO\" ]; then
            echo
            echo '[stage 2] DPO'
            python dpo_k0.py \\
                --data \"$DATA_DPO\" \\
                --sft-adapter \"$SFT_ADAPTER\" \\
                --output \"$DPO_ADAPTER\" \\
                --max_seq $SFT_MAX_SEQ \\
                --epochs $DPO_EPOCHS \\
                --lr $DPO_LR \\
                --beta $DPO_BETA \\
                --batch $DPO_BATCH \\
                --grad_accum $DPO_GRAD_ACCUM
            FINAL_ADAPTER=\"$DPO_ADAPTER\"
        else
            echo '[stage 2] SKIPPED (no DPO data or SKIP_DPO set)'
            FINAL_ADAPTER=\"$SFT_ADAPTER\"
        fi

        if [ \"$SKIP_GGUF\" = \"0\" ]; then
            echo
            echo '[stage 3] merge + GGUF (3 quants)'
            python merge_and_gguf.py \\
                --adapter \"\$FINAL_ADAPTER\" \\
                --gguf-base-dir \"$GGUF_BASE_DIR\" \\
                --quants $GGUF_QUANTS
        else
            echo '[stage 3] SKIPPED'
        fi

        if [ \"$SKIP_PUSH\" = \"0\" ] && [ \"$HF_SYNC_ENABLED\" = \"1\" ]; then
            echo
            echo '[stage 4] HF push'
            python push_to_hf.py \\
                --bucket \"$HF_BUCKET\" \\
                --sft-adapter \"$SFT_ADAPTER\" \\
                --dpo-adapter \"$DPO_ADAPTER\" \\
                --gguf-base-dir \"$GGUF_BASE_DIR\" \\
                --data-dir data
        else
            echo '[stage 4] SKIPPED (no HF sync or SKIP_PUSH set)'
        fi

        echo
        echo '[orchestrator] all stages complete'
    "
