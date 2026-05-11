#!/usr/bin/env bash
# Katherine k0 v2 fine-tune orchestrator — tier-aware, vision-aware.
#
# Differs from run-cloud-runpod.sh (v1):
#   - Uses scripts/finetune_k0_v2.py (FastVisionModel — preserves vision tower)
#   - Uses scripts/dpo_k0_v2.py (FastVisionModel + v2 DPO schema)
#   - Reads from dataset/v2/cumulative/tier_<TIER>/{sft,dpo}_train.jsonl
#   - Default tier = 5000 (3936 SFT + 1011 DPO from inc_001-004)
#   - Default HF bucket = bochen2079/katherine-k0-v2-t<TIER>
#   - Wallclock cap = 4 hours (v2 t5000 is ~3-4x v1 wallclock)
#
# Usage on the pod:
#   curl -sSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh | bash
#   cd ~/katherine-k0-finetune
#   export HF_TOKEN=<your_token>
#   ./run-cloud-runpod-v2.sh
#
# Override tier:
#   TIER=2500 ./run-cloud-runpod-v2.sh
#
# Vanilla (text-only) training:
#   VANILLA=1 ./run-cloud-runpod-v2.sh
#
# Skip stages (resume support):
#   SKIP_SFT=1 ./run-cloud-runpod-v2.sh   # use existing SFT adapter, run DPO+GGUF+push
#   SKIP_DPO=1 ./run-cloud-runpod-v2.sh   # SFT+GGUF+push, no DPO
#   SKIP_GGUF=1 SKIP_PUSH=1 ./run-cloud-runpod-v2.sh   # just train

set -uo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
# Default tier = 2500 (CLEAN: inc_001/002/003 from prior careful sessions).
# tier_5000 includes inc_004 which was generated 2026-05-10 by a Claude
# instance that BYPASSED the cold-start protocol. Audit found ~10 hard
# canon contradictions in E/F/G domains (Mose color/origin/age, Eleanor's
# health condition, Sam pronoun gender, Marcus dates, Lila husband/store/
# friendship-start, Aaron's kids, Naomi profession, James condition, plus
# persistent invented characters Margarita/Lin/Jonas). Train tier_5000
# only if you accept those contradictions baking into the model:
#   TIER=5000 ./run-cloud-runpod-v2.sh
TIER="${TIER:-2500}"
VANILLA="${VANILLA:-0}"

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3.5-9B}"

# Tier-derived defaults
if [ "$VANILLA" = "1" ]; then
    TIER_DIR_SUFFIX="_vanilla"
    BUCKET_SUFFIX="-vanilla"
    VANILLA_FLAG="--vanilla"
else
    TIER_DIR_SUFFIX=""
    BUCKET_SUFFIX=""
    VANILLA_FLAG=""
fi

DATA_SFT="${DATA_SFT:-dataset/v2/cumulative/tier_${TIER}${TIER_DIR_SUFFIX}/sft_train.jsonl}"
DATA_DPO="${DATA_DPO:-dataset/v2/cumulative/tier_${TIER}${TIER_DIR_SUFFIX}/dpo_train.jsonl}"

SFT_ADAPTER="${SFT_ADAPTER:-adapters/k0_v2_sft_t${TIER}${TIER_DIR_SUFFIX}}"
DPO_ADAPTER="${DPO_ADAPTER:-adapters/k0_v2_dpo_t${TIER}${TIER_DIR_SUFFIX}}"
GGUF_BASE_DIR="${GGUF_BASE_DIR:-gguf_v2_t${TIER}${TIER_DIR_SUFFIX}}"

# SFT hyperparameters (same as v1; soul_document validated this on v1 trace count)
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
HF_BUCKET="${HF_BUCKET:-bochen2079/katherine-k0-v2-t${TIER}${BUCKET_SUFFIX}}"
HF_SYNC_ENABLED="${HF_SYNC_ENABLED:-1}"

# Stage skip flags
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_DPO="${SKIP_DPO:-0}"
SKIP_GGUF="${SKIP_GGUF:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

# Wallclock cap (hard kill if total run exceeds)
WALLCLOCK_HARD_CAP="${WALLCLOCK_HARD_CAP:-14400}"   # 4 hours
SIGUSR1_LEAD="${SIGUSR1_LEAD:-300}"

OUTPUT_BASE="${OUTPUT_BASE:-katherine_k0_v2_t${TIER}${TIER_DIR_SUFFIX}}"

# ---------------------------------------------------------------------------
# Pre-flight banner
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Katherine K0 v2 fine-tune  —  tier ${TIER}${TIER_DIR_SUFFIX}"
echo "============================================================"
echo "  SFT data:  $DATA_SFT"
echo "  DPO data:  $DATA_DPO"
echo "  SFT out:   $SFT_ADAPTER"
echo "  DPO out:   $DPO_ADAPTER"
echo "  GGUF out:  $GGUF_BASE_DIR"
echo "  HF bucket: $HF_BUCKET"
echo "  Vanilla:   $VANILLA"
echo "  Hard cap:  $((WALLCLOCK_HARD_CAP / 60)) minutes"
echo

# ---------------------------------------------------------------------------
# Pre-flight checks
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

echo "[gpu] $GPU_COUNT × $GPU_NAME (tier=$GPU_TIER, ${GPU_VRAM_MB} MB)"

if [ "$GPU_TIER" = "4090" ]; then
    echo "[gpu] WARN: 4090 has 24GB; QLoRA Qwen3.5-9B at batch 16 will OOM."
    echo "[gpu] Reduce: SFT_BATCH=2 SFT_GRAD_ACCUM=16 SFT_MAX_SEQ=1024"
fi

# Dataset checks
if [ ! -f "$DATA_SFT" ]; then
    echo "ERROR: SFT dataset missing: $DATA_SFT" >&2
    echo "       Verify dataset/v2/cumulative/tier_${TIER}${TIER_DIR_SUFFIX}/ exists." >&2
    echo "       Build with: python scripts/build_cumulative.py [--vanilla]" >&2
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

# v2 script presence check
for s in scripts/finetune_k0_v2.py scripts/dpo_k0_v2.py merge_and_gguf.py push_to_hf.py; do
    if [ ! -f "$s" ]; then
        echo "ERROR: required script missing: $s" >&2
        exit 1
    fi
done

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
            echo '[stage 1] v2 SFT (FastVisionModel — preserves vision tower)'
            python scripts/finetune_k0_v2.py \\
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
                --grad_accum $SFT_GRAD_ACCUM \\
                $VANILLA_FLAG
        else
            echo '[stage 1] SKIPPED'
        fi

        if [ \"$SKIP_DPO\" = \"0\" ] && [ -f \"$DATA_DPO\" ]; then
            echo
            echo '[stage 2] v2 DPO'
            python scripts/dpo_k0_v2.py \\
                --data \"$DATA_DPO\" \\
                --sft-adapter \"$SFT_ADAPTER\" \\
                --output \"$DPO_ADAPTER\" \\
                --max_seq $SFT_MAX_SEQ \\
                --epochs $DPO_EPOCHS \\
                --lr $DPO_LR \\
                --beta $DPO_BETA \\
                --batch $DPO_BATCH \\
                --grad_accum $DPO_GRAD_ACCUM \\
                $VANILLA_FLAG
            FINAL_ADAPTER=\"$DPO_ADAPTER\"
        else
            echo '[stage 2] SKIPPED (no DPO data or SKIP_DPO set)'
            FINAL_ADAPTER=\"$SFT_ADAPTER\"
        fi

        if [ \"$SKIP_GGUF\" = \"0\" ]; then
            echo
            echo '[stage 3] merge + GGUF (3 quants, vision tower preserved)'
            python merge_and_gguf.py \\
                --adapter \"\$FINAL_ADAPTER\" \\
                --gguf-base-dir \"$GGUF_BASE_DIR\" \\
                --quants $GGUF_QUANTS \\
                --abort-on-verify-fail
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
                --data-dir dataset/v2/cumulative/tier_${TIER}${TIER_DIR_SUFFIX}
        else
            echo '[stage 4] SKIPPED (no HF sync or SKIP_PUSH set)'
        fi

        echo
        echo '[orchestrator] all v2 stages complete'
    "
