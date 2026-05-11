#!/usr/bin/env bash
# run-h100-hyperbolic.sh
#
# Single-file end-to-end Katherine K0 v2 fine-tune entrypoint, sized for
# 1× H100 80GB on Hyperbolic.xyz.
#
# Single-line invocation (paste into the Hyperbolic shell):
#   HF_TOKEN=hf_xxx bash <(curl -fsSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/run-h100-hyperbolic.sh)
#
# What it does, end to end:
#   1. Bootstraps the env (clones the repo to $HOME/katherine-k0-finetune,
#      installs pinned torch 2.10.0 + unsloth + trl + peft + bitsandbytes,
#      apt-installs llama.cpp build deps, logs in to HF with HF_TOKEN).
#   2. Runs run-cloud-runpod-v2.sh with TIER=2500 VANILLA=1, which:
#      - Stage 1 SFT on dataset/v2/cumulative/tier_2500_vanilla/sft_train.jsonl
#        (1773 examples; FastVisionModel-based; preserves vision tower).
#      - Stage 2 DPO on dataset/v2/cumulative/tier_2500_vanilla/dpo_train.jsonl
#        (435 pairs; chained on top of the SFT adapter).
#      - Stage 3 merge + GGUF (q4_k_m / q5_k_m / q6_k) via merge_and_gguf.py.
#        This script's audit-quality fixes apply automatically:
#          * Native max_position_embeddings (262144 for Qwen3.5-9B) is
#            forced onto the saved config — fixes the 4096-cap bug from
#            the 2026-05-10 run.
#          * Qwen3.5 chat_template's empty <think></think> generation-prompt
#            injection is patched out — matches training distribution.
#          * GGUF metadata is verified post-conversion (context_length +
#            chat_template), with PASS/FAIL/SKIP logged.
#      - Stage 4 push to HF bucket bochen2079/katherine-k0-v2-t2500-vanilla.
#
# Resume / skip:
#   If you need to skip a stage (e.g., SFT already done from a previous
#   attempt), set SKIP_SFT=1, SKIP_DPO=1, SKIP_GGUF=1, or SKIP_PUSH=1.
#
# Required env:
#   HF_TOKEN          your HF write-scope token (hf_*)
#
# Optional env (with sensible defaults):
#   TIER              dataset tier (default 2500; will train tier_2500_vanilla)
#   VANILLA           1 for vanilla (text-only), 0 for full (default 1)
#   HF_BUCKET         HF bucket destination (default derived from TIER)
#   SKIP_SFT/DPO/GGUF/PUSH  set 1 to skip a stage
#   SFT_BATCH         training batch size (default 16; H100 80GB handles it)
#   WALLCLOCK_HARD_CAP  hard kill in seconds (default 14400 = 4 hours)
#
# Wallclock target: ~75-110 min on 1× H100 80GB
# Cost target: ~$3-7 (Hyperbolic H100 pricing varies by tier)

set -euo pipefail

# -------------------------------------------------------------------
# 1. Sanity-check HF_TOKEN before doing anything expensive
# -------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var required." >&2
    echo "" >&2
    echo "  Usage:" >&2
    echo "    HF_TOKEN=hf_xxx bash <(curl -fsSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/run-h100-hyperbolic.sh)" >&2
    echo "" >&2
    echo "  Get a write-scope token at: https://huggingface.co/settings/tokens" >&2
    exit 1
fi
export HF_TOKEN

# Force the configuration this launcher is sized for.
export TIER="${TIER:-2500}"
export VANILLA="${VANILLA:-1}"

# -------------------------------------------------------------------
# 2. Environment fingerprint (audit log)
# -------------------------------------------------------------------
echo "============================================================"
echo "Katherine K0 v2 fine-tune — H100 / Hyperbolic.xyz entrypoint"
echo "============================================================"
echo "  Date (UTC):  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Host:        $(hostname 2>/dev/null || echo unknown)"
echo "  User:        $(whoami 2>/dev/null || echo unknown)"
echo "  HOME:        ${HOME:-(unset)}"
echo "  PWD:         $(pwd)"
echo "  TIER:        $TIER"
echo "  VANILLA:     $VANILLA"
echo

if command -v nvidia-smi >/dev/null; then
    echo "  GPU(s):"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
else
    echo "  WARN: nvidia-smi not on PATH; bootstrap will re-check." >&2
fi
echo

# -------------------------------------------------------------------
# 3. Bootstrap (clone repo, install deps, HF auth)
# -------------------------------------------------------------------
echo "============================================================"
echo "[1/2] Bootstrapping environment via bootstrap-runpod.sh"
echo "============================================================"
curl -fsSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh | bash

REPO_DIR="${HOME}/katherine-k0-finetune"
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: bootstrap did not create $REPO_DIR" >&2
    exit 1
fi

cd "$REPO_DIR"

# Audit: confirm dataset is present (the bootstrap clone includes it).
DATA_SFT="dataset/v2/cumulative/tier_${TIER}"
if [ "$VANILLA" = "1" ]; then DATA_SFT="${DATA_SFT}_vanilla"; fi
DATA_SFT="${DATA_SFT}/sft_train.jsonl"
DATA_DPO="$(dirname "$DATA_SFT")/dpo_train.jsonl"

if [ ! -f "$DATA_SFT" ]; then
    echo "ERROR: SFT dataset missing after clone: $DATA_SFT" >&2
    echo "       Repo state may have diverged from expectations." >&2
    exit 1
fi
if [ ! -f "$DATA_DPO" ]; then
    echo "ERROR: DPO dataset missing after clone: $DATA_DPO" >&2
    exit 1
fi

echo
echo "  Dataset (SFT): $DATA_SFT ($(wc -l < "$DATA_SFT") lines)"
echo "  Dataset (DPO): $DATA_DPO ($(wc -l < "$DATA_DPO") lines)"
echo

# -------------------------------------------------------------------
# 4. Launch the v2 orchestrator (SFT → DPO → merge+GGUF → push)
# -------------------------------------------------------------------
echo "============================================================"
echo "[2/2] Launching run-cloud-runpod-v2.sh (TIER=$TIER VANILLA=$VANILLA)"
echo "============================================================"

exec ./run-cloud-runpod-v2.sh
