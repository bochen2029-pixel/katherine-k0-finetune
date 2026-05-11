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
# TRAIN-HARDER hyperparameters (defaults below):
#   This launcher sets harder hyperparameters than the v2 launcher defaults
#   to maximize H100 utilization for K0 persona consolidation. Rationale:
#     - K0 v1 (rank 64, 3 epochs) showed trope amplification — UNDER-trained.
#     - v2 dataset is cleaner (exemplar-anchored, em-dash banned, trope-rotated).
#     - H100 80GB has 5× headroom over the conservative batch=16 default.
#     - 2000-sample QLoRA at rank=128 stays out of overfit zone with lr=5e-5.
#   Each value is env-overridable (e.g., SFT_EPOCHS=4 ... bash <(curl ...)).
#
#   SFT_RANK=128       (was 64)    LoRA rank; capacity for dense persona.
#   SFT_ALPHA=256      (was 128)   2× rank, standard convention.
#   SFT_EPOCHS=6       (was 3)     more passes; save_strategy="epoch" keeps all.
#   SFT_LR=5e-5        (was 1e-4)  halved to offset more epochs + higher rank.
#   SFT_BATCH=32       (was 16)    H100 80GB handles it; cleaner gradients.
#   SFT_GRAD_ACCUM=2   (was 2)     effective batch 64.
#   DPO_EPOCHS=4       (was 2)     DPO benefits from more passes.
#   DPO_BATCH=8        (was 4)     H100 headroom; effective DPO batch 16.
#   DPO_GRAD_ACCUM=2   (was 2)
#
# Required env:
#   HF_TOKEN          your HF write-scope token (hf_*)
#
# Optional env (with sensible defaults):
#   TIER              dataset tier (default 2500; will train tier_2500_vanilla)
#   VANILLA           1 for vanilla (text-only), 0 for full (default 1)
#   HF_BUCKET         HF bucket destination (default derived from TIER)
#   SKIP_SFT/DPO/GGUF/PUSH  set 1 to skip a stage
#   WALLCLOCK_HARD_CAP  hard kill in seconds (default 21600 = 6 hours)
#   Any of the SFT_/DPO_ vars above to override train-harder defaults
#
# Wallclock target: ~90-120 min on 1× H100 80GB
# Cost target: ~$5-8 (Hyperbolic H100 pricing varies by tier)
#
# End artifacts: 6 SFT epoch checkpoints + 4 DPO epoch checkpoints all saved
# to disk. The final DPO adapter (last epoch) merges into the GGUFs that get
# pushed. To test other epoch checkpoints in LM Studio after the run, they
# live at adapters/k0_v2_sft_t2500_vanilla/checkpoint-* and
# adapters/k0_v2_dpo_t2500_vanilla/checkpoint-*.

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

# ----- TRAIN-HARDER hyperparameter overrides -----
# These export defaults that run-cloud-runpod-v2.sh picks up via its
# ${VAR:-fallback} pattern. If the user already set any of these in env
# before invoking this launcher, that value wins. See file header for
# rationale and original defaults.
export SFT_RANK="${SFT_RANK:-128}"
export SFT_ALPHA="${SFT_ALPHA:-256}"
export SFT_EPOCHS="${SFT_EPOCHS:-6}"
export SFT_LR="${SFT_LR:-5e-5}"
export SFT_BATCH="${SFT_BATCH:-32}"
export SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
export DPO_EPOCHS="${DPO_EPOCHS:-4}"
export DPO_BATCH="${DPO_BATCH:-8}"
export DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-2}"

# 6-hour hard cap (was 4h) to accommodate the harder training schedule.
export WALLCLOCK_HARD_CAP="${WALLCLOCK_HARD_CAP:-21600}"

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
echo "  Train-harder hyperparameters:"
echo "    SFT_RANK / ALPHA:   $SFT_RANK / $SFT_ALPHA"
echo "    SFT_EPOCHS:         $SFT_EPOCHS"
echo "    SFT_LR:             $SFT_LR"
echo "    SFT effective batch: $((SFT_BATCH * SFT_GRAD_ACCUM))  ($SFT_BATCH × $SFT_GRAD_ACCUM)"
echo "    DPO_EPOCHS:         $DPO_EPOCHS"
echo "    DPO effective batch: $((DPO_BATCH * DPO_GRAD_ACCUM))  ($DPO_BATCH × $DPO_GRAD_ACCUM)"
echo "    Wallclock cap:      $((WALLCLOCK_HARD_CAP / 60)) minutes"
echo

if command -v nvidia-smi >/dev/null; then
    echo "  GPU(s):"
    if GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1); then
        echo "$GPU_INFO" | sed 's/^/    /'
    else
        # nvidia-smi failed; surface the raw error and diagnose common patterns.
        echo "    nvidia-smi reported:" >&2
        echo "$GPU_INFO" | sed 's/^/      /' >&2
        if echo "$GPU_INFO" | grep -qE "Driver/library version mismatch|NVML library version"; then
            cat >&2 <<'NVML_DIAG'

  ============================================================
  NVIDIA driver/library version mismatch detected.
  ============================================================
  This is a Hyperbolic.xyz host configuration issue, not a
  problem with the training scripts. The training cannot start
  until nvidia-smi works on this instance.

  Step 1: Find out what kernel the host actually runs:
    cat /proc/driver/nvidia/version

  Step 2: Try, in order of cost (lowest first):

    A) Reload kernel modules (works on bare-metal, not on most
       container/VM setups but cheap to try):
         sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
         sudo modprobe nvidia
         sudo modprobe nvidia_uvm
         nvidia-smi

    B) Install userspace utils matching the host kernel module.
       Suppose Step 1 said "Kernel Module 550.127.05" — then:
         sudo apt-get update
         sudo apt-get install -y nvidia-utils-550
         nvidia-smi
       Pick the major version (e.g. 535, 550, 560) matching the
       kernel module.

    C) Terminate this Hyperbolic instance and provision a fresh
       one. Hyperbolic's Ubuntu images sometimes ship with stale
       nvidia userspace vs the running host kernel. A fresh pod
       often arrives with matching versions.

  Once nvidia-smi prints a GPU table successfully, re-run the
  single-line command from your shell. No code changes needed.
  ============================================================
NVML_DIAG
            exit 2
        fi
        echo "    (nvidia-smi failed with an unrecognized error)" >&2
        exit 1
    fi
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

# When non-root, bootstrap installed pip packages with --user, which puts
# binaries in ~/.local/bin (notably the `hf` CLI used by push_to_hf.py and
# the watchdog HF sync). bootstrap's PATH export was in its own subshell;
# re-establish it here so run-cloud-runpod-v2.sh inherits it.
if [ "$(id -u)" -ne 0 ] && [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
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
