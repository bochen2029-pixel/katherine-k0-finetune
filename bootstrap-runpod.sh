#!/usr/bin/env bash
# One-shot bootstrap for fresh RunPod / Lambda Labs / similar Linux GPU pod.
#
# Usage on the pod:
#   curl -sSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh | bash
#
# Or after manual clone:
#   cd katherine-k0-finetune && ./bootstrap-runpod.sh
#
# Sets up: clone repo, install Python deps (unsloth + trl + transformers + hf),
# verify CUDA, optional HF auth, leave you ready to run ./run-cloud-runpod.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bochen2029-pixel/katherine-k0-finetune.git}"
REPO_DIR="${REPO_DIR:-$HOME/katherine-k0-finetune}"

echo "============================================================"
echo "Katherine k0 fine-tune — bootstrap"
echo "============================================================"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo

# -----------------------------------------------------------------------
# Detect privilege level. RunPod's pytorch image runs as root; Hyperbolic
# (and most user-friendly cloud GPU pods) run as a normal user with
# passwordless sudo. Pick the right invocation pattern for apt + pip.
# -----------------------------------------------------------------------
SUDO=""
PIP_USER_FLAG=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
    PIP_USER_FLAG="--user"
fi
echo "Privilege: $(id -un) (uid=$(id -u)); apt prefix='${SUDO:-(none)}'; pip flag='${PIP_USER_FLAG:-(none)}'"
echo

# 1. CUDA toolkit check
echo "[1/6] Verifying CUDA toolkit..."
if ! command -v nvcc >/dev/null; then
    echo "  nvcc not in PATH; trying /usr/local/cuda/bin"
    if [ -d /usr/local/cuda/bin ]; then
        export PATH=/usr/local/cuda/bin:$PATH
    fi
fi
if command -v nvcc >/dev/null; then
    nvcc --version | grep release
else
    echo "  WARN: nvcc not found. Unsloth doesn't strictly need it, but llama.cpp"
    echo "        compilation for GGUF export may. Continue at your own risk."
fi

echo
echo "[2/6] Detecting GPUs..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Pre-install apt packages that Unsloth's save_pretrained_gguf() needs for
# llama.cpp compilation. Unsloth otherwise tries to install these
# interactively at GGUF time, which fails under the watchdog (closed stdin).
echo
echo "[2b/6] Installing apt packages for llama.cpp / GGUF export..."
$SUDO apt-get update -qq 2>&1 | tail -2
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
    cmake libssl-dev libcurl4-openssl-dev build-essential \
    2>&1 | tail -3

# 3. Clone repo
echo
echo "[3/6] Cloning repo..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  repo already at $REPO_DIR; pulling latest"
    cd "$REPO_DIR"
    # The bootstrap chmod +x's tracked scripts further down. With Linux's
    # default core.filemode=true, git records the mode change as a local
    # modification and `git pull --ff-only` refuses to merge. Disabling
    # core.filemode tells git to ignore mode bits — chmod no longer
    # creates phantom modifications.
    git config core.filemode false
    # If the ff-only pull still trips on a real content drift, force-align
    # to origin/master. Rented instances are reproducible-from-clean, so
    # nuking any uncommitted local state is the correct behavior.
    if ! git pull --ff-only 2>&1; then
        echo "  WARN: ff-only pull failed; force-aligning to origin/master"
        git fetch origin master
        git reset --hard origin/master
    fi
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git config core.filemode false
fi
echo "[3/6] In: $(pwd)"

# 4. Install Python deps
echo
echo "[4/6] Installing Python dependencies..."
echo "  (this can take 5-10 min on first run — unsloth pulls a lot)"

# Use the system Python (RunPod's pytorch image has Python 3.11 with pip)
PY=python3
if ! command -v $PY >/dev/null; then PY=python; fi

# Probe for --break-system-packages (pip 23.0+; required by PEP 668 on Ubuntu
# 23.04+ images; harmless if also passed on older pip — except that older pip
# refuses unknown flags. So we test before adding it).
PIP_BREAK_FLAG=""
if $PY -m pip install --break-system-packages --dry-run pip >/dev/null 2>&1; then
    PIP_BREAK_FLAG="--break-system-packages"
fi

PIP_INSTALL="$PY -m pip install --quiet $PIP_USER_FLAG $PIP_BREAK_FLAG"
echo "  pip install pattern: $PIP_INSTALL"

$PIP_INSTALL --upgrade pip

# Pin torch + torchvision + torchaudio to a cu121 wheel set so the install
# is compatible with the kernel-side NVIDIA driver that Hyperbolic.xyz pods
# typically ship (535.183.06 as of 2026-05-11).
#
# Driver-CUDA support matrix (Linux):
#   driver 535.x → CUDA 12.2 native, 12.3-12.4 via cuda-compat libs
#   driver 545+  → CUDA 12.3 native
#   driver 550+  → CUDA 12.4 native
#   driver 570+  → CUDA 12.8 native
#
# torch wheel CUDA tags:
#   cu118 → CUDA 11.8 (any 5XX driver)
#   cu121 → CUDA 12.1 (driver 525+) ← THIS ROW for driver 535
#   cu124 → CUDA 12.4 (driver 545+ native, or 535 + cuda-compat)
#   cu128 → CUDA 12.8 (driver 570+ native) ← FAILS on 535 at first kernel launch
#
# torch 2.5.1 is the last release shipping cu121 wheels — torch 2.6+ moved
# default builds to cu124. unsloth-zoo's pin (torch<2.11, >=2.4) is satisfied.
$PIP_INSTALL \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1"

# unsloth + unsloth_zoo must come from the SAME source (git main) to keep
# their internal TRL pins consistent. Their current main requires:
#   trl>=0.18.2,!=0.19.0,<=0.24.0
#   unsloth_zoo>=2026.4.8
# If we let pip pull unsloth_zoo from PyPI, it may resolve to an older
# release whose TRL pin (trl<0.14) clashes with unsloth main's TRL pin
# (trl>=0.18.2). pip then reports an unsatisfiable conflict.
#
# --upgrade (without --upgrade-strategy=eager) refreshes the explicitly
# named packages (unsloth, unsloth_zoo) from git main but does NOT cascade
# upgrades into their dependencies. Important: if --upgrade-strategy=eager
# were set, pip would see unsloth-zoo's pin (torch<2.11,>=2.4) and bump
# the just-installed torch 2.5.1+cu121 up to the latest version satisfying
# that range — likely 2.10.x with cu128 wheels — re-introducing the
# driver-535 incompatibility we just fixed above.
$PIP_INSTALL --upgrade \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git" \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "transformers>=4.50.0" \
    "trl>=0.18.2,!=0.19.0,<=0.24.0" \
    "peft>=0.12.0" \
    "bitsandbytes>=0.43.0" \
    "accelerate>=1.0.0" \
    "datasets>=2.20.0" \
    "huggingface_hub>=0.27.0" \
    "sentencepiece" \
    "protobuf" \
    "xformers" \
    "gguf>=0.10.0"

# When pip --user installs binaries (hf CLI, etc.), they land in ~/.local/bin.
# Make sure that's on PATH within this shell so the post-install verification
# and `hf auth login` calls below find them. The launcher (run-h100-hyperbolic.sh)
# repeats this export so the subsequent v2 launcher invocation also sees them.
if [ -n "$PIP_USER_FLAG" ] && [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify import
$PY -c "import unsloth; print(f'  unsloth: {unsloth.__version__}')"
$PY -c "import transformers; print(f'  transformers: {transformers.__version__}')"
$PY -c "import trl; print(f'  trl: {trl.__version__}')"
$PY -c "import peft; print(f'  peft: {peft.__version__}')"

echo
echo "[5/6] HF CLI + auth..."
if command -v hf >/dev/null; then
    HF_VER=$(hf --version 2>&1 | head -1 || echo "unknown")
    echo "  hf CLI: $HF_VER"
else
    echo "  WARN: hf command not in PATH after install; trying huggingface-cli fallback"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    if hf auth login --token "$HF_TOKEN" >/dev/null 2>&1; then
        echo "  HF logged in as $(hf auth whoami 2>&1 | grep user: | awk '{print $2}')"
    else
        echo "  WARN: HF login failed; HF sync will skip at run-time"
    fi
elif [ -f "$HOME/.hf_token" ]; then
    HF_TOKEN=$(cat "$HOME/.hf_token")
    export HF_TOKEN
    hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
    echo "  HF token loaded from \$HOME/.hf_token"
else
    echo "  HF_TOKEN not set; HF sync will be disabled at run-time"
    echo "  To enable: export HF_TOKEN=<your_token> before running ./run-cloud-runpod.sh"
fi

# 6. Verify dataset
echo
echo "[6/6] Verifying canonical datasets..."
chmod +x \
    run-cloud-runpod.sh \
    run-cloud-runpod-v2.sh \
    run-h100-hyperbolic.sh \
    _supervise-cloud.sh \
    bootstrap-runpod.sh \
    2>/dev/null || true

if [ -f data/k0_canonical.jsonl ]; then
    SFT_LINES=$(wc -l < data/k0_canonical.jsonl)
    echo "  ✓ data/k0_canonical.jsonl ($SFT_LINES SFT examples)"
else
    echo "  WARN: data/k0_canonical.jsonl missing; rebuild with prep_dataset.py if you have raw sources"
fi
if [ -f data/k0_dpo_curated.jsonl ]; then
    DPO_LINES=$(wc -l < data/k0_dpo_curated.jsonl)
    echo "  ✓ data/k0_dpo_curated.jsonl ($DPO_LINES DPO pairs)"
else
    echo "  (no DPO data; DPO stage will skip)"
fi

echo
echo "============================================================"
echo "Bootstrap complete."
echo
echo "To launch the full pipeline:"
echo "  cd $REPO_DIR"
echo "  export HF_TOKEN=<your_token>     # if not already set"
echo "  ./run-cloud-runpod.sh"
echo
echo "Stages: SFT → DPO → merge+GGUF (3 quants) → push to HF bucket"
echo "Total wallclock: ~50-70 min on H200, ~75-90 min on H100"
echo "============================================================"
