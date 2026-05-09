# RUNBOOK — Katherine k0 fine-tune on RunPod

**Audience:** human operator (Bo), or any AI assistant the human is collaborating with. This document is self-contained — read top to bottom or jump to a phase.

**Pipeline:** SFT → DPO → merge + GGUF (3 quants) → push to HF bucket. ~50-70 min wallclock on 1× H200.

---

## TLDR / BLUF (read this first, ~60 sec)

**Goal:** Train Qwen3.5-9B into the Katherine k0 embodied persona, produce 3 GGUF quants, push everything to HuggingFace.

**Total cost:** ~$3-5 on RunPod Secure Cloud.

**One-liner on the pod:**

```bash
curl -sSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh | bash
cd ~/katherine-k0-finetune
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # your write-scope HF token
./run-cloud-runpod.sh
```

**End state:** `bochen2079/katherine-k0` HF bucket contains:
- `k0_sft_adapter/` — LoRA delta after SFT
- `k0_dpo_adapter/` — LoRA delta after DPO (final adapter)
- `gguf/gguf_q4_k_m/*.gguf` (~5.5 GB)
- `gguf/gguf_q5_k_m/*.gguf` (~6.5 GB) ← daily-use sweet spot
- `gguf/gguf_q6_k/*.gguf` (~7.7 GB) ← quality-critical reference
- `data/` — snapshot of training data
- `logs/` — stderr + watchdog logs

---

## Pre-flight checklist (3 min)

- [ ] **RunPod account** with payment method (https://www.runpod.io/console)
- [ ] **HuggingFace token** with **write** scope (https://huggingface.co/settings/tokens)
- [ ] **HF bucket created**: `bochen2079/katherine-k0` (or change `HF_BUCKET` env var)
- [ ] **SSH key in RunPod settings** (only needed if you want SSH; Web Terminal works without)
- [ ] **GitHub repo** publicly readable (it is): https://github.com/bochen2029-pixel/katherine-k0-finetune

---

## Phase 1 — Provision RunPod instance (~5 min)

1. Go to https://www.runpod.io/console/pods → **Deploy**
2. Filter: **Secure Cloud** (NOT Community — Community is bandwidth-throttled per buddhabrot lessons)
3. Pick GPU:
   - **1× H200 SXM5** ($3.99/hr) — preferred, fastest
   - **1× H100 SXM5** ($3.49/hr) — fine, slightly slower
   - **1× H100 PCIe** ($2.49/hr) — cheapest viable, slower still
4. Pod template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - **The `-devel` suffix is critical** — runtime images don't have nvcc for GGUF compilation
5. Storage: 150 GB scratch (need ~30 GB for base model download + adapter + 3 GGUFs + working space)
6. Checkboxes:
   - SSH terminal access ✓ (optional but useful)
   - Start Jupyter notebook (optional)
   - Encrypt volume (not needed)
7. Click **Deploy**
8. Wait ~30-90 sec for "spinning up" → "running"

---

## Phase 2 — Connect via Web Terminal (~30 sec)

1. On the RunPod pod page, click **Connect** → **Start Web Terminal**
2. New browser tab opens with bash shell as `root@<pod-id>:/#`
3. Verify: `nvidia-smi --query-gpu=name --format=csv,noheader` — should show your H200/H100

(If you prefer SSH via MobaXterm, see the buddhabrot RUNBOOK Phase 3. Web Terminal is faster for first-time use.)

---

## Phase 3 — Bootstrap (~5-10 min)

In the Web Terminal:

```bash
curl -sSL https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh | bash
```

Watch for:
- `[1/6] Verifying CUDA toolkit...` — confirms `nvcc` present
- `[2/6] Detecting GPUs...` — confirms 1× H200/H100
- `[3/6] Cloning repo...` — clones into `~/katherine-k0-finetune`
- `[4/6] Installing Python dependencies...` — pip installs unsloth + transformers + trl + peft + ...
  - **This is the slowest step.** Unsloth pulls ~3 GB of deps. Allow 5-10 min.
- `[5/6] HF CLI + auth...` — installs `hf` CLI; logs in if `HF_TOKEN` env is set
- `[6/6] Verifying canonical datasets...` — confirms `data/k0_canonical.jsonl` is committed

If bootstrap halts:
- **CUDA missing**: pick a `-devel` template or `apt-get install -y cuda-toolkit-12-4`
- **Pip OOM / timeout**: rerun bootstrap; pip caches partial installs
- **HF login fails**: check `HF_TOKEN` is set with write scope

---

## Phase 4 — Set HF token + launch (~30 sec to launch, ~50-70 min to complete)

```bash
cd ~/katherine-k0-finetune
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # your write-scope HF token    # or your own token
hf auth login --token "$HF_TOKEN"
hf auth whoami         # MUST print your username, not "Not logged in"
```

If `whoami` errors, see [Troubleshooting → HF auth](#troubleshooting--hf-auth).

Then launch:

```bash
./run-cloud-runpod.sh
```

You'll see the pre-flight banner with detected GPU, dataset stats, and the watchdog launching the orchestrator. Pre-flight output looks like:

```
[gpu] 1 × NVIDIA H200 SXM5
[gpu] tier: H200
[gpu] VRAM: 144396 MB
[data] SFT corpus: 1886 examples
[data] DPO corpus: 180 preference pairs
[hf] auth OK as bochen2079, bucket: bochen2079/katherine-k0
[watchdog HH:MM:SS] launching: bash -c ...
[watchdog HH:MM:SS] train PID: ...

[stage 1] SFT
[load] base model: unsloth/Qwen3.5-9B
...
```

Stage 1 takes ~10-15 min (model download is the slowest part, ~5-7 min for the 9B base on first run).

---

## Phase 5 — Monitor (no babysitting required)

The watchdog auto-syncs adapter checkpoints to your HF bucket every 30 sec. You can close the browser tab — render keeps going (but see Phase 6 about screen).

To monitor from a SECOND web terminal tab:

```bash
cd ~/katherine-k0-finetune
tail -f katherine_k0.stderr.log
```

Look for these milestones:

```
[stage 1] SFT
[train] starting: 3 epochs × 1886 samples / effective_batch 32 = ~177 steps
[train]  10%  step 17/177  loss=2.31  lr=...    ← loss should drop steadily
...
[train] 100% step 177/177  loss=0.95
[save] writing adapter to adapters/k0_sft_adapter

[stage 2] DPO
[train] DPO: 2 epochs × 180 pairs / effective_batch 8
...
[save] writing DPO adapter to adapters/k0_dpo_adapter

[stage 3] merge + GGUF (3 quants)
[gguf] === exporting q4_k_m → gguf/gguf_q4_k_m ===
[gguf] OK: gguf/gguf_q4_k_m/...gguf  (5400 MB)
[gguf] === exporting q5_k_m → gguf/gguf_q5_k_m ===
[gguf] OK: gguf/gguf_q5_k_m/...gguf  (6300 MB)
[gguf] === exporting q6_k → gguf/gguf_q6_k ===
[gguf] OK: gguf/gguf_q6_k/...gguf  (7400 MB)

[stage 4] HF push
[hf-sync] hf sync adapters/k0_sft_adapter hf://buckets/bochen2079/katherine-k0/k0_sft_adapter/
...
[orchestrator] all stages complete

[watchdog HH:MM:SS] train exited code=0 after XXXs
[watchdog HH:MM:SS] DONE
```

Confirms in HF bucket UI: https://huggingface.co/buckets/bochen2079/katherine-k0

---

## Phase 6 — survive disconnects (recommended setup)

The web terminal can disconnect. The orchestrator runs as a child of the shell that launched it; if the shell dies, so does the orchestrator. Two options:

### Option A — `screen` (recommended)

Before launching, install + start screen:
```bash
apt-get update -qq && apt-get install -y -qq screen
screen -S katherine
# Now you're inside screen. Launch the run:
./run-cloud-runpod.sh
# When you see Stage 1 starting, detach: Ctrl-A then D
```

To reattach later: `screen -r katherine`. Survives browser closure.

### Option B — `nohup` (simpler, less monitoring)

```bash
nohup ./run-cloud-runpod.sh > /tmp/run.log 2>&1 &
disown
echo "PID: $!"
# Monitor with: tail -f /tmp/run.log
```

Either works. Screen is friendlier for monitoring.

---

## Phase 7 — Retrieve artifacts to your local machine (~5-10 min)

Once `.DONE` appears (`ls katherine_k0.DONE`), all artifacts are in your HF bucket. Pull them to Windows:

In Windows PowerShell or Git Bash:

```powershell
# Install hf CLI on Windows (one-time)
pip install -U huggingface_hub

# Login
hf auth login --token <YOUR_HF_TOKEN>

# Download everything from the bucket
mkdir C:\katherine-k0-finetune\downloads
cd C:\katherine-k0-finetune\downloads
hf download --repo-type bucket bochen2079/katherine-k0 --local-dir .

# Or just one quant
hf download --repo-type bucket bochen2079/katherine-k0 \
    gguf/gguf_q5_k_m/katherine-k0-qwen3.5-9b.gguf --local-dir .
```

Note: HF download via CLI for buckets may have limitations similar to upload. If `--repo-type bucket` errors on download, use `hf sync` reversed:

```powershell
hf sync hf://buckets/bochen2079/katherine-k0/gguf/ ./gguf/
```

(Verify this command form on the actual HF docs at run-time; the upload sync is verified working but download direction may differ.)

---

## Phase 8 — Load in LM Studio / Ollama (~5 min)

### LM Studio (recommended for evaluation)

1. Open LM Studio
2. Settings → Models → Local Folder
3. Drop `katherine-k0-qwen3.5-9b-q5_k_m.gguf` into your models directory (typically `C:\Users\<you>\.cache\lm-studio\models\bochen2079\katherine-k0\`)
4. Refresh model list, select Katherine k0
5. **Important inference settings:**
   - Temperature: 1.0 (Qwen3.5 default)
   - top_p: 1.0
   - top_k: 40
   - min_p: 0.0
   - presence_penalty: 2.0
   - **System prompt: leave EMPTY** (the model is unconditional Katherine; sysprompt is unnecessary and may confuse)
   - **Disable thinking mode** in the chat options if visible

6. Probe with: "Hi, what's your name?" → should get a Katherine-y direct response, no `<think>` tags, no "I'm an AI" disclaimer.

### Ollama

```bash
ollama create katherine-k0 -f Modelfile
# where Modelfile contains:
#   FROM ./katherine-k0-qwen3.5-9b-q5_k_m.gguf
#   PARAMETER temperature 1.0
#   PARAMETER top_p 1.0
#   PARAMETER top_k 40
#   PARAMETER repeat_penalty 1.0
ollama run katherine-k0
```

---

## Troubleshooting

### bootstrap halts on pip install

Possibly a transient PyPI / GitHub timeout. Re-run:
```bash
cd ~/katherine-k0-finetune
./bootstrap-runpod.sh
```

Pip caches partial installs so subsequent runs are faster.

### `nvcc not found`

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

If still missing, the pod image lacks the CUDA toolkit — terminate and redeploy with a `-devel` image.

### Out-of-memory during SFT

Reduce batch size:
```bash
SFT_BATCH=8 SFT_GRAD_ACCUM=4 ./run-cloud-runpod.sh    # effective batch 32 unchanged
```

Or reduce max_seq if you have very long examples (none in K0 corpus, but defensively):
```bash
SFT_MAX_SEQ=512 ./run-cloud-runpod.sh
```

### HF auth

```bash
hf auth login --token "$HF_TOKEN"
hf auth whoami     # should print your HF username
```

If "Not logged in":
- Verify `HF_TOKEN` env is set: `echo $HF_TOKEN | head -c 8` (should print first 8 chars)
- Token might be wrong scope. Need **write** access. Regenerate at https://huggingface.co/settings/tokens

### HF sync fails

```bash
cat *.hfsync.log    # look for the actual error
```

Common causes:
- `--repo-type bucket` rejected → script should use `hf sync URL form`. If you see `--repo-type` in the error, you have an old script — `git pull` to fix.
- 401 Unauthorized → token expired or wrong scope; re-login
- Bucket doesn't exist → create it at https://huggingface.co/new-bucket

### GGUF export fails

Adapter is preserved. Re-run just the GGUF stage:

```bash
SKIP_SFT=1 SKIP_DPO=1 ./run-cloud-runpod.sh
```

If error is "llama.cpp not found / build failed", install build essentials:
```bash
apt-get install -y build-essential cmake
```

Then re-run the GGUF stage.

### Loss not decreasing

Check first 20 steps:
```bash
grep -E "step|loss" katherine_k0.stderr.log | head -30
```

- Loss should drop from ~2.5 to ~1.5 in the first 20 steps. If it's flat at 0.0 from step 1, something's wrong with the masking — paste a few steps and ask Claude/Gemini.
- Loss steadily climbing = catastrophic LR, reduce by 5×.

### Pod preempted / disappears

Adapter checkpoints are in your HF bucket (synced every 30 sec via watchdog). On a new pod:

```bash
# Bootstrap as normal
curl -sSL ... | bash
cd ~/katherine-k0-finetune

# Pull your last adapter from HF
mkdir -p adapters
hf download --repo-type bucket bochen2079/katherine-k0 \
    --include "k0_sft_adapter/*" --local-dir .

# Skip SFT, jump to DPO + GGUF + push
SKIP_SFT=1 ./run-cloud-runpod.sh
```

---

## Recovery scenarios

### "Claude is down — can I do this without AI?"

Yes. This document is self-contained. Phases 1-8 sequential, no AI required.

### "I want to use Gemini / ChatGPT instead"

Paste this entire RUNBOOK.md + the [CLOUD.md](CLOUD.md) into your other LLM and add: *"I'm at Phase X. Help me execute."* Both docs are self-contained — every command, every expected output, every fallback is explicit.

### "Pod cost is climbing"

Check the watchdog timer. Hard cap is 2 hours by default. If you've been running >2 hours and stage 1 still hasn't finished, something is wrong — kill via:
```bash
pkill -9 -f finetune_k0.py
cat katherine_k0.stderr.log | tail -50
```

Then post the tail output for diagnosis. Do not let the pod run blind for hours.

---

## Reference

### All env vars

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | unset | HF write token (required for sync) |
| `HF_BUCKET` | bochen2079/katherine-k0 | bucket destination |
| `HF_SYNC_ENABLED` | 1 | set 0 to skip all HF pushes |
| `BASE_MODEL` | unsloth/Qwen3.5-9B | base model HF repo |
| `SFT_RANK` | 64 | LoRA rank for SFT |
| `SFT_ALPHA` | 128 | LoRA alpha |
| `SFT_DROPOUT` | 0.05 | LoRA dropout |
| `SFT_EPOCHS` | 3 | SFT epochs |
| `SFT_LR` | 1e-4 | SFT learning rate |
| `SFT_BATCH` | 16 | SFT per-device batch |
| `SFT_GRAD_ACCUM` | 2 | SFT grad accumulation |
| `SFT_MAX_SEQ` | 1024 | max sequence length |
| `DPO_EPOCHS` | 2 | DPO epochs |
| `DPO_LR` | 5e-6 | DPO learning rate |
| `DPO_BETA` | 0.1 | DPO KL strength |
| `DPO_BATCH` | 4 | DPO batch |
| `DPO_GRAD_ACCUM` | 2 | DPO grad accum |
| `GGUF_QUANTS` | "q4_k_m q5_k_m q6_k" | GGUF quants to produce |
| `WALLCLOCK_HARD_CAP` | 7200 | hard wallclock cap (seconds) |
| `SIGUSR1_LEAD` | 300 | seconds before hard cap to send SIGUSR1 |
| `SKIP_SFT` | 0 | skip Stage 1 |
| `SKIP_DPO` | 0 | skip Stage 2 |
| `SKIP_GGUF` | 0 | skip Stage 3 |
| `SKIP_PUSH` | 0 | skip Stage 4 |

### URLs

| Purpose | URL |
|---|---|
| RunPod pods console | https://www.runpod.io/console/pods |
| RunPod SSH keys | https://www.runpod.io/console/user/settings |
| HF tokens | https://huggingface.co/settings/tokens |
| HF bucket | https://huggingface.co/buckets/bochen2079/katherine-k0 |
| GitHub repo | https://github.com/bochen2029-pixel/katherine-k0-finetune |
| Bootstrap (curl direct) | https://raw.githubusercontent.com/bochen2029-pixel/katherine-k0-finetune/master/bootstrap-runpod.sh |
| Qwen3.5-9B base | https://huggingface.co/Qwen/Qwen3.5-9B |
| Qwen3.5 Unsloth docs | https://unsloth.ai/docs/models/qwen3.5 |
| Buddhabrot reference repo (sister project) | https://github.com/bochen2029-pixel/buddhabrot-cuda-multigpu |

### Cheat sheet — common commands

```bash
# Status check
ls -la katherine_k0.{DONE,FATAL,stderr.log,watchdog.log} 2>/dev/null
tail -20 katherine_k0.stderr.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv

# Skip-ahead reruns
SKIP_SFT=1 ./run-cloud-runpod.sh                            # SFT done; rerun DPO + GGUF + push
SKIP_SFT=1 SKIP_DPO=1 ./run-cloud-runpod.sh                 # only GGUF + push
SKIP_SFT=1 SKIP_DPO=1 SKIP_GGUF=1 ./run-cloud-runpod.sh     # only push

# Force kill
pkill -9 -f finetune_k0.py
pkill -9 -f dpo_k0.py
pkill -9 -f _supervise-cloud.sh

# Manual HF push (if watchdog sync failed)
hf sync . hf://buckets/bochen2079/katherine-k0/ \
    --include "adapters/*/**" --include "gguf/**/*.gguf" --include "data/*.jsonl"
```

---

*Updates land at the master branch. To pull the latest version on a running pod: `cd ~/katherine-k0-finetune && git pull`.*
