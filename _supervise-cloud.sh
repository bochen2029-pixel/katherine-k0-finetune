#!/usr/bin/env bash
# Cloud watchdog for katherine-k0-finetune. Wraps the orchestrator with:
#   - Hard wallclock cap (SIGUSR1 at T-N, then SIGTERM, then SIGKILL)
#   - Background HF sync of adapter checkpoints as they're written
#   - Stderr captured to log file
#   - .DONE / .FATAL sentinel flags
#
# Same shape as buddhabrot-cuda-multigpu/_supervise-cloud.sh — the bones are
# generic "long-running GPU compute job under watchdog" infrastructure.

set -uo pipefail

OUTPUT_BASE=""
HARD_CAP=7200
SIGUSR1_LEAD=300
HF_SYNC_ENABLED=0
HF_BUCKET=""

while [ $# -gt 0 ]; do
    case "$1" in
        --output-base)   OUTPUT_BASE="$2"; shift 2 ;;
        --hard-cap)      HARD_CAP="$2";    shift 2 ;;
        --sigusr1-lead)  SIGUSR1_LEAD="$2"; shift 2 ;;
        --hf-sync)       HF_SYNC_ENABLED="$2"; shift 2 ;;
        --hf-bucket)     HF_BUCKET="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUTPUT_BASE" ] || [ $# -eq 0 ]; then
    echo "usage: _supervise-cloud.sh --output-base BASE [opts] -- CMD ARGS..." >&2
    exit 2
fi

LOG_PATH="${OUTPUT_BASE}.stderr.log"
PID_PATH="${OUTPUT_BASE}.pid"
DONE_PATH="${OUTPUT_BASE}.DONE"
FATAL_PATH="${OUTPUT_BASE}.FATAL"
WATCHDOG_LOG="${OUTPUT_BASE}.watchdog.log"

log() {
    echo "[watchdog $(date -u +%H:%M:%S)] $*" | tee -a "$WATCHDOG_LOG" >&2
}

# Background HF sync of adapter dirs as they appear or change.
# Uses `hf sync` URL form (the only working bucket-upload syntax).
hf_sync_dir() {
    local local_dir="$1"
    local remote_subdir="$2"
    if [ "$HF_SYNC_ENABLED" != "1" ] || [ -z "$HF_BUCKET" ]; then return; fi
    if [ ! -d "$local_dir" ]; then return; fi
    local synclog="${local_dir%/}.hfsync.log"
    log "  HF sync $local_dir → hf://buckets/$HF_BUCKET/$remote_subdir/ (background)"
    (
        hf sync "$local_dir" "hf://buckets/$HF_BUCKET/$remote_subdir/" \
            > "$synclog" 2>&1 \
            || echo "[hf-sync FAIL] $local_dir" >> "$synclog"
    ) &
}

# Track sync timestamps so we don't re-sync unchanged dirs constantly
declare -A LAST_SYNC

scan_and_sync() {
    [ "$HF_SYNC_ENABLED" = "1" ] || return
    # Sync any adapter checkpoint dir that's >5 min newer than its last sync
    for d in adapters/k0_sft_adapter adapters/k0_dpo_adapter; do
        if [ -d "$d" ]; then
            local mtime
            mtime=$(stat -c %Y "$d" 2>/dev/null || echo 0)
            local last="${LAST_SYNC[$d]:-0}"
            if [ "$mtime" -gt "$last" ]; then
                local subdir
                subdir=$(basename "$d")
                hf_sync_dir "$d" "$subdir"
                LAST_SYNC[$d]="$mtime"
            fi
        fi
    done
}

log "launching: $* (hard-cap ${HARD_CAP}s, SIGUSR1 at T-${SIGUSR1_LEAD}s)"
"$@" > "$LOG_PATH" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_PATH"
log "train PID: $TRAIN_PID"

START_TS=$(date +%s)
SIGUSR1_AT=$(( START_TS + HARD_CAP - SIGUSR1_LEAD ))
HARD_AT=$(( START_TS + HARD_CAP ))
SIGUSR1_FIRED=0
SIGTERM_FIRED=0

forward_term() {
    log "received SIGTERM/SIGINT; forwarding to train PID $TRAIN_PID"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
}
trap forward_term TERM INT

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TS ))

    if [ "$SIGUSR1_FIRED" = "0" ] && [ "$NOW" -ge "$SIGUSR1_AT" ]; then
        log "T-${SIGUSR1_LEAD}s reached; firing SIGUSR1 to train PID $TRAIN_PID"
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
        SIGUSR1_FIRED=1
    fi
    if [ "$SIGTERM_FIRED" = "0" ] && [ "$NOW" -ge "$HARD_AT" ]; then
        log "HARD CAP reached at ${ELAPSED}s; firing SIGTERM"
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        SIGTERM_FIRED=1
        sleep 60
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            log "SIGTERM did not work; SIGKILL"
            kill -KILL "$TRAIN_PID" 2>/dev/null || true
        fi
    fi

    # Periodic HF sync of adapter dirs as they update
    scan_and_sync

    sleep 30
done

wait "$TRAIN_PID"
EXIT_CODE=$?
END_TS=$(date +%s)
TOTAL_SEC=$(( END_TS - START_TS ))

log "train exited code=$EXIT_CODE after ${TOTAL_SEC}s"

# Final sync of any final state
scan_and_sync

# Sync logs at the end
if [ "$HF_SYNC_ENABLED" = "1" ] && [ -n "$HF_BUCKET" ]; then
    log "final log push"
    for f in "$LOG_PATH" "$WATCHDOG_LOG" "${OUTPUT_BASE}.launch.log"; do
        if [ -f "$f" ]; then
            hf_sync_dir "$(dirname "$f")" "logs"  # just the dir; --include filter would be ideal
            break
        fi
    done
    sleep 10  # give background syncs a head start
fi

if [ "$EXIT_CODE" = "0" ]; then
    log "DONE"
    : > "$DONE_PATH"
else
    log "FATAL (exit $EXIT_CODE)"
    : > "$FATAL_PATH"
fi

exit "$EXIT_CODE"
