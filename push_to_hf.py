"""
Stage 4 — push trained adapters + GGUFs to HF bucket.

Bucket: bochen2079/katherine-k0  (pre-existing or auto-created on first push)

Subdirectory layout in bucket:
  k0_sft_adapter/         the SFT-only LoRA adapter
  k0_dpo_adapter/         the SFT+DPO LoRA adapter
  gguf/gguf_q4_k_m/       merged + quantized GGUF
  gguf/gguf_q5_k_m/
  gguf/gguf_q6_k/
  data/                   the canonical training data (snapshot)
  logs/                   training stderr + watchdog logs

Uses `hf sync DIR URL` (the verified bucket-upload syntax). Does NOT use
`hf upload --repo-type bucket` (broken in current CLI; rejects 'bucket' as
a repo-type).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def hf_sync(local_dir: str, remote_subdir: str, bucket: str, includes: list = None) -> bool:
    """Run `hf sync local_dir hf://buckets/<bucket>/<remote_subdir>/`.
    Returns True on success.
    """
    if not os.path.isdir(local_dir):
        print(f"[hf-sync] skip: {local_dir} does not exist")
        return False
    remote_url = f"hf://buckets/{bucket}/{remote_subdir}/"
    cmd = ["hf", "sync", local_dir, remote_url]
    if includes:
        for inc in includes:
            cmd += ["--include", inc]
    print(f"[hf-sync] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        out = (result.stdout or "") + (result.stderr or "")
        # Print last 20 lines for visibility
        for line in out.splitlines()[-20:]:
            print(f"  {line}")
        if result.returncode != 0:
            print(f"[hf-sync] FAIL (returncode={result.returncode})")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[hf-sync] TIMEOUT after 1 hour")
        return False
    except Exception as e:
        print(f"[hf-sync] EXC: {type(e).__name__}: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="bochen2079/katherine-k0")
    p.add_argument("--sft-adapter", default="adapters/k0_sft_adapter")
    p.add_argument("--dpo-adapter", default="adapters/k0_dpo_adapter")
    p.add_argument("--gguf-base-dir", default="gguf")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--logs-dir", default=".",
                   help="Directory containing *.stderr.log / *.watchdog.log to push")
    args = p.parse_args()

    # Auth check
    print("[auth] verifying HF login...")
    r = subprocess.run(["hf", "auth", "whoami"], capture_output=True, text=True)
    if r.returncode != 0 or "Logged in" not in (r.stdout + r.stderr):
        print("[auth] not logged in. Run: hf auth login --token $HF_TOKEN", file=sys.stderr)
        print(r.stdout)
        print(r.stderr)
        sys.exit(1)
    print("[auth]", (r.stdout or r.stderr).strip().split("\n")[0])

    results = {}

    # 1. SFT adapter
    if os.path.isdir(args.sft_adapter):
        results["sft_adapter"] = hf_sync(args.sft_adapter, "k0_sft_adapter", args.bucket)

    # 2. DPO adapter (if exists)
    if os.path.isdir(args.dpo_adapter):
        results["dpo_adapter"] = hf_sync(args.dpo_adapter, "k0_dpo_adapter", args.bucket)

    # 3. GGUFs (each quant subdir as its own remote subdir)
    if os.path.isdir(args.gguf_base_dir):
        for quant_subdir in sorted(Path(args.gguf_base_dir).iterdir()):
            if quant_subdir.is_dir() and quant_subdir.name.startswith("gguf_"):
                key = f"gguf/{quant_subdir.name}"
                results[key] = hf_sync(str(quant_subdir), key, args.bucket,
                                      includes=["*.gguf", "config.json", "tokenizer*"])

    # 4. Training data snapshot (small, useful for reproducibility)
    if os.path.isdir(args.data_dir):
        results["data"] = hf_sync(args.data_dir, "data", args.bucket,
                                 includes=["*.jsonl"])

    # 5. Logs
    if os.path.isdir(args.logs_dir):
        results["logs"] = hf_sync(args.logs_dir, "logs", args.bucket,
                                 includes=["*.stderr.log", "*.watchdog.log",
                                           "*.launch.log", "*.log"])

    print()
    print("=" * 60)
    print("[summary]")
    for key, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {key}")
    print()
    if all(results.values()):
        print("[done] all artifacts pushed")
        sys.exit(0)
    else:
        print("[partial] some pushes failed; see logs above")
        sys.exit(2)


if __name__ == "__main__":
    main()
