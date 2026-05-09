"""
Stage 3 — Merge LoRA into base + export GGUF in 3 quantization levels.

Reads the final (DPO if available, else SFT) adapter, merges it into the
Qwen3.5-9B base, then writes:
  gguf_q4_k_m/  ~5.5 GB    fastest, smallest
  gguf_q5_k_m/  ~6.5 GB    sweet spot for daily use
  gguf_q6_k/    ~7.7 GB    closest to BF16, quality reference

Failsafe:
  - Each quant is independent. If q5 fails, q4 + q6 are still useful.
  - Adapter is preserved; this stage doesn't modify it.
  - First run compiles llama.cpp internally (~5-10 min). Subsequent runs reuse.
"""
import argparse
import os
import sys
from pathlib import Path

from unsloth import FastModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True,
                   help="Path to the final adapter (SFT or DPO).")
    p.add_argument("--max_seq", type=int, default=1024)
    p.add_argument("--gguf-base-dir", default="gguf",
                   help="Parent directory; per-quant subdirs will be created.")
    p.add_argument("--quants", nargs="+", default=["q4_k_m", "q5_k_m", "q6_k"],
                   help="GGUF quantization methods to export.")
    args = p.parse_args()

    if not os.path.isdir(args.adapter):
        print(f"[error] adapter not found: {args.adapter}", file=sys.stderr)
        sys.exit(1)

    print(f"[load] adapter: {args.adapter}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        full_finetuning=False,
    )

    Path(args.gguf_base_dir).mkdir(parents=True, exist_ok=True)

    successes = []
    failures = []
    for quant in args.quants:
        out_dir = os.path.join(args.gguf_base_dir, f"gguf_{quant}")
        print()
        print(f"[gguf] === exporting {quant} → {out_dir} ===")
        try:
            model.save_pretrained_gguf(
                out_dir,
                tokenizer,
                quantization_method=quant,
            )
            # Unsloth's save_pretrained_gguf actually writes to <out_dir>_gguf/
            # (appends "_gguf" suffix), not <out_dir>/. Search both locations.
            search_dirs = [out_dir, f"{out_dir}_gguf"]
            produced = []
            for sd in search_dirs:
                if not os.path.isdir(sd):
                    continue
                for root, _, files in os.walk(sd):
                    for fn in files:
                        if fn.endswith(".gguf") and "mmproj" not in fn:
                            full = os.path.join(root, fn)
                            size_mb = os.path.getsize(full) / (1024 * 1024)
                            produced.append((full, size_mb))
            if produced:
                for f, sz in produced:
                    print(f"[gguf] OK: {f}  ({sz:.0f} MB)")
                successes.append((quant, produced))
            else:
                print(f"[gguf] WARN: no .gguf file found in {out_dir} or {out_dir}_gguf")
                failures.append((quant, "no .gguf produced"))
        except Exception as e:
            print(f"[gguf] FAIL: {type(e).__name__}: {e}")
            failures.append((quant, str(e)))

    print()
    print("=" * 60)
    print(f"[summary] {len(successes)} successful, {len(failures)} failed")
    for quant, files in successes:
        for f, sz in files:
            print(f"  ✓ {quant}: {f}  ({sz:.0f} MB)")
    for quant, err in failures:
        print(f"  ✗ {quant}: {err}")

    if not successes:
        print("[fatal] no quants succeeded; adapter is preserved")
        sys.exit(2)
    print()
    print("[done] merge + GGUF stage complete")


if __name__ == "__main__":
    main()
