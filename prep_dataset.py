"""
prep_dataset.py — build canonical k0 datasets from raw Katherine JSONL exports.

Input:  C:\\Katherine\\JSONLs\\*.jsonl  (~38 files, ~6,164 lines, mixed schema)
Output: data/k0_canonical.jsonl    — SFT examples, deduped, system-prompts stripped
        data/k0_dpo_curated.jsonl  — DPO pairs from k0_dpo_only.jsonl, prompts system-stripped

Why strip system prompts:
  See CLOUD.md § "Why unconditional Katherine". Short version: training with
  the K0 system prompt teaches `P(K_output | sysprompt_K)` — the model learns
  to be Katherine WHEN PROMPTED. Stripping makes it `P(K_output | nothing)` —
  the model unconditionally IS Katherine. Robust against sysprompt-removal,
  more honest as a persona deployment.

Run:
  python prep_dataset.py
  python prep_dataset.py --src /path/to/JSONLs --out-dir data/

Idempotent: re-running overwrites the canonical files atomically.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def load_unique_sft(src_dir: Path, exclude_files: set) -> dict:
    """Load all SFT examples (have 'messages' key), dedupe by content hash."""
    unique = {}
    for f in sorted(src_dir.glob("*.jsonl")):
        if f.name in exclude_files:
            continue
        with open(f, "r", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" not in obj or not obj.get("messages"):
                    continue
                msgs_str = json.dumps(obj["messages"], sort_keys=True, ensure_ascii=False)
                h = hashlib.md5(msgs_str.encode()).hexdigest()
                unique[h] = obj["messages"]
    return unique


def strip_system_messages(messages: list) -> list:
    """Drop all role=system entries. Return remaining user/assistant chain."""
    return [m for m in messages if m.get("role") != "system"]


def is_valid_sft(messages: list) -> bool:
    """Must start with user, contain at least one assistant turn."""
    if not messages:
        return False
    if messages[0].get("role") != "user":
        return False
    if not any(m.get("role") == "assistant" for m in messages):
        return False
    return True


def write_jsonl_atomic(path: Path, items: list) -> None:
    """Write JSONL via tmp+rename so partial writes never appear at the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_dpo_curated(src_file: Path) -> list:
    """Load k0_dpo_only.jsonl, strip system from prompts, return preference pairs."""
    out = []
    with open(src_file, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = obj.get("prompt", [])
            chosen = obj.get("chosen", [])
            rejected = obj.get("rejected", [])
            if not (prompt and chosen and rejected):
                continue
            prompt_stripped = strip_system_messages(prompt)
            if not prompt_stripped:
                continue
            out.append({
                "prompt": prompt_stripped,
                "chosen": chosen,
                "rejected": rejected,
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=r"C:\Katherine\JSONLs",
                   help="Source directory of raw Katherine JSONL files.")
    p.add_argument("--out-dir", default="data",
                   help="Output directory for canonical datasets.")
    p.add_argument("--dpo-source-file", default="k0_dpo_only.jsonl",
                   help="Filename within --src that holds the curated DPO pairs.")
    args = p.parse_args()

    src_dir = Path(args.src)
    if not src_dir.is_dir():
        print(f"[ERROR] source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.out_dir)

    # === SFT ===
    print(f"[sft] scanning {src_dir} (excluding {args.dpo_source_file})")
    sft_unique = load_unique_sft(src_dir, exclude_files={args.dpo_source_file})
    print(f"[sft] {len(sft_unique)} unique SFT examples after dedup")

    sft_stripped = []
    for h, msgs in sft_unique.items():
        clean = strip_system_messages(msgs)
        if is_valid_sft(clean):
            sft_stripped.append({"messages": clean})
    print(f"[sft] {len(sft_stripped)} valid examples after stripping system prompts")

    out_sft = out_dir / "k0_canonical.jsonl"
    write_jsonl_atomic(out_sft, sft_stripped)
    print(f"[sft] wrote {out_sft} ({os.path.getsize(out_sft)} bytes)")

    # === DPO ===
    dpo_src_file = src_dir / args.dpo_source_file
    if dpo_src_file.is_file():
        dpo_pairs = build_dpo_curated(dpo_src_file)
        print(f"[dpo] {len(dpo_pairs)} curated DPO pairs after stripping system from prompts")
        out_dpo = out_dir / "k0_dpo_curated.jsonl"
        write_jsonl_atomic(out_dpo, dpo_pairs)
        print(f"[dpo] wrote {out_dpo} ({os.path.getsize(out_dpo)} bytes)")
    else:
        print(f"[dpo] WARN: {dpo_src_file} not found; skipping DPO build")

    # === Stats ===
    turn_counts = {}
    char_lens = []
    for ex in sft_stripped:
        n = len(ex["messages"])
        turn_counts[n] = turn_counts.get(n, 0) + 1
        char_lens.append(sum(len(m.get("content", "")) for m in ex["messages"]))
    char_lens.sort()
    nn = len(char_lens)

    print()
    print("=== SFT corpus stats ===")
    for n in sorted(turn_counts.keys()):
        print(f"  {n}-turn: {turn_counts[n]}")
    print(f"  char-len p50: {char_lens[nn//2]}")
    print(f"  char-len p99: {char_lens[int(nn*0.99)]}")
    print(f"  approx tokens (chars/4) p50: {char_lens[nn//2]//4}")
    print(f"  approx tokens (chars/4) p99: {char_lens[int(nn*0.99)]//4}")
    print()
    print("[done] datasets ready under", out_dir)


if __name__ == "__main__":
    main()
