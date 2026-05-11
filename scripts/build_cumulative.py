#!/usr/bin/env python3
"""
build_cumulative.py: Auto-build cumulative tier files from increments.

Reads increment_001/ through increment_005/ and produces the five
cumulative tier_N/sft_train.jsonl + tier_N/dpo_train.jsonl files
under cumulative/.

Cumulative tiers are nested:
  tier_500    = increment_001
  tier_1000   = tier_500 + increment_002
  tier_2500   = tier_1000 + increment_003
  tier_5000   = tier_2500 + increment_004
  tier_7500   = tier_5000 + increment_005

If an increment is missing or partial, the script builds whichever tiers
it can and reports the rest as not-yet-buildable. Idempotent. Re-run
any time after generating new increments.

Usage:
    python scripts/build_cumulative.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / 'dataset' / 'v2'

# Vanilla mode: strip vision (Domain I) and audio (Domain P) SFT, plus
# DPO-VISION-REGISTER and DPO-VOICE-REGISTER pairs. Used when operator wants
# to train text-only K0 from a tier that includes vision/audio registers.
VANILLA_SFT_DOMAIN_PREFIXES = ('I', 'P')
VANILLA_DPO_EXCLUDED_CATS = {'DPO-VISION-REGISTER', 'DPO-VOICE-REGISTER'}

TIERS = [
    ('tier_500',   ['increment_001']),
    ('tier_1000',  ['increment_001', 'increment_002']),
    ('tier_2500',  ['increment_001', 'increment_002', 'increment_003']),
    ('tier_5000',  ['increment_001', 'increment_002', 'increment_003', 'increment_004']),
    ('tier_7500',  ['increment_001', 'increment_002', 'increment_003', 'increment_004', 'increment_005']),
    ('tier_10000', ['increment_001', 'increment_002', 'increment_003', 'increment_004', 'increment_005', 'increment_006']),
]


def load_increment_files(increment_path, kind):
    """kind is 'sft' or 'dpo'. Returns list of all jsonl lines as dicts."""
    files = sorted((increment_path / kind).glob('*.jsonl'))
    rows = []
    for f in files:
        with f.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [warn] {f.name} line skipped (parse error): {e}")
    return rows


def increment_has_content(increment_path):
    """An increment counts as 'present' only if it has actual jsonl files in
    sft/ or dpo/. Empty scaffolded directories do NOT count."""
    if not increment_path.exists():
        return False
    sft_files = list((increment_path / 'sft').glob('*.jsonl'))
    dpo_files = list((increment_path / 'dpo').glob('*.jsonl'))
    return bool(sft_files or dpo_files)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla", action="store_true",
                   help="Build text-only tiers (strip Domain I + P SFT, "
                        "strip DPO-VISION-REGISTER + DPO-VOICE-REGISTER). "
                        "Output goes to cumulative/tier_X_vanilla/ instead of cumulative/tier_X/.")
    args = p.parse_args()

    suffix = '_vanilla' if args.vanilla else ''
    print(f"[v2] Building cumulative tiers under {V2}/cumulative (vanilla={args.vanilla})")

    for tier_name, increments in TIERS:
        print(f"\n[{tier_name}{suffix}]")

        sft_rows = []
        dpo_rows = []
        missing_increments = []

        for inc_name in increments:
            inc_path = V2 / inc_name
            if not increment_has_content(inc_path):
                missing_increments.append(inc_name)
                continue
            sft_rows.extend(load_increment_files(inc_path, 'sft'))
            dpo_rows.extend(load_increment_files(inc_path, 'dpo'))

        if missing_increments:
            print(f"  [skip] missing increments: {missing_increments}")
            print(f"  [skip] cannot build {tier_name}{suffix} until those exist")
            continue

        # Vanilla filter: strip vision (Domain I) and audio (Domain P) SFT,
        # plus DPO-VISION-REGISTER and DPO-VOICE-REGISTER. Output to a
        # separate tier_X_vanilla directory so the full and stripped versions
        # coexist on disk for parallel experimentation.
        if args.vanilla:
            sft_before = len(sft_rows)
            sft_rows = [r for r in sft_rows
                        if not any(r.get('_cat', '').startswith(d)
                                   for d in VANILLA_SFT_DOMAIN_PREFIXES)]
            dpo_before = len(dpo_rows)
            dpo_rows = [r for r in dpo_rows
                        if r.get('_cat', '') not in VANILLA_DPO_EXCLUDED_CATS]
            print(f"  [vanilla] SFT filter (drop {VANILLA_SFT_DOMAIN_PREFIXES}): {sft_before} -> {len(sft_rows)}")
            print(f"  [vanilla] DPO filter (drop {sorted(VANILLA_DPO_EXCLUDED_CATS)}): {dpo_before} -> {len(dpo_rows)}")

        sft_out = V2 / 'cumulative' / f'{tier_name}{suffix}' / 'sft_train.jsonl'
        dpo_out = V2 / 'cumulative' / f'{tier_name}{suffix}' / 'dpo_train.jsonl'

        write_jsonl(sft_out, sft_rows)
        write_jsonl(dpo_out, dpo_rows)

        print(f"  SFT: {len(sft_rows)} rows -> {sft_out}")
        print(f"  DPO: {len(dpo_rows)} rows -> {dpo_out}")

        # 80/20 SFT/DPO sanity check (added 2026-05-10 audit)
        if dpo_rows:
            ratio = len(dpo_rows) / (len(sft_rows) + len(dpo_rows))
            if ratio < 0.15 or ratio > 0.25:
                print(f"  [warn] SFT/DPO ratio = {ratio:.1%} DPO (target ~20%). Check increment DPO file completeness.")
        else:
            print(f"  [warn] DPO is empty. Increments missing dpo/*.jsonl files. Train will be SFT-only.")

    print("\n[v2] Done.")


if __name__ == '__main__':
    main()
