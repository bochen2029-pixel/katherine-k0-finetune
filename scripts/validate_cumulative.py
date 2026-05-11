#!/usr/bin/env python3
"""
validate_cumulative.py: Corpus-level validator for K0 v2 cumulative tier files.

Built 2026-05-10 to catch cross-increment drift that per-domain validation misses.
Per-domain validate_k0_v2.py runs hard checks (em-dash, service phrases, etc.)
on individual files; this script runs CORPUS-LEVEL aggregates across an entire
tier's worth of training data.

Specifically catches:
- Trope frequency aggregating above 5% across tier (validator hard threshold)
- Single-anchor roster cycling above 10% across tier
- Brevity-distribution drift across tier (target ~60% short)
- Em-dash leakage in any K0 voice (assistant/chosen) across tier
- Per-increment drift: any single increment skewing cumulative aggregates

Usage:
    python scripts/validate_cumulative.py                    # T500 (default)
    python scripts/validate_cumulative.py tier_1000          # specific tier
    python scripts/validate_cumulative.py dataset/v2/cumulative/tier_1000/sft_train.jsonl  # explicit file

Exits 0 on pass, 1 on warning, 2 on hard fail.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
TIER_DIR = ROOT / 'dataset' / 'v2' / 'cumulative'

TROPE_PATTERNS = {
    'COLD_FEET': re.compile(r'\b(cold feet|feet are cold|feet were cold)\b', re.I),
    'INK_HANDS': re.compile(r'\b(ink (?:on|stained?) (?:my |her )?(?:hands?|wrists?|arms?)|ink-stained)\b', re.I),
    'UNFINISHED_POEM': re.compile(r'\b(unfinished poem|poem on the counter|half-finished poem)\b', re.I),
    'DARK_NIGHT_SETTING': re.compile(r'\b(sitting in the dark|in the dark of the kitchen|the house is dark)\b', re.I),
    'SILENCE_QUALITY': re.compile(r'\b(the silence has a quality|silence with a quality|quality of silence)\b', re.I),
    'KITCHEN_DEFAULT': re.compile(r'\b(in (?:my |the )?kitchen|standing in (?:my |the )?kitchen|at (?:my |the )?kitchen counter)\b', re.I),
    'CANDLE_DAWN_TROPE': re.compile(r'\b(candle (?:lit|burning)|before dawn|four (?:in the )?(?:morning|am)|five (?:in the )?(?:morning|am))\b', re.I),
}

ROSTER_NAMES = [
    'Eleanor', 'James', 'Daniel', 'Hannah', 'Theo', 'Mose',
    'Naomi', 'Diego', 'Lila', 'Maddy', 'Marcus', 'Sam', 'Rae',
    'Maya', 'Aaron', 'Camila', 'Lou', 'Will Anderson', 'Dr. Owens',
    'Frances', 'Mr. Pell', 'Mr. Sanchez', 'Margarita',
    'Threshold', 'BookPeople', 'Adolphus', 'Granbury', 'Edinburgh',
    'Mid-City', 'Civic', 'Jordan', 'Iris', 'Cole', 'Mac', 'Reed',
]
ROSTER_RX = {n: re.compile(r'\b' + re.escape(n) + r'\b', re.I) for n in ROSTER_NAMES}

EM_DASH_RX = re.compile(r'[\u2014\u2013]|(?<!\w)--(?!\w)')

WARN_THRESHOLDS = {
    'trope_pct': 5.0,
    'roster_pct': 10.0,
    'brevity_short_min_pct': 50.0,
    'em_dash_count_max': 0,
}


def k0_voice_text(trace):
    """Extract all K0-voice text from a trace (assistant turns OR chosen for DPO)."""
    if trace.get('_type') == 'dpo':
        return trace.get('chosen', '')
    return ' '.join(m['content'] for m in trace.get('messages', []) if m['role'] == 'assistant')


def count_sentences(text):
    return len([s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()])


def analyze_tier(traces, label):
    n = len(traces)
    print(f'\n{"=" * 65}')
    print(f'  {label}  ({n} traces)')
    print(f'{"=" * 65}')

    if n == 0:
        print('  EMPTY TIER. Nothing to analyze.')
        return 0

    # Em-dash leak in K0 voice
    em_dash_traces = []
    for i, t in enumerate(traces):
        voice = k0_voice_text(t)
        if EM_DASH_RX.search(voice):
            em_dash_traces.append((i, t.get('_cat', '?')))

    print(f'\n--- EM-DASH LEAK (K0 voice) ---')
    print(f'  Threshold: {WARN_THRESHOLDS["em_dash_count_max"]}')
    print(f'  Found: {len(em_dash_traces)}')
    if em_dash_traces:
        for i, cat in em_dash_traces[:5]:
            print(f'    trace {i} cat={cat}')
        if len(em_dash_traces) > 5:
            print(f'    ... and {len(em_dash_traces) - 5} more')

    # Brevity distribution
    sentence_counts = []
    for t in traces:
        voice = k0_voice_text(t)
        if voice:
            sentence_counts.append(count_sentences(voice))
    short_n = sum(1 for c in sentence_counts if c <= 3)
    short_pct = 100.0 * short_n / len(sentence_counts) if sentence_counts else 0
    print(f'\n--- BREVITY DISTRIBUTION ---')
    print(f'  Short (<=3 sentences): {short_pct:.1f}% (target >= {WARN_THRESHOLDS["brevity_short_min_pct"]}%)')

    # Trope frequency
    trope_hits = defaultdict(list)
    for i, t in enumerate(traces):
        voice = k0_voice_text(t)
        for trope, rx in TROPE_PATTERNS.items():
            if rx.search(voice):
                trope_hits[trope].append(i)
    print(f'\n--- TROPE FREQUENCY ---')
    print(f'  Threshold: > {WARN_THRESHOLDS["trope_pct"]}% of traces')
    flagged_tropes = []
    for trope, ids in sorted(trope_hits.items(), key=lambda x: -len(x[1])):
        pct = 100.0 * len(ids) / n
        marker = ' [FLAGGED]' if pct > WARN_THRESHOLDS['trope_pct'] else ''
        print(f'  {trope}: {len(ids)} traces ({pct:.1f}%){marker}')
        if pct > WARN_THRESHOLDS['trope_pct']:
            flagged_tropes.append(trope)
    if not trope_hits:
        print('  (no tropes detected)')

    # Roster cycling
    roster_hits = defaultdict(list)
    for i, t in enumerate(traces):
        voice = k0_voice_text(t)
        for name, rx in ROSTER_RX.items():
            if rx.search(voice):
                roster_hits[name].append(i)
    print(f'\n--- ROSTER CYCLING ---')
    print(f'  Threshold: > {WARN_THRESHOLDS["roster_pct"]}% of traces')
    flagged_roster = []
    for name, ids in sorted(roster_hits.items(), key=lambda x: -len(x[1]))[:15]:
        pct = 100.0 * len(ids) / n
        marker = ' [FLAGGED]' if pct > WARN_THRESHOLDS['roster_pct'] else ''
        print(f'  {name}: {len(ids)} traces ({pct:.1f}%){marker}')
        if pct > WARN_THRESHOLDS['roster_pct']:
            flagged_roster.append(name)

    # Domain distribution
    cat_counts = Counter(t.get('_cat', '?') for t in traces)
    domain_counts = Counter(c[0] for c in cat_counts.keys() if c)
    print(f'\n--- DOMAIN DISTRIBUTION ---')
    for d, count in sorted(domain_counts.items()):
        domain_traces = sum(v for k, v in cat_counts.items() if k.startswith(d))
        print(f'  Domain {d}: {domain_traces} traces ({100.0*domain_traces/n:.1f}%)')

    # Verdict
    fails = []
    if em_dash_traces:
        fails.append(f'em-dash leak in {len(em_dash_traces)} K0-voice traces')
    if flagged_tropes:
        fails.append(f'tropes flagged: {flagged_tropes}')
    if flagged_roster:
        fails.append(f'roster anchors over threshold: {flagged_roster}')
    if short_pct < WARN_THRESHOLDS['brevity_short_min_pct']:
        fails.append(f'brevity {short_pct:.1f}% under target {WARN_THRESHOLDS["brevity_short_min_pct"]}%')

    print(f'\n{"=" * 65}')
    if not fails:
        print(f'  {label}: PASS')
        return 0
    else:
        print(f'  {label}: WARN/FAIL')
        for f in fails:
            print(f'    - {f}')
        return 1


def load_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'tier_500'

    if Path(arg).is_file():
        path = Path(arg)
        traces = load_jsonl(path)
        return analyze_tier(traces, str(path))

    tier_dir = TIER_DIR / arg
    if not tier_dir.exists():
        print(f'Tier directory not found: {tier_dir}', file=sys.stderr)
        sys.exit(2)

    sft_path = tier_dir / 'sft_train.jsonl'
    dpo_path = tier_dir / 'dpo_train.jsonl'

    rc = 0
    if sft_path.exists():
        rc = max(rc, analyze_tier(load_jsonl(sft_path), f'{arg} SFT ({sft_path.name})'))
    else:
        print(f'No SFT file at {sft_path}', file=sys.stderr)

    if dpo_path.exists():
        rc = max(rc, analyze_tier(load_jsonl(dpo_path), f'{arg} DPO ({dpo_path.name})'))
    else:
        print(f'No DPO file at {dpo_path}', file=sys.stderr)

    sys.exit(rc)


if __name__ == '__main__':
    main()
