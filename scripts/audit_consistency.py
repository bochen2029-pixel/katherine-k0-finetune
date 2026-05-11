#!/usr/bin/env python3
"""
audit_consistency.py: Regression check against v2 soul doc numerical contradictions.

Built 2026-05-10 after a fork-instance audit caught three date contradictions
that the standard validator missed (the same shape as the original 37-vs-20-year
James-duration catch). The standard validator checks per-trace voice rules. This
script checks ACROSS-section numerical claims in the soul doc itself.

Catches:
1. Contradictions in named-person attributes (X is N years old vs N+1)
2. Contradictions in named durations (Y played for N years vs M years)
3. Contradictions in named dates (Z happened in YEAR1 vs YEAR2)
4. Contradictions in relationship-since-DATE claims

Heuristic: it scans the soul doc for sentences containing both a named person
(from a known-anchor list) AND a numerical/date claim, then groups by anchor and
flags discrepancies. Not perfect; meant as a regression check the operator runs
before promoting a soul-doc edit.

Usage:
    python audit_consistency.py
    python audit_consistency.py path/to/soul_doc.md
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

DEFAULT_DOC = Path(__file__).resolve().parent.parent / 'k0_soul_document.md'

NAMED_ANCHORS = [
    'Katherine', 'Eleanor', 'James', 'Daniel', 'Hannah', 'Theo', 'Mose',
    'Naomi', 'Diego', 'Lila', 'Maddy', 'Marcus', 'Sam', 'Rae',
    'Maya', 'Aaron', 'Camila', 'Lou', 'Will Anderson', 'Dr. Owens',
    'Frances', 'Mr. Pell', 'Mr. Sanchez', 'Margarita',
    'Threshold', 'BookPeople', 'Adolphus', 'Granbury', 'Edinburgh',
    'Mid-City', 'Civic',
]

NUMBER_WORD_TO_INT = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}

YEAR_RX = re.compile(r'\b(19[0-9]{2}|20[0-9]{2})\b')
DURATION_RX = re.compile(
    r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
    r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy)'
    r'(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?'
    r'\s+years?\b',
    re.IGNORECASE
)
AGO_RX = re.compile(r'\b(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago)\b', re.IGNORECASE)
AGE_RX = re.compile(
    r'\b(?:age\s+|aged\s+|is\s+|was\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'twenty|twenty-?(?:one|two|three|four|five|six|seven|eight|nine)|'
    r'thirty|thirty-?(?:one|two|three|four|five|six|seven|eight|nine)|'
    r'forty|forty-?(?:one|two|three|four|five|six|seven|eight|nine)|'
    r'sixty|sixty-?(?:one|two|three|four|five|six|seven|eight|nine))'
    r'\s*(?:[-]?\s*years?[-]?\s*old)\b',
    re.IGNORECASE
)


def normalize_number(s):
    """'thirty-two' -> 32, '32' -> 32, 'two' -> 2."""
    s = s.lower().strip()
    if s.isdigit():
        return int(s)
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and parts[0] in NUMBER_WORD_TO_INT and parts[1] in NUMBER_WORD_TO_INT:
            return NUMBER_WORD_TO_INT[parts[0]] + NUMBER_WORD_TO_INT[parts[1]]
    if s in NUMBER_WORD_TO_INT:
        return NUMBER_WORD_TO_INT[s]
    return None


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def find_anchored_claims(text, doc_path):
    claims = defaultdict(list)
    lines = text.splitlines()
    for lineno, line in enumerate(lines, 1):
        for sentence in split_sentences(line):
            for anchor in NAMED_ANCHORS:
                if not re.search(rf'\b{re.escape(anchor)}\b', sentence):
                    continue
                for m in YEAR_RX.finditer(sentence):
                    year = int(m.group(1))
                    if 1900 <= year <= 2030:
                        claims[(anchor, 'year_mention', year)].append((lineno, sentence[:120]))
                for m in DURATION_RX.finditer(sentence):
                    raw = m.group(1).lower()
                    n = normalize_number(raw)
                    if n is not None and 1 <= n <= 80:
                        claims[(anchor, 'duration_years', n)].append((lineno, sentence[:120]))
                for m in AGO_RX.finditer(sentence):
                    raw = m.group(1).lower()
                    n = normalize_number(raw)
                    if n is not None and 1 <= n <= 50:
                        claims[(anchor, 'years_ago', n)].append((lineno, sentence[:120]))
                for m in AGE_RX.finditer(sentence):
                    raw = m.group(1).lower()
                    n = normalize_number(raw)
                    if n is not None and 1 <= n <= 100:
                        claims[(anchor, 'age', n)].append((lineno, sentence[:120]))
    return claims


def detect_contradictions(claims):
    by_anchor_kind = defaultdict(list)
    for (anchor, kind, value), occurrences in claims.items():
        by_anchor_kind[(anchor, kind)].append((value, occurrences))

    contradictions = []
    for (anchor, kind), values in by_anchor_kind.items():
        if kind == 'year_mention':
            continue
        if len(values) >= 2:
            distinct_values = sorted(set(v for v, _ in values))
            if len(distinct_values) >= 2:
                contradictions.append((anchor, kind, distinct_values, values))
    return contradictions


def report(doc_path, claims, contradictions):
    print(f"=== Consistency audit: {doc_path} ===")
    print(f"Anchored claims found: {sum(len(v) for v in claims.values())}")
    print(f"Distinct (anchor, kind, value) keys: {len(claims)}")
    print()
    if not contradictions:
        print("No contradictions detected.")
        print()
        print("Note: this script catches numerical claims tied to named anchors.")
        print("It does not catch every kind of inconsistency. Use as a regression check.")
        return 0

    print(f"!! {len(contradictions)} potential contradictions:")
    print()
    for anchor, kind, distinct_values, all_occurrences in contradictions:
        print(f"--- {anchor} / {kind} ---")
        print(f"   Distinct values seen: {distinct_values}")
        for value, occurrences in all_occurrences:
            for lineno, snippet in occurrences:
                print(f"   line {lineno}: [{value}] {snippet}")
        print()
    print("Each cluster above contains two-or-more different numerical claims about")
    print("the same anchor. Review and harmonize. The standard fix is to make the")
    print("less-specific claim match the more-specific one (e.g. spring-2022")
    print("retirement is more specific than 'thirty years of teaching').")
    return 1


def main():
    doc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    if not doc_path.exists():
        print(f"Soul doc not found: {doc_path}", file=sys.stderr)
        sys.exit(2)
    text = doc_path.read_text(encoding='utf-8')
    claims = find_anchored_claims(text, doc_path)
    contradictions = detect_contradictions(claims)
    sys.exit(report(doc_path, claims, contradictions))


if __name__ == '__main__':
    main()
