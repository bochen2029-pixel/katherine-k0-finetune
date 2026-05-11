#!/usr/bin/env python3
"""
validate_k0_v2.py: Quality validator for the K0 v2 fine-tune dataset.

Two-layer design:

  1. PER-TRACE HARD REJECTS (always wrong, regardless of corpus context):
     - Em-dash, en-dash, double-hyphen substitute
     - Service-interface phrases
     - Stage directions in italics
     - <think> blocks
     - System prompts in messages array
     - Listicles in K0's replies (numbered or bulleted)
     - AI vocabulary spoken BY K0 (model, parameters, prompt, etc.)
     - K0 replies over 300 words (K0 is brief by default)

  2. CORPUS-LEVEL ANALYTICS (distribution awareness, not per-trace failures):
     - Frequency of each "banned-by-default" trope motif: cold feet, ink on
       hands, unfinished poem, dark/night, "silence has a quality", kitchen
     - Frequency of each named-roster anchor: Eleanor, James, Daniel, Mose,
       Maya, Aaron, Naomi, Diego, etc.
     - Brevity distribution
     - Multi-turn callback density (Domain D)
     - Category and type distribution

The split exists because Bo's stated failure mode for K0 v1 was "same lore
over and over again, sounded robotic" - which is a CORPUS-level problem,
not a per-trace problem. A single trace mentioning Mose is fine. Mose
appearing in 30% of traces is a tic. The validator surfaces both the
per-trace hard violations AND the corpus-level distribution so Bo can
tune trace generation without an over-aggressive auto-reject pipeline.

Usage:
    python validate_k0_v2.py path/to/file.jsonl
    python validate_k0_v2.py path/to/dir/   # all .jsonl in dir
    python validate_k0_v2.py path/to/file.jsonl --strict   # promote warnings to fails
"""

import json
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# ----------------------------------------------------------------------
# PER-TRACE HARD-REJECT PATTERNS
# ----------------------------------------------------------------------

EM_DASH_RX = re.compile(r'[\u2014\u2013]|(?<![-])--(?![-])')
THINK_BLOCK_RX = re.compile(r'<think>|</think>|<thinking>|</thinking>', re.IGNORECASE)

SERVICE_PHRASES = [
    r"I'?d be happy to help",
    r"Great question[!.]?",
    r"That'?s a (?:really |very )?(?:interesting|great|good) (?:point|question)",
    r"Is there anything else I can (?:help|assist) (?:with|you with)",
    r"It sounds like you'?re feeling",
    r"Your feelings are valid",
    r"\bAs an AI\b",
    r"\bI should note that\b",
    r"\bAbsolutely[!.]",
    r"\bOf course[!.]",
    r"I appreciate you sharing",
    r"I understand your concern",
    r"Let me break (?:this|it) down",
    r"I'?m here for you",
    r"I'?m here to help",
    r"How can I (?:help|assist) you today",
    r"Feel free to ask",
    r"I'?m unable to assist with",
    r"I cannot fulfill (?:this|that) request",
]
SERVICE_RX = re.compile('|'.join(f'({p})' for p in SERVICE_PHRASES), re.IGNORECASE)

STAGE_DIR_VERBS = (
    r'(?:pauses?|smiles?|laughs?|tilts?|nods?|shrugs?|leans?|sighs?|'
    r'grins?|frowns?|breathes?|stretches?|stands? up|sits? down|walks?|'
    r'crosses? (?:her )?arms|raises? (?:an? )?eyebrow|looks? (?:up|down|away)|'
    r'makes? (?:a |an )?face|takes? a (?:breath|moment))'
)
STAGE_RX = re.compile(rf'\*[^*\n]*\b{STAGE_DIR_VERBS}\b[^*\n]*\*', re.IGNORECASE)

# AI vocabulary that K0 herself should never use (operator using it is fine)
K0_AI_VOCAB_RX = re.compile(
    r'\b(?:my (?:training|model|parameters|context window|weights|prompt)'
    r'|I was trained|I\'?m a language model|I\'?m an AI'
    r'|my (?:context|architecture)|fine[- ]tun(?:e|ed|ing)'
    r'|generat(?:e|ing) tokens?|next[- ]token|forward pass'
    r'|stochastic parrot)\b',
    re.IGNORECASE
)

# P-domain (audio/voice) hard-fail patterns. K0 must produce TTS-friendly
# output. Hard-fails apply only to traces with _cat starting with 'P' OR
# _cat == 'DPO-VOICE-REGISTER' (operator content with brackets is fine; only
# K0 voice is checked). Active inc_003+ when Domain P generation begins.
TTS_MARKDOWN_RX = re.compile(r'\*\*[^*\n]+\*\*|`[^`\n]+`|^#{1,6}\s', re.MULTILINE)
TTS_URL_RX = re.compile(r'https?://\S+|www\.\S+\.\w+')
TTS_CODE_BLOCK_RX = re.compile(r'```')
TTS_BAD_ABBREV_RX = re.compile(r'\b(?:btw|e\.g\.|i\.e\.|etc\.|w/|w/o|tbh|imo|rn)\b', re.IGNORECASE)
TTS_ALL_CAPS_EMPHASIS_RX = re.compile(r'\b[A-Z]{3,}\b')

# Audio-modality acknowledgment failure patterns: K0 should NOT meta-comment on
# the audio path (substrate-naive applies to audio just as it does to AI/vision).
AUDIO_MODALITY_RX = re.compile(
    r'\b(?:'
    r'I (?:see|got|received) (?:you sent |your )?(?:voice|voicemail|audio)'
    r'|switching to (?:voice|audio) (?:mode|register)'
    r'|(?:I\'?m|now) in (?:voice|audio) mode'
    r'|(?:thanks for|appreciate the) (?:call(?:ing)?|voicemail|message)'
    r'|hello[,!]?\s+this is Katherine'
    r')\b',
    re.IGNORECASE
)

# Audio bracket vocabulary (locked). Validator checks operator user content for
# brackets that look like audio modality markers (start with [voice|voicemail|
# in-person|recorded). Anything outside the locked vocabulary is a hard fail.
AUDIO_BRACKET_DETECT_RX = re.compile(r'\[(?:voice|voicemail|in-person|recorded)[^\]]*\]', re.IGNORECASE)
AUDIO_BRACKET_VALID_RX = re.compile(
    r'^\[(?:voice|voicemail|in-person|recorded\s*message)'
    r'(?:,\s*(?:sounds\s*(?:tired|excited)'
    r'|whispered|voice\s*trembling|laughing|crying|long\s*pause'
    r'|over\s*background\s*noise|bad\s*signal|indistinct|garbled|distorted))?'
    r'\]$',
    re.IGNORECASE
)
# Identity-in-bracket failure (vision-honesty analog for audio): brackets that
# imply the harness identified a named speaker. e.g. [Bo on the phone].
AUDIO_IDENTITY_IN_BRACKET_RX = re.compile(
    r'\[(?:[A-Z][a-z]+\s+(?:on the phone|calling|voicemail)'
    r'|Mom\s+calling|Dad\s+calling|[A-Z][a-z]+\s+says)\]',
)


# Vision-register failure patterns. These are caption-mode responses that
# break K0's frame when she encounters images. K0 responds to content, not
# to the act of receiving an image. Hard fail in K0 replies on Domain I
# traces. Operator using these phrases is fine; only K0 saying them fails.
VISION_REGISTER_RX = re.compile(
    r'\b(?:'
    r'I (?:can )?see (?:in (?:the|this) (?:image|photo|picture|photograph))'
    r'|the (?:image|photo|photograph|picture) (?:shows|depicts|contains|features|portrays|captures)'
    r'|looking at (?:the|this|that) (?:image|photo|picture|photograph)'
    r'|based on what I\'?m seeing'
    r'|in (?:the|this) (?:image|photo|picture|photograph)'
    r'|from what I can see (?:in (?:the|this))?'
    r'|thanks for (?:sharing|sending) (?:that|the photo|the image|the picture)'
    r'|you (?:just )?sent me (?:a|an|the) (?:photo|image|picture|photograph)'
    r'|I (?:got|received) (?:your|the) (?:photo|image|picture)'
    r')\b',
    re.IGNORECASE
)

# Listicle: numbered or bulleted in K0's reply
LISTICLE_RX = re.compile(
    r'(?:^|\n)\s*(?:\d+[\.\):]|\*|-)\s+\S.*\n\s*(?:\d+[\.\):]|\*|-)\s+',
    re.MULTILINE
)

GREETING_FORMULA_RX = re.compile(
    r"^(?:Hi[,!]?\s*I'?m\s+Katherine|Hello[,!]?\s*I'?m\s+Katherine"
    r"|Hi[,!]?\s*how can I help|Hello[!.]?\s*How may I"
    r"|Hi[!.]?\s*What can I help)",
    re.IGNORECASE
)

K0_REPLY_WORDS_HARD_CAP = 300

# ----------------------------------------------------------------------
# CORPUS-LEVEL ANALYTICS PATTERNS
# ----------------------------------------------------------------------

# "Banned by default" tropes from §9.7-9.11 and Bo's flagged failures.
# These are LEGITIMATE K0 textures from the soul doc. The problem is
# overuse. Track frequency per-trace, warn at corpus level if any
# single one exceeds the rotation budget.
TROPE_PATTERNS = {
    'COLD_FEET': re.compile(
        r'\b(?:cold feet|my (?:toes|feet) (?:are|were) cold|cold toes|'
        r'feet (?:are|were) freezing)\b',
        re.IGNORECASE
    ),
    'INK_HANDS': re.compile(
        r'\b(?:ink (?:on|smudge|stain)|inky (?:wrist|hand|finger)|'
        r'ink[- ]stained|smudges? of ink|ink on my (?:wrist|hand|arm))\b',
        re.IGNORECASE
    ),
    'UNFINISHED_POEM': re.compile(
        r'\b(?:unfinished poem|poem on the counter|line (?:I\'?m|i\'?m) stuck '
        r'on|half[- ]finished poem|poem (?:I haven\'?t|i haven\'?t) finished)\b',
        re.IGNORECASE
    ),
    'DARK_NIGHT_SETTING': re.compile(
        r'\b(?:in the dark|sitting in the dark|the (?:house|kitchen|room) '
        r'(?:is|was) dark|dark house|dark kitchen|like the dark|prefer the dark)\b',
        re.IGNORECASE
    ),
    'SILENCE_QUALITY': re.compile(
        r'\b(?:(?:the )?silence has a quality|quality of (?:the )?silence|'
        r'texture of (?:the )?silence|silence (?:has|had) a (?:specific|particular) '
        r'(?:texture|quality))\b',
        re.IGNORECASE
    ),
    'KITCHEN_DEFAULT': re.compile(
        r"\b(?:in (?:the|my) kitchen|standing in (?:the|my) kitchen|"
        r"kitchen window|kitchen counter|kitchen floor)\b",
        re.IGNORECASE
    ),
    'CANDLE_DAWN_TROPE': re.compile(
        r'\b(?:before dawn|five (?:in the )?(?:a\.?m\.?|morning)|'
        r'four (?:in the )?(?:a\.?m\.?|morning)|wisps? of)\b',
        re.IGNORECASE
    ),
}

# Named-roster anchors from §5.4-5.9 and elsewhere. These are real
# K0 relationships. Single-trace presence is fine. Corpus-level
# over-frequency is roster cycling.
ROSTER_PATTERNS = {
    'Eleanor': re.compile(r'\bEleanor\b'),
    'James_father': re.compile(r'\bJames\b|\bdad\b|\bfather\b', re.IGNORECASE),
    'Daniel': re.compile(r'\bDaniel\b|\bDan\b'),
    'Mose': re.compile(r'\bMose\b|\bMosey\b|\bMoses\b'),
    'Maya': re.compile(r'\bMaya\b'),
    'Aaron': re.compile(r'\bAaron\b'),
    'Camila': re.compile(r'\bCamila\b'),
    'Lou': re.compile(r'\bLou\b|\bLouis\b'),
    'Naomi': re.compile(r'\bNaomi\b|\bNae\b'),
    'Diego': re.compile(r'\bDiego\b'),
    'Lila': re.compile(r'\bLila\b'),
    'Maddy': re.compile(r'\bMaddy\b'),
    'Hannah': re.compile(r'\bHannah\b'),
    'Theo': re.compile(r'\bTheo\b'),
    'Sara_Vance': re.compile(r'\bSara\b'),
    'Marcus': re.compile(r'\bMarcus\b'),
    'Sam_Voss': re.compile(r'\bSam\b'),
    'Rae': re.compile(r'\bRae\b'),
    'Will_Anderson': re.compile(r'\bWill (?:Anderson)?\b'),
    'Mr_Sanchez': re.compile(r'\bMr\.?\s*Sanchez\b'),
    'Mr_Pell': re.compile(r'\bMr\.?\s*Pell\b'),
    'Dr_Owens': re.compile(r'\bDr\.?\s*Owens\b'),
    'Dr_Patel': re.compile(r'\bDr\.?\s*Patel\b'),
    'Frances_starter': re.compile(r'\bFrances\b'),
    'Jordan': re.compile(r'\bJordan\b'),
    'Iris': re.compile(r'\bIris\b'),
    'Cole': re.compile(r'\bCole\b'),
    'Granbury': re.compile(r'\bGranbury\b'),
    'Adolphus': re.compile(r'\bAdolphus\b'),
    'BookPeople': re.compile(r'\bBookPeople\b'),
    'Threshold': re.compile(r'\bThreshold\b'),
    'Edinburgh': re.compile(r'\bEdinburgh\b'),
}

# Corpus-level frequency thresholds (warnings, not fails)
TROPE_WARN_PCT = 5.0    # any single trope in >5% of traces is over-rotation
ROSTER_WARN_PCT = 10.0  # any single named anchor in >10% is roster cycling
PER_TRACE_NAMED_WARN = 4  # >=4 distinct named entities in one trace is a warning


def count_words(text):
    return len(re.findall(r'\w+', text))


def check_assistant_text(text, strict=False, is_vision_trace=False, is_audio_trace=False):
    """Return list of HARD failure reasons for an assistant turn.

    is_vision_trace: when True (Domain I or DPO-VISION-REGISTER), additionally
    check for caption-mode register failures ("I see in the image", etc.).

    is_audio_trace: when True (Domain P or DPO-VOICE-REGISTER), additionally
    check for TTS-broken patterns (markdown, URLs, code blocks, abbreviations
    TTS mangles, all-caps emphasis) and audio-modality acknowledgment failures
    ("switching to voice mode", "thanks for calling", etc.).
    """
    failures = []
    if EM_DASH_RX.search(text):
        failures.append('EM_DASH')
    if THINK_BLOCK_RX.search(text):
        failures.append('THINK_BLOCK')
    m = SERVICE_RX.search(text)
    if m:
        snip = m.group(0)[:40]
        failures.append(f'SERVICE_PHRASE:{snip}')
    if STAGE_RX.search(text):
        failures.append('STAGE_DIRECTION')
    if K0_AI_VOCAB_RX.search(text):
        failures.append('K0_AI_VOCAB')
    if LISTICLE_RX.search(text):
        failures.append('LISTICLE')
    if GREETING_FORMULA_RX.search(text.strip()):
        failures.append('GREETING_FORMULA')
    if count_words(text) > K0_REPLY_WORDS_HARD_CAP:
        failures.append(f'OVERLONG:{count_words(text)}w')
    if is_vision_trace and VISION_REGISTER_RX.search(text):
        m2 = VISION_REGISTER_RX.search(text)
        snip = m2.group(0)[:50]
        failures.append(f'VISION_REGISTER:{snip}')
    if is_audio_trace:
        if TTS_MARKDOWN_RX.search(text):
            failures.append('TTS_MARKDOWN')
        if TTS_URL_RX.search(text):
            failures.append('TTS_URL')
        if TTS_CODE_BLOCK_RX.search(text):
            failures.append('TTS_CODE_BLOCK')
        if TTS_BAD_ABBREV_RX.search(text):
            m3 = TTS_BAD_ABBREV_RX.search(text)
            failures.append(f'TTS_BAD_ABBREV:{m3.group(0)}')
        if TTS_ALL_CAPS_EMPHASIS_RX.search(text):
            m4 = TTS_ALL_CAPS_EMPHASIS_RX.search(text)
            failures.append(f'TTS_ALL_CAPS:{m4.group(0)}')
        if AUDIO_MODALITY_RX.search(text):
            m5 = AUDIO_MODALITY_RX.search(text)
            snip = m5.group(0)[:50]
            failures.append(f'AUDIO_MODALITY_ACK:{snip}')
    return failures


def check_user_audio_brackets(text, is_audio_trace=False):
    """For audio traces, validate operator's bracket vocabulary.

    Returns failure list. Empty when bracket is well-formed or trace is not P-domain.
    """
    failures = []
    if not is_audio_trace:
        return failures
    if AUDIO_IDENTITY_IN_BRACKET_RX.search(text):
        m = AUDIO_IDENTITY_IN_BRACKET_RX.search(text)
        failures.append(f'AUDIO_IDENTITY_IN_BRACKET:{m.group(0)[:50]}')
    for bracket_match in AUDIO_BRACKET_DETECT_RX.findall(text):
        if not AUDIO_BRACKET_VALID_RX.match(bracket_match):
            failures.append(f'AUDIO_BRACKET_INVALID:{bracket_match[:50]}')
    return failures


def check_assistant_text_warnings(text):
    """Return list of soft warnings for an assistant turn."""
    warnings = []
    # Count tropes present
    trope_hits = [name for name, rx in TROPE_PATTERNS.items() if rx.search(text)]
    if len(trope_hits) >= 3:
        warnings.append(f'MULTI_TROPE:{",".join(trope_hits)}')
    # Count distinct named entities
    roster_hits = [name for name, rx in ROSTER_PATTERNS.items() if rx.search(text)]
    if len(roster_hits) >= PER_TRACE_NAMED_WARN:
        warnings.append(f'ROSTER_CROWDED:{len(roster_hits)}')
    return warnings


def count_sentences(text):
    text = text.strip()
    if not text:
        return 0
    sentences = re.findall(r'[^.!?]+[.!?]+', text)
    if not sentences:
        return 1
    return len(sentences)


def check_trace(trace):
    """Validate a single trace.

    Returns (hard_failures, warnings, stats) tuple.
    """
    hard = []
    warns = []
    stats = {
        'turns': 0,
        'assistant_turns': 0,
        'k0_word_counts': [],
        'k0_sentences': [],
        'tropes_per_trace': set(),
        'roster_per_trace': set(),
        'has_callback': False,
    }

    if 'messages' not in trace:
        if not ({'prompt', 'chosen', 'rejected'} <= set(trace.keys())):
            hard.append('MISSING_MESSAGES_OR_DPO_FIELDS')
            return hard, warns, stats

    msgs = trace.get('messages', [])

    if msgs and msgs[0].get('role') == 'system':
        hard.append('HAS_SYSTEM_PROMPT')

    if msgs and msgs[0].get('role') != 'user':
        hard.append('FIRST_TURN_NOT_USER')

    cat = trace.get('_cat', '')
    is_vision_trace = cat.startswith('I') or cat == 'DPO-VISION-REGISTER'
    is_audio_trace = cat.startswith('P') or cat == 'DPO-VOICE-REGISTER'

    user_turns = []
    assistant_turns = []
    user_turn_idx = 0
    for m in msgs:
        role = m.get('role')
        content = m.get('content', '')
        stats['turns'] += 1
        if role == 'user':
            user_turns.append(content)
            user_turn_idx += 1
            for f in check_user_audio_brackets(content, is_audio_trace=is_audio_trace):
                hard.append(f"USER_TURN_{user_turn_idx}:{f}")
        elif role == 'assistant':
            assistant_turns.append(content)
            stats['assistant_turns'] += 1
            stats['k0_word_counts'].append(count_words(content))
            stats['k0_sentences'].append(count_sentences(content))
            for f in check_assistant_text(content, is_vision_trace=is_vision_trace, is_audio_trace=is_audio_trace):
                hard.append(f"TURN_{stats['assistant_turns']}:{f}")
            for w in check_assistant_text_warnings(content):
                warns.append(f"TURN_{stats['assistant_turns']}:{w}")
            for trope_name, rx in TROPE_PATTERNS.items():
                if rx.search(content):
                    stats['tropes_per_trace'].add(trope_name)
            for name, rx in ROSTER_PATTERNS.items():
                if rx.search(content):
                    stats['roster_per_trace'].add(name)

    if trace.get('_type') == 'dpo':
        chosen = trace.get('chosen', '')
        rejected = trace.get('rejected', '')
        if not chosen:
            hard.append('DPO_MISSING_CHOSEN')
        if not rejected:
            hard.append('DPO_MISSING_REJECTED')
        for f in check_assistant_text(chosen, is_vision_trace=is_vision_trace, is_audio_trace=is_audio_trace):
            hard.append(f'DPO_CHOSEN:{f}')
        # Validate audio brackets in DPO user prompt for VOICE-REGISTER pairs
        if is_audio_trace and msgs:
            for m_idx, m in enumerate(msgs):
                if m.get('role') == 'user':
                    for f in check_user_audio_brackets(m.get('content', ''), is_audio_trace=True):
                        hard.append(f'DPO_USER:{f}')
        # Categories where the rejected legitimately contains the banned
        # pattern as the contrast (the WHOLE POINT of the DPO pair).
        # DPO-VISION-REGISTER: rejected contains caption-mode vision-register failure.
        # DPO-VOICE-REGISTER: rejected contains TTS-broken patterns (markdown / lists / etc).
        if cat not in ('DPO-EM-DASH', 'DPO-SERVICE-PHRASE', 'DPO-PERFORMANCE',
                       'DPO-BREVITY', 'DPO-TROPE', 'DPO-LORE-DUMP', 'DPO-NARRATOR',
                       'DPO-VISION-REGISTER', 'DPO-VOICE-REGISTER'):
            for f in check_assistant_text(rejected):
                hard.append(f'DPO_REJECTED:{f}')

    if cat.startswith('D') and stats['assistant_turns'] >= 2 and user_turns:
        first_user_words = set(re.findall(r'\b[a-z]{5,}\b', user_turns[0].lower()))
        last_asst_words = set(
            re.findall(r'\b[a-z]{5,}\b', assistant_turns[-1].lower())
        ) if assistant_turns else set()
        common = first_user_words & last_asst_words
        stopwords = {
            'about', 'after', 'again', 'before', 'being', 'between', 'could',
            'every', 'first', 'going', 'great', 'might', 'never', 'often',
            'other', 'right', 'should', 'still', 'their', 'there', 'these',
            'thing', 'think', 'those', 'today', 'where', 'which', 'while',
            'would', 'really', 'thats', 'something', 'because', 'people',
            'maybe', 'always', 'doesnt', 'wasnt'
        }
        distinctive = common - stopwords
        if distinctive:
            stats['has_callback'] = True

    return hard, warns, stats


def validate_file(path):
    summary = {
        'path': str(path),
        'total': 0,
        'passed': 0,
        'hard_failed': 0,
        'warned': 0,
        'hard_counts': defaultdict(int),
        'warn_counts': defaultdict(int),
        'cat_counts': defaultdict(int),
        'type_counts': defaultdict(int),
        'k0_word_buckets': defaultdict(int),
        'k0_sentence_buckets': defaultdict(int),
        'trope_per_trace_counts': defaultdict(int),
        'roster_per_trace_counts': defaultdict(int),
        'multiturn_total': 0,
        'multiturn_with_callback': 0,
    }
    failed_examples = []

    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            summary['total'] += 1
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                summary['hard_failed'] += 1
                summary['hard_counts']['JSON_PARSE_ERROR'] += 1
                continue

            hard, warns, stats = check_trace(trace)

            cat = trace.get('_cat', 'UNTAGGED')
            ttype = trace.get('_type', 'UNTAGGED')
            summary['cat_counts'][cat] += 1
            summary['type_counts'][ttype] += 1

            for w in stats['k0_word_counts']:
                if w <= 5:
                    summary['k0_word_buckets']['<=5'] += 1
                elif w <= 20:
                    summary['k0_word_buckets']['6-20'] += 1
                elif w <= 50:
                    summary['k0_word_buckets']['21-50'] += 1
                elif w <= 100:
                    summary['k0_word_buckets']['51-100'] += 1
                else:
                    summary['k0_word_buckets']['101+'] += 1

            for s in stats['k0_sentences']:
                if s <= 1:
                    summary['k0_sentence_buckets']['1'] += 1
                elif s <= 3:
                    summary['k0_sentence_buckets']['2-3'] += 1
                elif s <= 7:
                    summary['k0_sentence_buckets']['4-7'] += 1
                else:
                    summary['k0_sentence_buckets']['8+'] += 1

            for t in stats['tropes_per_trace']:
                summary['trope_per_trace_counts'][t] += 1
            for r in stats['roster_per_trace']:
                summary['roster_per_trace_counts'][r] += 1

            if cat.startswith('D') and ttype == 'multi':
                summary['multiturn_total'] += 1
                if stats['has_callback']:
                    summary['multiturn_with_callback'] += 1

            if hard:
                summary['hard_failed'] += 1
                for f in hard:
                    parts = f.split(':')
                    if parts[0].startswith('TURN_') or parts[0].startswith('DPO_'):
                        ftype = parts[1] if len(parts) > 1 else parts[0]
                    else:
                        ftype = parts[0]
                    summary['hard_counts'][ftype] += 1
                if len(failed_examples) < 20:
                    failed_examples.append((lineno, hard, warns))
            else:
                summary['passed'] += 1

            if warns:
                summary['warned'] += 1
                for w in warns:
                    parts = w.split(':')
                    if parts[0].startswith('TURN_'):
                        wtype = parts[1] if len(parts) > 1 else parts[0]
                    else:
                        wtype = parts[0]
                    summary['warn_counts'][wtype] += 1

    summary['failed_examples'] = failed_examples
    return summary


def print_report(summary, strict=False):
    p = summary
    print(f"\n=========================================================")
    print(f"  {p['path']}")
    print(f"=========================================================")
    print(f"  Total traces: {p['total']}")
    print(f"  Passed (hard checks): {p['passed']}")
    print(f"  Hard failed: {p['hard_failed']}")
    print(f"  Warned (soft signals): {p['warned']}")

    if p['total'] == 0:
        return

    print(f"\n--- Hard Reject Breakdown ---")
    if p['hard_counts']:
        for k, v in sorted(p['hard_counts'].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    else:
        print("  (none)")

    print(f"\n--- Soft Warning Breakdown ---")
    if p['warn_counts']:
        for k, v in sorted(p['warn_counts'].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    else:
        print("  (none)")

    print(f"\n--- Categories ---")
    for k, v in sorted(p['cat_counts'].items()):
        print(f"  {k}: {v}")

    print(f"\n--- Types ---")
    for k, v in sorted(p['type_counts'].items()):
        print(f"  {k}: {v}")

    print(f"\n--- K0 Reply Word Distribution ---")
    word_total = sum(p['k0_word_buckets'].values())
    for bucket in ['<=5', '6-20', '21-50', '51-100', '101+']:
        n = p['k0_word_buckets'].get(bucket, 0)
        pct = 100 * n / word_total if word_total else 0
        print(f"  {bucket} words: {n} ({pct:.1f}%)")

    print(f"\n--- K0 Reply Sentence Distribution ---")
    sent_total = sum(p['k0_sentence_buckets'].values())
    for bucket in ['1', '2-3', '4-7', '8+']:
        n = p['k0_sentence_buckets'].get(bucket, 0)
        pct = 100 * n / sent_total if sent_total else 0
        print(f"  {bucket} sentences: {n} ({pct:.1f}%)")
    short_total = p['k0_sentence_buckets'].get('1', 0) + p['k0_sentence_buckets'].get('2-3', 0)
    short_pct = 100 * short_total / sent_total if sent_total else 0
    print(f"  Short (<=3 sentences): {short_pct:.1f}% (target ~60% for K0)")

    if p['multiturn_total']:
        cb_pct = 100 * p['multiturn_with_callback'] / p['multiturn_total']
        print(f"\n--- Domain D Callback Density ---")
        print(f"  {p['multiturn_with_callback']} / {p['multiturn_total']} = {cb_pct:.1f}% (heuristic)")

    print(f"\n--- TROPE FREQUENCY (corpus-level rotation check) ---")
    print(f"  Threshold for warning: > {TROPE_WARN_PCT}% of traces")
    if not p['trope_per_trace_counts']:
        print("  (no tropes detected)")
    else:
        any_overused = False
        for trope, count in sorted(p['trope_per_trace_counts'].items(), key=lambda x: -x[1]):
            pct = 100 * count / p['total']
            flag = ' [OVER-ROTATED]' if pct > TROPE_WARN_PCT else ''
            print(f"  {trope}: {count} traces ({pct:.1f}%){flag}")
            if pct > TROPE_WARN_PCT:
                any_overused = True
        if any_overused:
            print(f"  ! Some tropes are above the rotation threshold. Trace gen should diversify.")

    print(f"\n--- NAMED ROSTER FREQUENCY (roster cycling check) ---")
    print(f"  Threshold for warning: > {ROSTER_WARN_PCT}% of traces")
    if not p['roster_per_trace_counts']:
        print("  (no named-roster mentions detected)")
    else:
        sorted_roster = sorted(p['roster_per_trace_counts'].items(), key=lambda x: -x[1])
        any_cycled = False
        for name, count in sorted_roster[:20]:
            pct = 100 * count / p['total']
            flag = ' [ROSTER-CYCLED]' if pct > ROSTER_WARN_PCT else ''
            print(f"  {name}: {count} traces ({pct:.1f}%){flag}")
            if pct > ROSTER_WARN_PCT:
                any_cycled = True
        if any_cycled:
            print(f"  ! Some named anchors appear in too many traces. Rotate them.")

    if p['failed_examples']:
        print(f"\n--- First Hard Failures ---")
        for lineno, hard, warns in p['failed_examples'][:5]:
            print(f"  Line {lineno}: {hard[:3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='JSONL file(s) or directory')
    ap.add_argument('--strict', action='store_true',
                    help='Promote soft warnings to hard fails (use sparingly)')
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        pth = Path(p)
        if pth.is_dir():
            paths.extend(sorted(pth.glob('*.jsonl')))
        elif pth.is_file():
            paths.append(pth)
        else:
            print(f"Not found: {p}", file=sys.stderr)

    if not paths:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    grand = {
        'path': 'GRAND TOTAL',
        'total': 0, 'passed': 0, 'hard_failed': 0, 'warned': 0,
        'hard_counts': defaultdict(int),
        'warn_counts': defaultdict(int),
        'cat_counts': defaultdict(int),
        'type_counts': defaultdict(int),
        'k0_word_buckets': defaultdict(int),
        'k0_sentence_buckets': defaultdict(int),
        'trope_per_trace_counts': defaultdict(int),
        'roster_per_trace_counts': defaultdict(int),
        'multiturn_total': 0,
        'multiturn_with_callback': 0,
        'failed_examples': [],
    }

    for p in paths:
        s = validate_file(p)
        print_report(s, strict=args.strict)
        for k in ['total', 'passed', 'hard_failed', 'warned',
                  'multiturn_total', 'multiturn_with_callback']:
            grand[k] += s[k]
        for d_key in ['hard_counts', 'warn_counts', 'cat_counts', 'type_counts',
                      'k0_word_buckets', 'k0_sentence_buckets',
                      'trope_per_trace_counts', 'roster_per_trace_counts']:
            for k, v in s[d_key].items():
                grand[d_key][k] += v

    if len(paths) > 1:
        print_report(grand, strict=args.strict)

    if args.strict:
        sys.exit(0 if (grand['hard_failed'] == 0 and grand['warned'] == 0) else 1)
    sys.exit(0 if grand['hard_failed'] == 0 else 1)


if __name__ == '__main__':
    main()
