"""How deeply to teach: the level of explanation a request asked for.

Difficulty describes the *explanation*, never the person. It says how much
context to give, how much jargon is fair and how large a step may be - it is not
a judgement about the learner, and it is never inferred from who someone appears
to be. A technically demanding question can still be answered for a beginner, so
an advanced-sounding topic on its own changes nothing.

The level comes from the words of the request itself and nothing else, which
means it costs no extra model call: the wording is matched locally and the result
is handed to the prompts that already exist. When the request says nothing about
depth, the level is the neutral middle one.

Sibling modules
---------------
``app.teaching.followup`` uses :func:`adjust_difficulty` when the learner asks
for a simpler or deeper explanation of the same material.
"""

import re
from enum import Enum


class TeachingDifficulty(str, Enum):
    """Depth of explanation for one teaching plan."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


DEFAULT_DIFFICULTY = TeachingDifficulty.INTERMEDIATE

# From easiest to hardest, which is also the order :func:`adjust_difficulty`
# walks. There are deliberately only three levels.
_LEVELS = (
    TeachingDifficulty.BEGINNER,
    TeachingDifficulty.INTERMEDIATE,
    TeachingDifficulty.ADVANCED,
)

# An explicit refusal to simplify is the one instruction that beats a request for
# simplicity, so it is checked before anything else ("don't oversimplify").
_NEVER_SIMPLIFY = re.compile(
    r"\bdon'?t (oversimplify|simplify|dumb it down|talk down)\b"
    r"|\bskip (the|all) basics\b"
    r"|\bno need to explain the basics\b"
    r"|\bnot (too|over) ?simpl(e|ified)\b"
    r"|\bdon'?t keep it simple\b",
    re.IGNORECASE,
)

_BEGINNER = re.compile(
    r"\bbeginner\b"
    r"|\blike i'?m (a |an |totally )?(beginner|new|stupid|five|5)\b"
    r"|\bsimpl(e|er|est|ify|ified|y)\b"
    r"|\bin (plain|simple) (english|terms|language|words)\b"
    r"|\bkeep it simple\b"
    r"|\bno jargon\b"
    r"|\bwithout jargon\b"
    r"|\beli5\b"
    r"|\blayman'?s?\b"
    r"|\bi'?m new to\b"
    r"|\bnew to (this|that|it|the topic)\b"
    r"|\bbasic explanation\b"
    r"|\bin basic (terms|language)\b"
    r"|\bassume i know nothing\b"
    r"|\bbaby steps\b"
    r"|\bfor dummies\b"
    r"|\bbreak it down\b",
    re.IGNORECASE,
)

_ADVANCED = re.compile(
    r"\badvanced\b"
    r"|\btechnical (details?|explanation|version|level|depth)\b"
    r"|\bdetailed technical\b"
    r"|\bimplementation details?\b"
    r"|\bthe internals?\b"
    r"|\bunder the hood\b"
    r"|\bin[- ]depth\b"
    r"|\bdeep dive\b"
    r"|\bgo deeper\b"
    r"|\bmore (technical|detail)\b"
    r"|\btechnical(ly)? accurate\b"
    r"|\bexpert\b"
    r"|\bassume i (already )?(know|understand)\b"
    r"|\bdon'?t explain the basics\b",
    re.IGNORECASE,
)

_INTERMEDIATE = re.compile(
    r"\bnormal(ly)?\b"
    r"|\bstandard (explanation|level|way)\b"
    r"|\bneither too simple nor too deep\b"
    r"|\busual (level|amount)\b"
    r"|\bnot too (deep|detailed|simple)\b",
    re.IGNORECASE,
)

_GUIDANCE = {
    TeachingDifficulty.BEGINNER: (
        "Teach this for someone new to the topic. Use plain language and no unnecessary "
        "jargon; when a term is unavoidable, introduce it before you rely on it. Explain "
        "why something works before the details of how it works. Keep every step small - "
        "one idea at a time - use concrete examples, and never assume prior knowledge."
    ),
    TeachingDifficulty.INTERMEDIATE: (
        "Teach this at a normal technical level. Use standard terminology without "
        "over-explaining it, bring out the relationships that matter, and assume the "
        "basics are familiar. Keep the sequence concise and balanced, with a moderate "
        "amount of detail per step."
    ),
    TeachingDifficulty.ADVANCED: (
        "Teach this for someone who already knows the fundamentals. Use precise technical "
        "terminology, go straight to the details and the relationships between them, and "
        "skip introductory framing. More may be said in each step - but stay on the "
        "question that was asked instead of giving an exhaustive lecture."
    ),
}


def detect_difficulty(request, default=DEFAULT_DIFFICULTY):
    """Return the level of explanation ``request`` asks for.

    Only the wording of the request is read, so the same question is always
    taught at the same level and nothing about the learner is guessed at. Pass
    ``default=None`` to find out whether the request mentioned depth at all.
    """
    text = str(request or "")

    if _NEVER_SIMPLIFY.search(text):
        return TeachingDifficulty.ADVANCED
    if _BEGINNER.search(text):
        return TeachingDifficulty.BEGINNER
    if _ADVANCED.search(text):
        return TeachingDifficulty.ADVANCED
    if _INTERMEDIATE.search(text):
        return TeachingDifficulty.INTERMEDIATE
    return default


def difficulty_guidance(difficulty):
    """Return the teaching instruction for ``difficulty``, written for a prompt.

    This is the single source of how each level is taught, so every prompt that
    needs it says the same thing.
    """
    level = difficulty or DEFAULT_DIFFICULTY
    return _GUIDANCE.get(level, _GUIDANCE[DEFAULT_DIFFICULTY])


def adjust_difficulty(difficulty, steps):
    """Return ``difficulty`` moved ``steps`` levels, staying inside the range.

    Moving one level at a time keeps adaptation bounded: asking for a simpler
    explanation goes down a level, never straight to the floor.
    """
    try:
        index = _LEVELS.index(difficulty)
    except ValueError:
        index = _LEVELS.index(DEFAULT_DIFFICULTY)

    moved = min(max(index + int(steps), 0), len(_LEVELS) - 1)
    return _LEVELS[moved]


__all__ = [
    "DEFAULT_DIFFICULTY",
    "TeachingDifficulty",
    "adjust_difficulty",
    "detect_difficulty",
    "difficulty_guidance",
]
