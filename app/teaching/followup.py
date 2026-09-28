"""What the learner asks for after a lesson, as a small bounded vocabulary.

A follow-up continues the teaching interaction that just finished: "explain that
again", "make it simpler", "I don't understand step 2". It is recognised from the
words the learner used, so it costs no extra model call and never depends on
guessing how someone feels or how well they understood.

Recognising one changes nothing about the lesson already taught. A follow-up
produces a *new* plan, which is then prepared (grounded in full) and executed
like any other lesson, so a plan is never mutated while it is being taught.

Nothing here is autonomous: a follow-up only happens because the learner asked
for it, and the context sent with it is the immediate lesson, kept short on
purpose - there is no memory of earlier sessions.
"""

import re
from dataclasses import dataclass
from enum import Enum

from app.teaching.difficulty import (
    DEFAULT_DIFFICULTY,
    TeachingDifficulty,
    adjust_difficulty,
    detect_difficulty,
)

# No follow-up is longer than a sentence; the context sent to the generator is
# capped so a long lesson cannot grow the request without limit.
_CONTEXT_LIMIT = 2000
_LINE_LIMIT = 240


class TeachingFollowUpType(str, Enum):
    """What the learner is asking for after a lesson."""

    REPEAT = "repeat"
    SIMPLIFY = "simplify"
    DEEPEN = "deepen"
    EXAMPLE = "example"
    EXPLAIN_STEP = "explain_step"
    SUMMARY = "summary"
    CONTINUE = "continue"


# Follow-ups answered in words alone. Everything else may need to be shown, so it
# is planned against the current screen and grounded against it.
_SPOKEN_ONLY = (TeachingFollowUpType.REPEAT, TeachingFollowUpType.SUMMARY)

_STEP_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_STEP_REFERENCE = re.compile(
    r"\bstep\s*(?:number\s*)?(?P<step>\d{1,2}|"
    + "|".join(_STEP_WORDS)
    + r")\b"
    r"|\b(?P<ordinal>"
    + "|".join(_ORDINALS)
    + r")\s+step\b",
    re.IGNORECASE,
)

# Ordered by specificity: the first rule that matches wins, so "explain step 2
# again" is about step 2 rather than a plain repeat.
_RULES = (
    (
        TeachingFollowUpType.SUMMARY,
        re.compile(
            r"\bsummari[sz]e\b|\bsummary\b|\brecap\b|\btl;?dr\b"
            r"|\bmain (idea|point|takeaway|thing)\b|\bkey takeaway\b"
            r"|\bwhat should i (remember|take away|know)\b|\bin a nutshell\b"
            r"|\bwhat was the (point|idea)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TeachingFollowUpType.SIMPLIFY,
        re.compile(
            r"\bsimpl(e|er|est|ify|ified)\b|\bmore simply\b|\bexplain it easier\b"
            r"|\bdumb it down\b|\beasier to (understand|follow)\b"
            r"|\bplain (english|terms)\b|\bmore basic\b|\bless technical\b"
            r"|\bexplain (it|that|this) like i'?m (five|5|a beginner|new)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TeachingFollowUpType.DEEPEN,
        re.compile(
            r"\bgo(ing)? deeper\b|\bdeeper\b|\bmore detail(s|ed)?\b"
            r"|\bin[- ]depth\b|\belaborate\b|\btell me more\b|\bexpand on\b"
            r"|\bmore technical\b|\btechnical (version|details?)\b"
            r"|\bwhy does (that|it|this) work\b|\bwhy (is that|does that)\b"
            r"|\bhow does (that|it|this) work\b|\bthe (full|whole) picture\b",
            re.IGNORECASE,
        ),
    ),
    (
        TeachingFollowUpType.EXAMPLE,
        re.compile(
            r"\banother example\b|\b(an|one more|a new|different) example\b"
            r"|\bshow me an example\b|\bexample of (that|this|it)\b"
            r"|\bdemonstrate\b|\bshow me (that|this|it) (on|in) the screen\b"
            r"|\bwhere (is|are) (that|those|it) on the screen\b"
            r"|\bshow me where\b",
            re.IGNORECASE,
        ),
    ),
    (
        TeachingFollowUpType.REPEAT,
        re.compile(
            r"\b(explain|say|tell|show)( me)?( it| that| this)? again\b"
            r"|\brepeat (that|it|this)\b|\bone more time\b|\bagain please\b"
            r"|\bwhat did you say\b|\bcome again\b|\bsay that (once )?more\b"
            r"|\bthat (again|once more)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TeachingFollowUpType.CONTINUE,
        re.compile(
            r"^(continue|keep going|go on|carry on)\b|\bwhat'?s next\b"
            r"|\bnext (step|part|bit|concept)\b|\bkeep teaching\b|\band then\b"
            r"|\bgo ahead\b|\bmore please\b",
            re.IGNORECASE,
        ),
    ),
)

_GUIDANCE = {
    TeachingFollowUpType.REPEAT: (
        "The learner wants the same idea again. Say it again in slightly different, "
        "clearer words - do not replay the sentence word for word, and keep the same "
        "objective and the same level."
    ),
    TeachingFollowUpType.SIMPLIFY: (
        "The learner found this too hard. Explain the same objective more simply: plainer "
        "language, smaller steps, a concrete comparison, and no jargon that has not just "
        "been explained."
    ),
    TeachingFollowUpType.DEEPEN: (
        "The learner wants more depth. Keep the same objective, but add the technical "
        "detail, the relationships behind it and what follows from it. Do not repeat the "
        "basics."
    ),
    TeachingFollowUpType.EXAMPLE: (
        "The learner wants another example. Teach the same objective again through a "
        "different, concrete example - preferably something visible on screen now."
    ),
    TeachingFollowUpType.EXPLAIN_STEP: (
        "The learner lost the thread part-way through. Teach only that step's objective "
        "again, from the beginning and concretely. Do not re-teach the rest of the lesson."
    ),
    TeachingFollowUpType.SUMMARY: (
        "The learner wants the takeaway, not the lesson again. One or two steps at most: "
        "the central idea and what is worth remembering. Introduce nothing new."
    ),
    TeachingFollowUpType.CONTINUE: (
        "The lesson finished. Continue from where it stopped with the next idea that "
        "follows, without repeating what has already been taught."
    ),
}


@dataclass(frozen=True)
class TeachingFollowUp:
    """One recognised follow-up request."""

    type: TeachingFollowUpType
    text: str
    step: int | None = None

    @property
    def needs_screenshot(self):
        """True when the follow-up may need to be shown, not just said.

        A recap or a repeat is answered in words; anything else may refer to what
        is on screen now, so it is planned and grounded against the current
        screen.
        """
        return self.type not in _SPOKEN_ONLY


def detect_follow_up(text, has_previous_plan=True):
    """Return the follow-up ``text`` asks for, or None when it asks for something new.

    A follow-up only exists when a lesson was actually just taught, so without
    one this is always None.
    """
    if not has_previous_plan:
        return None

    request = str(text or "").strip()
    if not request:
        return None

    match = _STEP_REFERENCE.search(request)
    if match is not None:
        return TeachingFollowUp(
            TeachingFollowUpType.EXPLAIN_STEP, request, _step_number(match)
        )

    for follow_up_type, pattern in _RULES:
        if pattern.search(request):
            return TeachingFollowUp(follow_up_type, request)

    return None


def follow_up_guidance(follow_up):
    """Return the teaching instruction for ``follow_up``, written for a prompt."""
    if follow_up is None:
        return ""
    return _GUIDANCE.get(follow_up.type, "")


def follow_up_difficulty(difficulty, follow_up, request=""):
    """Return the level a follow-up should be taught at.

    An explicit instruction in the follow-up itself always wins. Otherwise asking
    to simplify or to go deeper moves exactly one level, and every other
    follow-up stays where the lesson was. The level is never changed on the
    learner's behalf.
    """
    explicit = detect_difficulty(request, default=None)
    if explicit is not None:
        return explicit

    level = difficulty or DEFAULT_DIFFICULTY
    if follow_up is None:
        return level
    if follow_up.type is TeachingFollowUpType.SIMPLIFY:
        return adjust_difficulty(level, -1)
    if follow_up.type is TeachingFollowUpType.DEEPEN:
        return adjust_difficulty(level, 1)
    return level


def build_follow_up_context(question, plan, follow_up, limit=_CONTEXT_LIMIT):
    """Describe the lesson just taught, so a follow-up can continue it.

    Only the immediate lesson travels: the original question, what the lesson was
    meant to teach, the level it was taught at, the steps in the order the
    learner heard them, and which step they are asking about. The result is
    length-capped, so one lesson cannot grow the request without bound.
    """
    steps = list(plan.steps)
    lines = [
        "The lesson that just finished:",
        f"- the learner's original question: {_shorten(question) or '(not recorded)'}",
        f"- what it was meant to teach: {_shorten(plan.objective)}",
        f"- the level it was written at: {plan.difficulty.value}",
        "- its steps, numbered in the order the learner heard them:",
    ]

    for index, step in enumerate(steps, start=1):
        lines.append(f"  {index}. {_shorten(step.explanation)}")

    if plan.conclusion:
        lines.append(f"- how it closed: {_shorten(plan.conclusion)}")

    if follow_up is not None and follow_up.step is not None:
        lines.append(_describe_asked_step(steps, follow_up.step))

    lines.append(
        f"- the lesson is finished; the learner now says: \"{_shorten(follow_up.text)}\""
    )
    return _shorten_text("\n".join(lines), limit)


def _describe_asked_step(steps, number):
    if 1 <= number <= len(steps):
        return f"- they are asking about step {number}: {_shorten(steps[number - 1].explanation)}"
    return f"- they refer to step {number}, which this lesson does not have"


def _step_number(match):
    word = match.group("step")
    if word is not None:
        return int(word) if word.isdigit() else _STEP_WORDS[word.lower()]
    return _ORDINALS[match.group("ordinal").lower()]


def _shorten(text):
    return _shorten_text(str(text or "").replace("\n", " ").strip(), _LINE_LIMIT)


def _shorten_text(text, limit):
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "\u2026"


__all__ = [
    "TeachingFollowUp",
    "TeachingFollowUpType",
    "build_follow_up_context",
    "detect_follow_up",
    "follow_up_difficulty",
    "follow_up_guidance",
]
