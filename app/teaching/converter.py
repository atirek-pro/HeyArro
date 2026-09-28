"""Bridge from the existing response contract to the teaching plan.

The vision provider answers with a ``HeyArroResponse``: one overall answer plus a
flat list of steps carrying visual actions. The teaching architecture wants a
``TeachingPlan``: an objective, an introduction, ordered steps with their own
objectives, and a conclusion. This module converts one into the other.

The conversion is deterministic and offline. It never calls a model, refines or
grounds a target, speaks, draws, captures a screen, or changes any existing
behaviour - it is a compatibility bridge that later steps can build on.

Mapping
-------
``response.text``
    The natural opening, so it becomes ``introduction``, and it is the basis of
    ``objective`` ("Understand: <text>"): the response has no explicit learning
    objective, and framing its own answer is better than inventing one.
``teaching.steps[i]``
    Becomes plan step ``i`` with ``order`` ``i + 1`` and ``step_id``
    ``"step_<i+1>"``, in the order they were given. Nothing is reordered,
    renumbered or added.
``step.instruction``
    Becomes both ``objective`` and ``explanation``. The response has no separate
    objective field, so the sentence the learner hears is also the best
    statement of what that step is for; a plan produced by the teaching-plan
    generator has richer objectives.
``step.visual_actions``
    Carried over untouched - same objects, same coordinates. Grounding refines
    them later.
``transition`` and ``conclusion``
    Always ``None``. The response contract has nothing to preserve here, and
    inventing filler ("Now let's move on") would be worse than having none.
``difficulty``
    Comes from the caller, who knows what the learner asked for. The response
    carries no difficulty of its own, so the neutral middle level is used when
    none is given.
"""

from app.llm.response import HeyArroResponse
from app.teaching.difficulty import DEFAULT_DIFFICULTY
from app.teaching.plan import TeachingPlan, TeachingStep

# The response carries no explicit objective, so its answer is framed as one.
# Kept in one place so the rule is easy to see and to change.
_OBJECTIVE_PREFIX = "Understand: "


class TeachingPlanConversionError(RuntimeError):
    """Raised when a response cannot be converted into a teaching plan."""


def _text(value) -> str:
    return str(value or "").strip()


def response_to_teaching_plan(response: HeyArroResponse, difficulty=None) -> TeachingPlan:
    """Convert a validated ``HeyArroResponse`` into a ``TeachingPlan``.

    Raises TeachingPlanConversionError when the response has no teaching steps,
    or no text at all to teach from - a plan is never faked into existence.

    ``difficulty`` is the level the learner asked for; when it is not given the
    plan carries the neutral one.
    """
    steps = list(response.teaching.steps)
    if not steps:
        raise TeachingPlanConversionError(
            "Cannot build a teaching plan: the response contains no teaching steps"
        )

    answer = _text(response.response.text)
    instructions = [_text(step.instruction) for step in steps]

    # What the lesson is about: the response's own words, or failing that the
    # first thing it asks the learner to look at.
    subject = answer or next((text for text in instructions if text), "")
    if not subject:
        raise TeachingPlanConversionError(
            "Cannot build a teaching plan: the response contains no text to teach from"
        )

    plan_steps = []
    for order, (step, instruction) in enumerate(zip(steps, instructions), start=1):
        plan_steps.append(
            TeachingStep(
                step_id=f"step_{order}",
                order=order,
                objective=instruction or subject,
                # A blank instruction would leave nothing to say, so the overall
                # answer stands in for it.
                explanation=instruction or answer or subject,
                visual_actions=list(step.visual_actions),
                transition=None,
            )
        )

    return TeachingPlan(
        objective=f"{_OBJECTIVE_PREFIX}{subject}",
        difficulty=difficulty or DEFAULT_DIFFICULTY,
        introduction=answer or subject,
        steps=plan_steps,
        conclusion=None,
    )


def answer_only_teaching_plan(text: str, difficulty=None) -> TeachingPlan:
    """Build the simplest valid plan for an answer with nothing to teach step by step.

    A direct answer carries no teaching steps, so there is no plan to convert.
    Rather than speak nothing, the answer becomes the introduction *and* the
    single step: the sequence speaks each sentence once, so it is heard once and
    nothing is shown. Like the conversion above, this is deterministic and adds
    no content of its own.
    """
    answer = _text(text)
    if not answer:
        raise TeachingPlanConversionError(
            "Cannot build a teaching plan: there is no text to say"
        )

    return TeachingPlan(
        objective=f"{_OBJECTIVE_PREFIX}{answer}",
        difficulty=difficulty or DEFAULT_DIFFICULTY,
        introduction=answer,
        steps=[
            TeachingStep(
                step_id="step_1",
                order=1,
                objective=answer,
                explanation=answer,
                visual_actions=[],
                transition=None,
            )
        ],
        conclusion=None,
    )


__all__ = [
    "TeachingPlanConversionError",
    "answer_only_teaching_plan",
    "response_to_teaching_plan",
]
