"""Choosing how deeply to teach, from the words of the request alone.

The level is about the explanation, never about the person: a hard question does
not make an advanced learner, and nothing about the user is inferred. These tests
pin the wording that decides the level, the guidance each level produces, and the
fact that the decision reads nothing but the request.
"""

import inspect

import pytest

from app.llm.provider import VisionLLMRequest, build_user_prompt
from app.teaching.difficulty import (
    DEFAULT_DIFFICULTY,
    TeachingDifficulty,
    adjust_difficulty,
    detect_difficulty,
    difficulty_guidance,
)
from app.teaching.plan_prompt import (
    TEACHING_PLAN_SYSTEM_INSTRUCTION,
    build_teaching_plan_prompt,
)

# --- the levels -------------------------------------------------------------


def test_there_are_exactly_three_levels():
    assert [level.value for level in TeachingDifficulty] == [
        "beginner",
        "intermediate",
        "advanced",
    ]


def test_the_default_level_is_intermediate():
    assert DEFAULT_DIFFICULTY is TeachingDifficulty.INTERMEDIATE


# --- explicit beginner requests ---------------------------------------------


@pytest.mark.parametrize(
    "wording",
    [
        "Explain this like I'm a beginner.",
        "Explain this simply.",
        "Can you put that in simple terms?",
        "I'm new to this, keep it simple.",
        "ELI5 how this works.",
        "Break it down for me, no jargon please.",
    ],
)
def test_an_explicit_beginner_request_asks_for_the_beginner_level(wording):
    assert detect_difficulty(wording) is TeachingDifficulty.BEGINNER


# --- explicit advanced requests ---------------------------------------------


@pytest.mark.parametrize(
    "wording",
    [
        "Give me a detailed technical explanation.",
        "Give me the technical implementation details.",
        "Explain the implementation details of this.",
        "Go deeper, explain what happens under the hood.",
        "Don't oversimplify this one.",
    ],
)
def test_an_explicit_advanced_request_asks_for_the_advanced_level(wording):
    assert detect_difficulty(wording) is TeachingDifficulty.ADVANCED


# --- ordinary requests ------------------------------------------------------


@pytest.mark.parametrize(
    "wording",
    [
        "Explain this normally.",
        "What is this?",
        "How do I export a file?",
        "Why is that button greyed out?",
    ],
)
def test_an_ordinary_request_asks_for_the_intermediate_level(wording):
    assert detect_difficulty(wording) is TeachingDifficulty.INTERMEDIATE


def test_a_request_that_says_nothing_about_depth_has_no_explicit_level():
    assert detect_difficulty("What is this?", default=None) is None
    assert detect_difficulty("How do I export a file?", default=None) is None


def test_a_stated_level_is_explicit_and_a_normal_one_is_stated_too():
    assert detect_difficulty("Explain this simply.", default=None) is TeachingDifficulty.BEGINNER
    assert (
        detect_difficulty("Explain this normally.", default=None)
        is TeachingDifficulty.INTERMEDIATE
    )


# --- it is about the explanation, not the person ----------------------------


def test_a_technical_question_does_not_imply_an_advanced_learner():
    """The topic is not a signal: only what the learner asked for is."""
    question = "Explain how the garbage collector's write barrier works."

    assert detect_difficulty(question) is TeachingDifficulty.INTERMEDIATE
    assert detect_difficulty(question) == detect_difficulty("What is this?")


def test_the_same_topic_can_be_asked_for_at_every_level():
    topic = "Explain how the garbage collector's write barrier works"

    assert detect_difficulty(f"{topic} like I'm a beginner.") is TeachingDifficulty.BEGINNER
    assert detect_difficulty(topic) is TeachingDifficulty.INTERMEDIATE
    assert detect_difficulty(f"{topic}, and skip the basics.") is TeachingDifficulty.ADVANCED


def test_the_decision_is_deterministic():
    request = "Explain this simply, like I'm new to it."

    assert detect_difficulty(request) == detect_difficulty(request)


def test_the_decision_reads_nothing_but_the_request():
    """No other input can reach the decision: the signature is the whole story."""
    assert list(inspect.signature(detect_difficulty).parameters) == ["request", "default"]


def test_a_missing_request_is_neutral_rather_than_an_error():
    assert detect_difficulty(None) is TeachingDifficulty.INTERMEDIATE
    assert detect_difficulty("   ") is TeachingDifficulty.INTERMEDIATE


# --- guidance ---------------------------------------------------------------


def test_every_level_has_guidance_of_its_own():
    beginner = difficulty_guidance(TeachingDifficulty.BEGINNER)
    intermediate = difficulty_guidance(TeachingDifficulty.INTERMEDIATE)
    advanced = difficulty_guidance(TeachingDifficulty.ADVANCED)

    assert len({beginner, intermediate, advanced}) == 3
    assert all(guidance.strip() for guidance in (beginner, intermediate, advanced))


def test_the_beginner_guidance_asks_for_plain_language_and_small_steps():
    guidance = difficulty_guidance(TeachingDifficulty.BEGINNER).lower()

    assert "jargon" in guidance
    assert "small" in guidance
    assert "prior knowledge" in guidance


def test_the_advanced_guidance_asks_for_detail_without_the_basics():
    guidance = difficulty_guidance(TeachingDifficulty.ADVANCED).lower()

    assert "terminology" in guidance
    assert "skip" in guidance or "straight to" in guidance
    assert "lecture" in guidance  # stays on the question asked


def test_a_missing_level_falls_back_to_the_intermediate_guidance():
    assert difficulty_guidance(None) == difficulty_guidance(TeachingDifficulty.INTERMEDIATE)


# --- moving one level -------------------------------------------------------


def test_adjusting_a_level_moves_exactly_one_step():
    assert adjust_difficulty(TeachingDifficulty.INTERMEDIATE, -1) is TeachingDifficulty.BEGINNER
    assert adjust_difficulty(TeachingDifficulty.INTERMEDIATE, 1) is TeachingDifficulty.ADVANCED


def test_adjusting_a_level_never_leaves_the_range():
    assert adjust_difficulty(TeachingDifficulty.BEGINNER, -1) is TeachingDifficulty.BEGINNER
    assert adjust_difficulty(TeachingDifficulty.ADVANCED, 1) is TeachingDifficulty.ADVANCED
    assert adjust_difficulty(TeachingDifficulty.BEGINNER, 10) is TeachingDifficulty.ADVANCED


def test_an_unknown_level_is_treated_as_the_default():
    assert adjust_difficulty(None, 0) is TeachingDifficulty.INTERMEDIATE


# --- the prompts receive it -------------------------------------------------


def test_the_plan_prompt_carries_the_guidance():
    prompt = build_teaching_plan_prompt(
        "explain this", True, (1280, 720), guidance=difficulty_guidance(TeachingDifficulty.BEGINNER)
    )

    assert "How to teach this request:" in prompt
    assert "plain language" in prompt


def test_the_plan_prompt_has_no_guidance_block_by_default():
    assert "How to teach this request:" not in build_teaching_plan_prompt("explain this", True)


def test_the_vision_prompt_carries_the_guidance():
    prompt = build_user_prompt("what is this?", True, None, "Teach it plainly.")

    assert "How to teach this:" in prompt
    assert "Teach it plainly." in prompt


def test_the_request_carries_the_guidance_to_the_provider():
    request = VisionLLMRequest(transcript="what is this?", guidance="Teach it plainly.")

    assert request.guidance == "Teach it plainly."
    # Always safe to leave out: the provider then answers in its usual way.
    assert VisionLLMRequest(transcript="what is this?").guidance == ""


def test_the_teaching_prompt_tells_the_model_the_level_is_never_about_the_person():
    lowered = TEACHING_PLAN_SYSTEM_INSTRUCTION.lower()

    assert "how deeply to teach" in lowered
    assert "never a judgement about the learner" in lowered
    assert "technical question" in lowered


def test_the_teaching_prompt_knows_about_continuing_a_lesson():
    lowered = TEACHING_PLAN_SYSTEM_INSTRUCTION.lower()

    assert "continuing a lesson" in lowered
    assert "step 2" in lowered
