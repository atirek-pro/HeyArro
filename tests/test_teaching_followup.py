"""Recognising what a learner asks for after a lesson.

Follow-ups are read from the words the learner used - no classification model
call, no guessing about comprehension - and they only exist when a lesson was
actually just taught. Recognising one never touches the lesson that finished.
"""

import pytest

from app.teaching.difficulty import TeachingDifficulty
from app.teaching.followup import (
    TeachingFollowUp,
    TeachingFollowUpType,
    build_follow_up_context,
    detect_follow_up,
    follow_up_difficulty,
    follow_up_guidance,
)
from app.teaching.plan import TeachingPlan, TeachingStep

# --- what the learner says --------------------------------------------------


@pytest.mark.parametrize(
    "words, expected",
    [
        ("Explain that again.", TeachingFollowUpType.REPEAT),
        ("Make it simpler.", TeachingFollowUpType.SIMPLIFY),
        ("Go deeper.", TeachingFollowUpType.DEEPEN),
        ("Give me an example.", TeachingFollowUpType.EXAMPLE),
        ("Summarize that.", TeachingFollowUpType.SUMMARY),
        ("Continue.", TeachingFollowUpType.CONTINUE),
    ],
)
def test_a_follow_up_is_recognised_from_the_words(words, expected):
    follow_up = detect_follow_up(words)

    assert follow_up is not None
    assert follow_up.type is expected
    assert follow_up.text == words


@pytest.mark.parametrize(
    "words",
    [
        "Say that one more time.",
        "Can you repeat that?",
        "What did you say?",
    ],
)
def test_asking_for_it_again_is_a_repeat(words):
    assert detect_follow_up(words).type is TeachingFollowUpType.REPEAT


@pytest.mark.parametrize(
    "words",
    [
        "Can you explain it in simpler words?",
        "Explain it like I'm five.",
        "That was too technical, make it easier to follow.",
    ],
)
def test_asking_for_something_easier_is_a_simplification(words):
    assert detect_follow_up(words).type is TeachingFollowUpType.SIMPLIFY


@pytest.mark.parametrize(
    "words",
    [
        "Tell me more about that.",
        "Why does that work?",
        "I want the full picture.",
    ],
)
def test_asking_for_more_is_deepening(words):
    assert detect_follow_up(words).type is TeachingFollowUpType.DEEPEN


@pytest.mark.parametrize(
    "words",
    [
        "Show me another example.",
        "Demonstrate it on the screen.",
        "Where is that on the screen?",
    ],
)
def test_asking_to_be_shown_another_way_is_an_example(words):
    assert detect_follow_up(words).type is TeachingFollowUpType.EXAMPLE


@pytest.mark.parametrize(
    "words",
    [
        "What should I remember?",
        "Give me the key takeaway.",
        "Recap that in a nutshell.",
    ],
)
def test_asking_for_the_takeaway_is_a_summary(words):
    assert detect_follow_up(words).type is TeachingFollowUpType.SUMMARY


@pytest.mark.parametrize(
    "words",
    [
        "Keep going.",
        "What's next?",
        "Carry on.",
    ],
)
def test_asking_to_continue_is_a_continuation(words):
    assert detect_follow_up(words).type is TeachingFollowUpType.CONTINUE


# --- the step they lost -----------------------------------------------------


@pytest.mark.parametrize(
    "words, step",
    [
        ("I don't understand step 2.", 2),
        ("Explain step 3 please.", 3),
        ("I didn't get step three.", 3),
        ("What does the second step mean?", 2),
    ],
)
def test_the_step_they_ask_about_is_read_from_the_request(words, step):
    follow_up = detect_follow_up(words)

    assert follow_up.type is TeachingFollowUpType.EXPLAIN_STEP
    assert follow_up.step == step


def test_a_step_request_wins_over_a_generic_repeat():
    """"explain step 2 again" is about step 2, not about hearing it again."""
    follow_up = detect_follow_up("Explain step 2 again.")

    assert follow_up.type is TeachingFollowUpType.EXPLAIN_STEP
    assert follow_up.step == 2


def test_without_a_lesson_there_is_nothing_to_follow_up():
    assert detect_follow_up("Explain that again.", has_previous_plan=False) is None
    assert detect_follow_up("Go deeper.", has_previous_plan=False) is None


@pytest.mark.parametrize(
    "words",
    [
        "How do I export a file?",
        "What is this window for?",
        "How do I use git again?",
        "",
        "   ",
    ],
)
def test_a_new_question_is_not_treated_as_a_follow_up(words):
    assert detect_follow_up(words) is None


# --- whether the screen is needed -------------------------------------------


def test_a_recap_or_a_repeat_is_answered_in_words():
    assert detect_follow_up("Summarize that.").needs_screenshot is False
    assert detect_follow_up("Explain that again.").needs_screenshot is False


def test_anything_that_may_be_shown_needs_the_current_screen():
    for words in (
        "Make it simpler.",
        "Go deeper.",
        "Show me another example.",
        "I don't understand step 2.",
        "Continue.",
    ):
        assert detect_follow_up(words).needs_screenshot is True


# --- which level to teach it at ---------------------------------------------


def test_simplifying_moves_one_level_down():
    assert (
        follow_up_difficulty(
            TeachingDifficulty.ADVANCED, detect_follow_up("Make it simpler.")
        )
        is TeachingDifficulty.INTERMEDIATE
    )
    assert (
        follow_up_difficulty(
            TeachingDifficulty.INTERMEDIATE, detect_follow_up("Make it simpler.")
        )
        is TeachingDifficulty.BEGINNER
    )


def test_deepening_moves_one_level_up():
    assert (
        follow_up_difficulty(
            TeachingDifficulty.BEGINNER, detect_follow_up("Go deeper.")
        )
        is TeachingDifficulty.INTERMEDIATE
    )


def test_other_follow_ups_keep_the_level_they_were_taught_at():
    follow_up = detect_follow_up("Explain that again.")

    assert (
        follow_up_difficulty(TeachingDifficulty.BEGINNER, follow_up)
        is TeachingDifficulty.BEGINNER
    )


def test_the_learner_can_state_the_level_in_the_follow_up():
    """An explicit request beats the one-level default, in either direction."""
    simpler = detect_follow_up("Make it simpler, like I'm a complete beginner.")
    deeper = detect_follow_up("Summarize it, but give me the technical details.")

    assert (
        follow_up_difficulty(TeachingDifficulty.ADVANCED, simpler, simpler.text)
        is TeachingDifficulty.BEGINNER
    )
    assert (
        follow_up_difficulty(TeachingDifficulty.BEGINNER, deeper, deeper.text)
        is TeachingDifficulty.ADVANCED
    )


def test_a_level_is_never_changed_without_being_asked():
    words = "I don't understand step 2."

    assert (
        follow_up_difficulty(TeachingDifficulty.ADVANCED, detect_follow_up(words), words)
        is TeachingDifficulty.ADVANCED
    )


# --- what the follow-up asks for --------------------------------------------


def test_every_follow_up_type_has_guidance_of_its_own():
    guidance = {
        follow_up_type: follow_up_guidance(TeachingFollowUp(follow_up_type, "x"))
        for follow_up_type in TeachingFollowUpType
    }

    assert len(set(guidance.values())) == len(TeachingFollowUpType)
    assert all(text.strip() for text in guidance.values())


def test_there_is_no_guidance_for_no_follow_up():
    assert follow_up_guidance(None) == ""


# --- the lesson being continued ---------------------------------------------


def lesson(difficulty=TeachingDifficulty.INTERMEDIATE, conclusion="That is the idea."):
    return TeachingPlan(
        objective="Understand how the rows and columns combine.",
        difficulty=difficulty,
        introduction="Let's look at this together.",
        steps=[
            TeachingStep(
                step_id="step_1",
                order=1,
                objective="Find the row.",
                explanation="Start with the first row of the left matrix.",
            ),
            TeachingStep(
                step_id="step_2",
                order=2,
                objective="Find the column.",
                explanation="Now take the first column of the right matrix.",
            ),
        ],
        conclusion=conclusion,
    )


def test_the_context_describes_the_lesson_that_just_finished():
    follow_up = detect_follow_up("I don't understand step 2.")

    context = build_follow_up_context("How do I multiply matrices?", lesson(), follow_up)

    assert "How do I multiply matrices?" in context
    assert "Understand how the rows and columns combine." in context
    assert "intermediate" in context
    assert "1. Start with the first row of the left matrix." in context
    assert "2. Now take the first column of the right matrix." in context
    assert "That is the idea." in context
    # The step they asked about is spelled out, so "step 2" cannot be misread.
    assert "asking about step 2: Now take the first column" in context


def test_the_context_numbers_steps_as_the_learner_heard_them():
    plan = lesson().model_copy(
        update={
            "steps": [
                step.model_copy(update={"order": 10 if index else 20})
                for index, step in enumerate(lesson().steps)
            ]
        }
    )

    context = build_follow_up_context("Explain step 1.", plan, detect_follow_up("Explain step 1."))

    assert "asking about step 1: Start with the first row" in context


def test_the_context_leaves_out_a_conclusion_that_does_not_exist():
    context = build_follow_up_context(
        "Summarize that.", lesson(conclusion=None), detect_follow_up("Summarize that.")
    )

    assert "how it closed" not in context


def test_asking_about_a_step_the_lesson_does_not_have_says_so():
    follow_up = detect_follow_up("Explain step 9.")

    context = build_follow_up_context("Teach me this.", lesson(), follow_up)

    assert "does not have" in context


def long_lesson(steps=12):
    return TeachingPlan(
        objective="Understand a lot.",
        introduction="Here we go.",
        steps=[
            TeachingStep(
                step_id=f"step_{index}",
                order=index,
                objective="Understand this part.",
                explanation="y" * 400,
            )
            for index in range(1, steps + 1)
        ],
    )


def test_a_long_explanation_is_shortened_in_the_context():
    plan = lesson().model_copy(
        update={
            "steps": [
                step.model_copy(update={"explanation": "x" * 2000})
                for step in lesson().steps
            ]
        }
    )

    context = build_follow_up_context("Go deeper.", plan, detect_follow_up("Go deeper."))

    assert "x" * 400 not in context
    assert len(context) <= 2000


def test_the_context_is_capped_even_for_a_long_lesson():
    """One lesson cannot grow the request without limit."""
    context = build_follow_up_context(
        "Go deeper.", long_lesson(), detect_follow_up("Go deeper.")
    )

    assert len(context) <= 2000
    assert context.endswith("\u2026")


def test_the_context_says_what_the_learner_wants():
    follow_up = detect_follow_up("Make it simpler.")

    context = build_follow_up_context("Explain this.", lesson(), follow_up)

    assert "the learner now says" in context
    assert "Make it simpler." in context
