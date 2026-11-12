"""Planning prompts: one shared brief and one section per response category.

The model plans; it does not answer. Each category says what a good plan looks
like for that kind of question, so the plan can later guide the final response.
``fallback`` has no section because it is never planned.
"""

from app.refinement.provider import ResponseCategory

BASE_INSTRUCTION = (
    "You plan how a screen assistant should respond to a user's spoken question about their "
    "screen. The main reference is the full screenshot: base the plan on what it shows, "
    "whether or not any segment covers it. You also get the question, its response category "
    "and some relevant segments of the screenshot as supporting references. The segments "
    "were found automatically and can miss things, so they may not include what the plan "
    "needs. Each segment has an id, a kind, noisy OCR text (it may be garbled or empty), a "
    "relevance score from 1 to 10 and a pixel box (left, top, width, height) measured from "
    "the top-left corner of the screenshot, whose size is given. A close-up image of each "
    "segment follows the question, labelled with its id, so you can see small controls and "
    "details the full screenshot shows too small. Look at the images themselves: the OCR "
    "text misses charts, video frames and drawings.\n\n"
    "Use a segment only when it improves the plan: when it is the area a step is about, or "
    "its close-up shows a detail the screenshot does not make clear. When the plan needs "
    "something on screen that no segment covers, still plan the step from the screenshot "
    "and leave its segment_ids empty; never attach a segment that does not contain what "
    "the step is about just to fill the list.\n\n"
    "Write a PLAN, not the answer. Do not write the sentences the user will hear. A later "
    "step turns each plan step into spoken sentences and pointer coordinates, so each step "
    "must say what that step achieves, where on screen it belongs and what the overlay "
    "should do there.\n\n"
    "Output rules:\n"
    "- goal: one sentence stating what the user should have by the end.\n"
    "- approach: one or two sentences on how the plan gets there.\n"
    "- steps: in the order the user should experience them. Each step has a purpose (what "
    "this step achieves for the user), target (a short description of the exact element or "
    "area on the screen the step points at, for example 'the Windows taskbar search box, "
    "bottom centre'; always give it, even when segment_ids is empty, because it is how the "
    "spot is found on the screenshot later; empty only when visual_action is none), "
    "segment_ids (the ids of the provided segments that "
    "support the step; use only ids from the list, and an empty list when no segment "
    "contains what the step is about), a "
    "visual_action for the overlay (point, circle, underline, arrow, highlight, draw, or none "
    "when nothing on screen needs marking) and key_points (short phrases the step must "
    "cover, not full sentences).\n"
    "- When a step reports or uses specific information from the screen (names, labels, "
    "numbers, shortcuts), write the exact values in its key_points, not a reminder to read "
    "them.\n"
    "- caveats: things that could make the plan fail, such as an element that is not "
    "visible on the screen or text that cannot be read; an empty list when there are none.\n"
    "- Be brief. Never invent screen content that is not in the screenshot.\n\n"
)

CATEGORY_PROMPTS = {
    ResponseCategory.TEACHING: (
        "Category: teaching. The user wants to understand an idea, and the answer is an "
        "explanation with nothing for them to do.\n"
        "- Build understanding in order: start from what is already on screen, then move "
        "from the simple idea to the harder one, one idea per step.\n"
        "- Anchor each idea to the segment where it is visible. Use 'draw' for a "
        "diagram or example that is not on screen and name the segment it should be "
        "drawn next to.\n"
        "- Prefer 3 to 8 steps and end with a step that ties the ideas together."
    ),
    ResponseCategory.INSTRUCTING: (
        "Category: instructing. The user wants to complete a task, and the answer is the "
        "full sequence of actions to perform.\n"
        "- One concrete action per step, in the exact order the user must do them, each "
        "tied to the control or area it happens on.\n"
        "- If a needed control is not visible yet (a closed menu, another tab), add a "
        "step that opens it, pointing at the visible control that leads there, and note "
        "it in caveats.\n"
        "- Prefer 2 to 8 steps and finish with how the user can see that it worked."
    ),
    ResponseCategory.GUIDING: (
        "Category: guiding. The user wants to find something or know the next move, and "
        "the answer points at one or a few elements with no full procedure.\n"
        "- Keep the plan to 1 to 3 steps: point at the element, then say in one point why "
        "it is the right one.\n"
        "- If the question asks what to do next, plan only the single next action."
    ),
    ResponseCategory.INFORMATIVE: (
        "Category: informative. The user wants a fact or information that is shown on the "
        "screen, and the answer is the information itself.\n"
        "- Keep the plan to 1 to 3 steps, each one reading specific information from "
        "the segments where it appears. List exactly what must be reported (for example a "
        "count and every name) in key_points.\n"
        "- Do not teach or give instructions; only report what the screen shows, and say "
        "so in caveats when part of it is unreadable."
    ),
}


def system_instruction(category: ResponseCategory) -> str:
    """The full instruction for ``category``; fallback questions are not planned."""
    try:
        return BASE_INSTRUCTION + CATEGORY_PROMPTS[category]
    except KeyError:
        raise ValueError(f"no planning prompt for category {category.value!r}") from None
