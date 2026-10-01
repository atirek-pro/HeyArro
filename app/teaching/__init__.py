"""How deeply to explain an answer.

The only thing left of the teaching layer: the depth instruction that travels
with a request to the model. It is decided from the wording of the question, so
it costs no extra model call, and it changes only the words of the answer.
"""

from app.teaching.difficulty import (
    DEFAULT_DIFFICULTY,
    TeachingDifficulty,
    detect_difficulty,
    difficulty_guidance,
)

__all__ = [
    "DEFAULT_DIFFICULTY",
    "TeachingDifficulty",
    "detect_difficulty",
    "difficulty_guidance",
]
