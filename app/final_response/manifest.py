"""The final response file written next to a response plan.

``final_response.json`` lives in the screenshot's own folder beside the
manifests, ``response_plan.json`` and ``plan_pointers.json``, so one run can be
inspected by hand.
"""

import json
from pathlib import Path

FINAL_RESPONSE_NAME = "final_response.json"


def write_final_response(folder, document: dict) -> Path:
    """Write ``final_response.json`` into ``folder``."""
    path = Path(folder) / FINAL_RESPONSE_NAME
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
