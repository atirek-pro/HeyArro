"""The pointers file written next to a response plan.

``plan_pointers.json`` lives in the screenshot's own folder beside the manifests
and ``response_plan.json``, so one run can be inspected by hand.
"""

import json
from pathlib import Path

POINTERS_NAME = "plan_pointers.json"


def write_pointers(folder, pointers: dict) -> Path:
    """Write ``plan_pointers.json`` into ``folder``."""
    path = Path(folder) / POINTERS_NAME
    path.write_text(json.dumps(pointers, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
