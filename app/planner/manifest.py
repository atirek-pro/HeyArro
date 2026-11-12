"""The plan file written next to a refined screenshot.

``response_plan.json`` lives in the screenshot's own folder beside
``ss_segment_manifest.json`` and ``relevant_ss_manifest.json``, so one run can be
inspected by hand.
"""

import json
from pathlib import Path

PLAN_NAME = "response_plan.json"


def write_plan(folder, plan: dict) -> Path:
    """Write ``response_plan.json`` into ``folder``."""
    path = Path(folder) / PLAN_NAME
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
