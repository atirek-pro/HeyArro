"""JSON manifests written next to a refined screenshot.

``ss_segment_manifest.json`` describes every segment found on the screen.
``relevant_ss_manifest.json`` keeps only the segments that matter for the user's
question, each with the same details plus a ``score``.

Both live in the screenshot's own folder, so a run can be inspected by hand.
"""

import json
from pathlib import Path

from app.refinement.provider import RefinementResult, ScreenSegment

MANIFEST_NAME = "ss_segment_manifest.json"
RELEVANT_MANIFEST_NAME = "relevant_ss_manifest.json"


def segment_entry(segment: ScreenSegment, folder) -> dict:
    """One segment as a manifest entry; ``file`` is relative to the screenshot folder."""
    file = None
    if segment.path:
        try:
            file = Path(segment.path).relative_to(folder).as_posix()
        except ValueError:
            file = Path(segment.path).name

    return {
        "id": segment.index,
        "kind": segment.kind,
        "text": segment.text,
        "box": {
            "left": segment.left,
            "top": segment.top,
            "width": segment.width,
            "height": segment.height,
        },
        "file": file,
    }


def write_segment_manifest(result: RefinementResult) -> Path:
    """Write ``ss_segment_manifest.json`` for every segment of ``result``."""
    folder = Path(result.folder)
    manifest = {
        "screenshot": Path(result.screenshot_path).name,
        "strategy": result.strategy,
        "segment_count": result.segment_count,
        "segments": [segment_entry(segment, folder) for segment in result.segments],
    }
    return _write(folder / MANIFEST_NAME, manifest)


def write_relevant_manifest(
    result: RefinementResult, query, model, scores, seconds, response_category, confidence, category_reason
) -> Path:
    """Write ``relevant_ss_manifest.json``: the category of the question and the scored segments.

    ``scores`` maps a segment id to its score; segments without a score are left out,
    and the rest are listed highest score first.
    """
    folder = Path(result.folder)
    scored = [
        {**segment_entry(segment, folder), "score": scores[segment.index]}
        for segment in result.segments
        if segment.index in scores
    ]
    scored.sort(key=lambda entry: (-entry["score"], entry["id"]))

    manifest = {
        "query": query,
        "response_category": response_category,
        "confidence": confidence,
        "category_reason": category_reason,
        "model": model,
        "screenshot": Path(result.screenshot_path).name,
        "source_manifest": MANIFEST_NAME,
        "total_segments": result.segment_count,
        "selected_count": len(scored),
        "selection_seconds": round(seconds, 3),
        "segments": scored,
    }
    return _write(folder / RELEVANT_MANIFEST_NAME, manifest)


def _write(path: Path, manifest: dict) -> Path:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
