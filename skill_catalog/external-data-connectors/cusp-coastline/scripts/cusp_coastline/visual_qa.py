"""Agent visual QA manifests for coastline diagnostic maps."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping


VISUAL_REVIEW_CHECKLIST = [
    "Open every listed satellite PNG before accepting the coastline product.",
    "Confirm shoreline vectors trace the visible land-water interface throughout the bbox.",
    "Flag any long visible island, fjord, bay, or channel shoreline that has no vector overlay.",
    "Flag vector segments that stop abruptly in the middle of an obvious coastline.",
    "Flag obvious inland lakes, rivers, snow edges, or terrain edges incorrectly treated as coast.",
    "For merged products, confirm fallback lines fill gaps without duplicating primary CUSP lines.",
    "Treat cloud, shadow, snow, and image seam ambiguity as manual-review risk, not automatic pass.",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_visual_review_files(
    run_dir: str | Path,
    name: str,
    *,
    bbox: tuple[float, float, float, float],
    image_paths: Mapping[str, str | Path],
    vector_paths: Mapping[str, str | Path],
    context: Mapping[str, object],
) -> dict[str, str]:
    """Write JSON and Markdown files that require agent visual review."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"{name}_visual_review.json"
    md_path = run_dir / f"{name}_visual_review.md"

    manifest = {
        "name": name,
        "bbox_wsen": [float(x) for x in bbox],
        "status": "needs_agent_review",
        "decision": None,
        "reviewer": None,
        "reviewed_at_utc": None,
        "notes": None,
        "fail_reasons": [],
        "image_paths": {key: str(value) for key, value in image_paths.items()},
        "vector_paths": {key: str(value) for key, value in vector_paths.items()},
        "context": dict(context),
        "checklist": VISUAL_REVIEW_CHECKLIST,
        "created_at_utc": _utc_now(),
    }
    json_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_manifest_markdown(manifest), encoding="utf-8")
    return {"visual_review_json": str(json_path), "visual_review_md": str(md_path)}


def record_visual_review(
    manifest_path: str | Path,
    *,
    decision: str,
    reviewer: str,
    notes: str,
    fail_reasons: list[str] | None = None,
) -> dict[str, object]:
    """Record an agent or human visual review decision into a manifest."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = decision.lower().strip()
    if decision not in {"pass", "fail", "needs_followup"}:
        raise ValueError("decision must be pass, fail, or needs_followup")

    manifest["status"] = "reviewed"
    manifest["decision"] = decision
    manifest["reviewer"] = reviewer
    manifest["reviewed_at_utc"] = _utc_now()
    manifest["notes"] = notes
    manifest["fail_reasons"] = fail_reasons or []
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    md_path = manifest_path.with_suffix(".md")
    if md_path.exists() or md_path.name.endswith("_visual_review.md"):
        md_path.write_text(_render_manifest_markdown(manifest), encoding="utf-8")
    return manifest


def visual_review_passed(outputs: Mapping[str, object]) -> bool:
    """Return True only when the visual review JSON exists and records a pass."""

    path = outputs.get("visual_review_json")
    if not path:
        return False
    manifest_path = Path(str(path))
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("status") == "reviewed" and manifest.get("decision") == "pass"


def _render_manifest_markdown(manifest: Mapping[str, object]) -> str:
    lines = [
        f"# Visual Coastline Review: {manifest['name']}",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Decision: `{manifest.get('decision')}`",
        f"- Reviewer: `{manifest.get('reviewer')}`",
        f"- Reviewed UTC: `{manifest.get('reviewed_at_utc')}`",
        f"- Bbox WSEN: `{manifest['bbox_wsen']}`",
        "",
        "## Images To Inspect",
    ]
    for key, value in dict(manifest.get("image_paths", {})).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Vector Products"])
    for key, value in dict(manifest.get("vector_paths", {})).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Checklist"])
    for item in manifest.get("checklist", []):
        lines.append(f"- [ ] {item}")

    lines.extend(["", "## Notes", str(manifest.get("notes") or "")])
    fail_reasons = list(manifest.get("fail_reasons") or [])
    if fail_reasons:
        lines.extend(["", "## Fail Reasons"])
        for reason in fail_reasons:
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"
