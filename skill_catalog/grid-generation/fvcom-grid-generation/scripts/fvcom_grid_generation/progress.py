"""Progress and heartbeat artifacts for long FVCOM grid runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any


@dataclass
class ProgressTracker:
    """Write monotonic progress snapshots and JSONL heartbeat events."""

    run_dir: Path
    name: str
    interval_s: float = 10.0
    started_monotonic: float = field(default_factory=time.monotonic)
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    last_percent: float = 0.0

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.run_dir / "progress.json"
        self.jsonl_path = self.run_dir / "progress.jsonl"
        self.jsonl_path.write_text("", encoding="utf-8")

    def update(
        self,
        stage: str,
        percent: float,
        *,
        message: str = "",
        pid: int | None = None,
        iteration: int | None = None,
        total_iterations: int | None = None,
        artifact: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one progress event and return the serialized event."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        percent = max(float(percent), float(self.last_percent))
        percent = min(percent, 100.0)
        self.last_percent = percent
        event: dict[str, Any] = {
            "schema_version": "fvcom_grid_progress_v1",
            "name": self.name,
            "stage": stage,
            "percent": percent,
            "message": message,
            "started_utc": self.started_utc,
            "last_update_utc": now,
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 3),
        }
        if pid is not None:
            event["pid"] = int(pid)
        if iteration is not None:
            event["iteration"] = int(iteration)
        if total_iterations is not None:
            event["total_iterations"] = int(total_iterations)
        if artifact is not None:
            event["last_known_artifact"] = str(artifact)
        if extra:
            event["extra"] = _json_safe(extra)
        self.json_path.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        return event


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
