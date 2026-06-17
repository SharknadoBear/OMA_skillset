"""Progress logging helpers for long coastline fetch/merge workflows."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Iterator


def normalize_timeout(value: float | int | None) -> float | None:
    """Return None for no hard timeout; otherwise return a positive timeout."""

    if value is None:
        return None
    timeout = float(value)
    return timeout if timeout > 0 else None


class ProgressReporter:
    """Emit flushed human progress lines and optional JSONL events."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        heartbeat_seconds: float = 30.0,
        quiet: bool = False,
    ) -> None:
        self.path = Path(path) if path else None
        self.heartbeat_seconds = max(float(heartbeat_seconds), 1.0)
        self.quiet = quiet
        self._started = time.monotonic()
        self._stage_starts: dict[str, float] = {}
        self._stage_elapsed: dict[str, float] = {}
        self._last_emit: dict[str, float] = {}
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def stage_elapsed(self) -> dict[str, float]:
        """Return cumulative stage durations in seconds."""

        elapsed = dict(self._stage_elapsed)
        now = time.monotonic()
        for stage, started in self._stage_starts.items():
            elapsed[stage] = elapsed.get(stage, 0.0) + (now - started)
        return {key: round(value, 3) for key, value in elapsed.items()}

    def event(self, stage: str, message: str, *, force: bool = True, **details: object) -> None:
        """Emit one progress event."""

        now = time.monotonic()
        elapsed = now - self._started
        stage_elapsed = now - self._stage_starts.get(stage, now)
        payload = {
            "time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "stage": stage,
            "elapsed_seconds": round(elapsed, 3),
            "stage_elapsed_seconds": round(stage_elapsed, 3),
            "message": message,
            "details": details,
        }
        self._last_emit[stage] = now
        with self._lock:
            if not self.quiet:
                detail_text = " ".join(f"{key}={value}" for key, value in details.items() if value is not None)
                suffix = f" {detail_text}" if detail_text else ""
                print(f"[{payload['time_utc']}][{stage}][elapsed={int(elapsed)}s] {message}{suffix}", flush=True)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, default=str) + "\n")

    def heartbeat(self, stage: str, message: str, **details: object) -> None:
        """Emit an event when the heartbeat interval has elapsed for a stage."""

        now = time.monotonic()
        last = self._last_emit.get(stage, 0.0)
        if now - last >= self.heartbeat_seconds:
            self.event(stage, message, force=False, **details)

    @contextmanager
    def background_heartbeat(self, stage: str, message: str, **details: object) -> Iterator[None]:
        """Emit heartbeat events while a blocking call runs in this thread."""

        stop = threading.Event()

        def _beat() -> None:
            while not stop.wait(self.heartbeat_seconds):
                self.event(stage, message, force=False, **details)

        thread = threading.Thread(target=_beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1.0)

    @contextmanager
    def stage(self, stage: str, message: str, **details: object) -> Iterator[None]:
        """Context manager that records start/end progress for a stage."""

        started = time.monotonic()
        self._stage_starts[stage] = started
        self.event(stage, f"start: {message}", **details)
        try:
            yield
        except Exception as exc:
            elapsed = time.monotonic() - started
            self._stage_elapsed[stage] = self._stage_elapsed.get(stage, 0.0) + elapsed
            self._stage_starts.pop(stage, None)
            self.event(stage, f"failed after {elapsed:.1f}s: {exc}", error=type(exc).__name__)
            raise
        else:
            elapsed = time.monotonic() - started
            self._stage_elapsed[stage] = self._stage_elapsed.get(stage, 0.0) + elapsed
            self._stage_starts.pop(stage, None)
            self.event(stage, f"done: {message}", stage_seconds=round(elapsed, 3))
