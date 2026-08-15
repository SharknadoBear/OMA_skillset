"""Safe rclone discovery and subprocess helpers for authenticated TPXO staging."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

REMOTE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
FOLDER_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")
FOLDER_URL = re.compile(r"/folders/([A-Za-z0-9_-]+)")


class RcloneError(RuntimeError):
    """Raised when an rclone command cannot be completed safely."""


def find_rclone(explicit: str | None = None) -> Path | None:
    """Find rclone on PATH or in the standard Windows winget package directory."""

    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        located = shutil.which(explicit)
        return Path(located).resolve() if located else None

    located = shutil.which("rclone")
    if located:
        return Path(located).resolve()

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            matches = sorted(package_root.glob("Rclone.Rclone_*/rclone-*-windows-*/rclone.exe"), reverse=True)
            if matches:
                return matches[0].resolve()
    return None


def normalize_remote(remote: str) -> str:
    """Return a validated rclone configuration name without a trailing colon."""

    name = remote.strip().removesuffix(":")
    if not REMOTE_NAME.fullmatch(name):
        raise ValueError("--remote must be a configured rclone remote name, without a path.")
    return name


def parse_drive_folder(value: str) -> str:
    """Extract a Google Drive folder ID from a runtime URL or accept a bare ID."""

    text = value.strip()
    match = FOLDER_URL.search(text)
    folder_id = match.group(1) if match else text
    if not FOLDER_ID.fullmatch(folder_id):
        raise ValueError("--drive-folder must be a Google Drive folder URL or folder ID.")
    return folder_id


def run_rclone(
    executable: Path,
    arguments: Sequence[str],
    *,
    timeout: int,
    redact: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    """Run rclone without a shell and redact runtime identifiers from failures."""

    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RcloneError(f"rclone timed out after {timeout} seconds.") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "rclone failed").strip()
        for secret in redact:
            if secret:
                detail = detail.replace(secret, "<redacted-runtime-id>")
        raise RcloneError(f"rclone exited with code {completed.returncode}: {detail}")
    return completed


def configured_remotes(executable: Path, timeout: int = 30) -> set[str]:
    """Return configured rclone remote names without reading or printing tokens."""

    output = run_rclone(executable, ["listremotes"], timeout=timeout).stdout
    return {line.strip().removesuffix(":") for line in output.splitlines() if line.strip()}
