#!/usr/bin/env python3
"""Create a named, purpose-scoped local bridge session folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SKILL_ROOT = ROOT.parent if ROOT.name in {"hpc_bridge", "scripts"} else ROOT
DEFAULT_SESSIONS_ROOT = SKILL_ROOT / "bridge_sessions"
DEFAULT_INSTALLED_SKILLS_ROOT = Path.home() / ".codex" / "skills"
KNOWN_BRIDGE_SKILLS = ("kestrel-hpc", "cloudvm-bridge", "constance-hpc", "expanse-hpc")
EXCLUDED_NAMES = {
    ".venv",
    "__pycache__",
    "commands",
    "results",
    "bridge_sessions",
}
EXCLUDED_FILES = {
    "bridge_identity.json",
    "bridge_status.txt",
}


def normalize_purpose(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_project_root(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def normalize_remote_target(value: str) -> str:
    target = value.strip()
    if target.count("@") != 1 or any(not part for part in target.split("@", 1)):
        raise ValueError("--remote-target must be username@hostname")
    return target


def build_purpose_key(purpose: str, project_root: str) -> str:
    payload = f"{normalize_purpose(purpose)}\n{normalize_project_root(project_root)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_names() -> list[str]:
    path = ROOT / "bridge_names.json"
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Invalid bridge name list: {path}")
    return names


def iter_identity_files(session_roots: Iterable[Path]) -> Iterable[Path]:
    for root in session_roots:
        if root.exists():
            yield from root.rglob("bridge_identity.json")


def installed_session_roots(installed_skills_root: Path) -> list[Path]:
    roots: list[Path] = []
    for skill in KNOWN_BRIDGE_SKILLS:
        roots.append(installed_skills_root / skill / "bridge_sessions")
    return roots


def used_names(installed_skills_root: Path, extra_roots: Iterable[Path]) -> set[str]:
    roots = installed_session_roots(installed_skills_root)
    roots.extend(extra_roots)
    used: set[str] = set()
    for path in iter_identity_files(roots):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("bridge_name")
        if isinstance(name, str) and name:
            used.add(name)
    return used


def choose_name(names: list[str], unavailable: set[str]) -> str:
    available = [name for name in names if name not in unavailable]
    if not available:
        raise RuntimeError("All bridge names are already used by existing bridge_identity.json files.")
    return secrets.choice(available)


def safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return text.strip("-") or "bridge"


def unique_session_dir(root: Path, bridge_name: str, stamp: str) -> Path:
    base = root / f"{safe_component(bridge_name)}_{stamp}"
    if not base.exists():
        return base
    for index in range(1, 100):
        candidate = root / f"{safe_component(bridge_name)}_{stamp}_{index:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused session directory for {bridge_name}")


def copy_helper_files(destination: Path) -> None:
    for path in ROOT.iterdir():
        if path.name in EXCLUDED_NAMES or path.name in EXCLUDED_FILES:
            continue
        if path.is_dir():
            shutil.copytree(path, destination / path.name, ignore=shutil.ignore_patterns(*EXCLUDED_NAMES))
        elif path.is_file():
            shutil.copy2(path, destination / path.name)


def write_identity(path: Path, args: argparse.Namespace, bridge_name: str, timestamp: datetime) -> dict[str, str]:
    project_root = normalize_project_root(args.project_root)
    identity = {
        "bridge_name": bridge_name,
        "startup_datetime_local": timestamp.replace(microsecond=0).astimezone().isoformat(),
        "purpose": args.purpose.strip(),
        "purpose_key": build_purpose_key(args.purpose, project_root),
        "work_summary": args.work_summary.strip(),
        "local_project_root": project_root,
        "remote_target": normalize_remote_target(args.remote_target),
        "expected_host_key_sha256": args.host_key_fingerprint.strip(),
    }
    path.write_text(json.dumps(identity, indent=2), encoding="utf-8", newline="\n")
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purpose", required=True, help="Short purpose for this bridge session.")
    parser.add_argument("--work-summary", required=True, help="One to three sentence summary of intended bridge work.")
    parser.add_argument("--project-root", required=True, help="Absolute or relative local project root this bridge serves.")
    parser.add_argument("--remote-target", required=True, help="Private runtime SSH target in username@hostname form.")
    parser.add_argument("--host-key-fingerprint", required=True, help="Approved SHA256 host-key fingerprint for the runtime host.")
    parser.add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_ROOT, help="Directory where session folders are created.")
    parser.add_argument("--installed-skills-root", type=Path, default=DEFAULT_INSTALLED_SKILLS_ROOT, help="Root used to scan installed bridge session names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_root = args.sessions_root.expanduser().resolve()
    installed_root = args.installed_skills_root.expanduser().resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    unavailable = used_names(installed_root, [session_root])
    bridge_name = choose_name(load_names(), unavailable)
    now = datetime.now().astimezone()
    session_dir = unique_session_dir(session_root, bridge_name, now.strftime("%Y%m%d_%H%M%S"))
    session_dir.mkdir(parents=True, exist_ok=False)
    copy_helper_files(session_dir)
    identity = write_identity(session_dir / "bridge_identity.json", args, bridge_name, now)

    print(json.dumps({"session_dir": str(session_dir), "identity": identity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
