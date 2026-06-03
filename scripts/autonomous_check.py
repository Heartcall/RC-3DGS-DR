#!/usr/bin/env python3
"""Lightweight checker for the autonomous protocol files.

This script uses only the Python standard library and never modifies files.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTONOMOUS_DIR = ROOT / "docs" / "autonomous"

REQUIRED_FILES = [
    AUTONOMOUS_DIR / "DUAL_WINDOW_PROTOCOL.md",
    AUTONOMOUS_DIR / "PLAN.md",
    AUTONOMOUS_DIR / "STATE.md",
    AUTONOMOUS_DIR / "AUTONOMOUS_LOG.md",
    AUTONOMOUS_DIR / "COORDINATION_BOARD.md",
    AUTONOMOUS_DIR / "VERIFICATION_MATRIX.md",
]

STATE_REQUIRED_FIELDS = [
    "current_phase:",
    "last_completed_task:",
    "active_task:",
    "known_blockers:",
    "latest_verification_status:",
    "current_recommended_model:",
    "next_candidate_tasks:",
    "last_updated:",
    "git_baseline_note:",
]

BOARD_REQUIRED_COLUMNS = [
    "task_id",
    "status",
    "owner_model",
    "claimed_at_jst",
    "files_to_touch",
    "risk_level",
    "last_update",
    "notes",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_files(errors: list[str]) -> None:
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")


def check_plan(errors: list[str]) -> None:
    path = AUTONOMOUS_DIR / "PLAN.md"
    if not path.exists():
        return
    text = read_text(path)
    if "task_id:" not in text:
        errors.append("PLAN.md does not contain task_id fields")
    for task_id in ["RC-AUTO-000", "RC-AUTO-010", "RC-AUTO-100"]:
        if task_id not in text:
            errors.append(f"PLAN.md missing expected task {task_id}")


def check_state(errors: list[str]) -> None:
    path = AUTONOMOUS_DIR / "STATE.md"
    if not path.exists():
        return
    text = read_text(path)
    for field in STATE_REQUIRED_FIELDS:
        if field not in text:
            errors.append(f"STATE.md missing required field {field}")


def check_board(errors: list[str]) -> None:
    path = AUTONOMOUS_DIR / "COORDINATION_BOARD.md"
    if not path.exists():
        return
    lines = read_text(path).splitlines()
    header = next((line for line in lines if line.startswith("| task_id |")), "")
    if not header:
        errors.append("COORDINATION_BOARD.md missing required table header")
        return
    normalized = [cell.strip().strip("|") for cell in header.split("|") if cell.strip()]
    for column in BOARD_REQUIRED_COLUMNS:
        if column not in normalized:
            errors.append(f"COORDINATION_BOARD.md missing column {column}")


def check_log(errors: list[str]) -> None:
    path = AUTONOMOUS_DIR / "AUTONOMOUS_LOG.md"
    if not path.exists():
        errors.append("AUTONOMOUS_LOG.md does not exist")
        return
    text = read_text(path)
    if "RC-AUTO-000" not in text:
        errors.append("AUTONOMOUS_LOG.md missing RC-AUTO-000 entry")


def check_markdown_trailing_whitespace(errors: list[str]) -> None:
    if not AUTONOMOUS_DIR.exists():
        errors.append("docs/autonomous directory does not exist")
        return
    for path in sorted(AUTONOMOUS_DIR.glob("*.md")):
        for line_no, line in enumerate(read_text(path).splitlines(), start=1):
            if line.endswith((" ", "\t")):
                rel = path.relative_to(ROOT)
                errors.append(f"trailing whitespace: {rel}:{line_no}")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_plan(errors)
    check_state(errors)
    check_board(errors)
    check_log(errors)
    check_markdown_trailing_whitespace(errors)

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
