"""Shared helpers for CodexLab review pages and local artifact links."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def repo_rel(path: Path | str) -> str:
    """Return a repository-relative path string for local artifacts."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return os.path.relpath(path.resolve(), ROOT)


def rel_for_html(target: Path | str, index_dir: Path) -> str:
    """Return an escaped relative href/src from an index page to an artifact."""
    target = Path(target)
    if not target.is_absolute():
        target = ROOT / target
    return html.escape(os.path.relpath(target.resolve(), index_dir.resolve()))


def h(value: Any) -> str:
    return html.escape(str(value))


def code(value: Any) -> str:
    return f"<code>{h(value)}</code>"


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def html_value_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(html.escape(str(value)) for value in values)
