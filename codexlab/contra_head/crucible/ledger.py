"""Build Crucible discrepancy ledgers from transition corpora."""

from __future__ import annotations

import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .canon import canonicalize_grid
from .corpus import (
    DEFAULT_CORPUS_DIR,
    ROOT,
    resolve_repo_path,
    sha256_file,
    source_hashes,
    stable_id,
    stable_json,
    truth_boundary,
)
from .forward_v0 import MOVE_DELTAS, PREDICT_V0_VERSION, V0Prediction, predict_v0


ANOMALY_SCHEMA = "codexlab.crucible_anomaly.v1"
LEDGER_SCHEMA = "codexlab.crucible_discrepancy_ledger.v1"
DEFAULT_DISCREPANCY_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_discrepancy_v0"
OOB_COLOR = -1


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _tuple_delta(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = value
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return int(a), int(b)


def _left_of(delta: tuple[int, int]) -> tuple[int, int]:
    dx, dy = delta
    return dy, -dx


def _right_of(delta: tuple[int, int]) -> tuple[int, int]:
    dx, dy = delta
    return -dy, dx


def _reverse_of(delta: tuple[int, int]) -> tuple[int, int]:
    dx, dy = delta
    return -dx, -dy


def classify_anomaly(
    primitive: str,
    predicted_delta: tuple[int, int] | None,
    observed_delta: tuple[int, int] | None,
) -> str | None:
    if predicted_delta is None or observed_delta is None:
        return "other"
    if observed_delta == predicted_delta:
        return None
    if primitive in MOVE_DELTAS and observed_delta == (0, 0):
        return "blocked"
    if predicted_delta == (0, 0) and observed_delta != (0, 0):
        return "moved_when_static_expected"
    if primitive in MOVE_DELTAS:
        if observed_delta == _left_of(predicted_delta):
            return "redirect_left"
        if observed_delta == _right_of(predicted_delta):
            return "redirect_right"
        if observed_delta == _reverse_of(predicted_delta):
            return "reverse"
        if _is_slide(predicted_delta, observed_delta):
            return "slide"
        if abs(observed_delta[0]) + abs(observed_delta[1]) != 1:
            return "off_lattice"
    return "other"


def _is_slide(predicted_delta: tuple[int, int], observed_delta: tuple[int, int]) -> bool:
    px, py = predicted_delta
    ox, oy = observed_delta
    if predicted_delta == (0, 0):
        return False
    if px == 0 and ox == 0 and py != 0 and oy % py == 0 and abs(oy) > abs(py):
        return True
    if py == 0 and oy == 0 and px != 0 and ox % px == 0 and abs(ox) > abs(px):
        return True
    return False


def _cell(grid: np.ndarray, coord: tuple[int, int]) -> int:
    row, col = coord
    if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        return OOB_COLOR
    return int(grid[row, col])


def _offset(coord: tuple[int, int], delta: tuple[int, int], multiplier: int = 1) -> tuple[int, int]:
    return coord[0] + delta[0] * multiplier, coord[1] + delta[1] * multiplier


def _neighborhood_5x5(grid: np.ndarray, actor_coord: tuple[int, int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for dr in range(-2, 3):
        out: list[int] = []
        for dc in range(-2, 3):
            out.append(_cell(grid, (actor_coord[0] + dr, actor_coord[1] + dc)))
        rows.append(out)
    return rows


def _component_stats(grid: np.ndarray, actor_coord: tuple[int, int], color: int) -> dict[str, Any]:
    if _cell(grid, actor_coord) != color:
        return {"size": 0, "bbox": None, "axis": "unknown", "width": 0, "height": 0}
    seen = {actor_coord}
    queue: deque[tuple[int, int]] = deque([actor_coord])
    while queue:
        row, col = queue.popleft()
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + delta[0], col + delta[1])
            if nxt in seen or _cell(grid, nxt) != color:
                continue
            seen.add(nxt)
            queue.append(nxt)
    rs = [coord[0] for coord in seen]
    cs = [coord[1] for coord in seen]
    height = max(rs) - min(rs) + 1
    width = max(cs) - min(cs) + 1
    if width > height:
        axis = "h"
    elif height > width:
        axis = "v"
    elif width == height == 1:
        axis = "point"
    else:
        axis = "square"
    return {
        "size": len(seen),
        "bbox": [min(rs), min(cs), max(rs), max(cs)],
        "axis": axis,
        "width": width,
        "height": height,
    }


def build_local_context(row: dict[str, Any], prediction: V0Prediction) -> dict[str, Any]:
    before = row.get("before_grid") if isinstance(row.get("before_grid"), dict) else {}
    actor = row.get("actor") if isinstance(row.get("actor"), dict) else {}
    before_path = resolve_repo_path(str(before.get("path") or ""))
    grid = np.load(before_path)
    rotation = int(actor.get("rotation_quarters") or 0)
    canon = canonicalize_grid(grid, rotation)
    actor_coord = _tuple_delta(actor.get("before_canon_coord"))
    if actor_coord is None:
        raise ValueError("missing actor before_canon_coord")
    delta = prediction.predicted_canon_delta or (0, 0)
    forward_ray = [_cell(canon, _offset(actor_coord, delta, step)) for step in (1, 2, 3)]
    component = _component_stats(canon, actor_coord, int(actor.get("color") or 1))
    return {
        "canon_neighborhood_5x5": _neighborhood_5x5(canon, actor_coord),
        "forward_ray_3": forward_ray,
        "actor_color": int(actor.get("color") or 1),
        "actor_shape": component,
        "actor_axis": component.get("axis"),
    }


def build_anomaly_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    primitive = str(((row.get("action") or {}).get("primitive") or ""))
    actor = row.get("actor") if isinstance(row.get("actor"), dict) else {}
    if actor.get("selection_changed") is True:
        return None, "selection_changed_transition"
    prediction = predict_v0(primitive)
    predicted = prediction.predicted_canon_delta
    observed = _tuple_delta((row.get("observed") or {}).get("canon_delta"))
    anomaly_kind = classify_anomaly(primitive, predicted, observed)
    if anomaly_kind is None:
        return None, "matched_v0"
    context = build_local_context(row, prediction)
    anomaly_id = stable_id(
        "crucible_anomaly",
        row.get("transition_id"),
        primitive,
        prediction.to_dict(),
        observed,
        anomaly_kind,
    )
    return {
        "schema": ANOMALY_SCHEMA,
        "anomaly_id": anomaly_id,
        "transition_id": row.get("transition_id"),
        "action": row.get("action"),
        "primitive": primitive,
        "predicted": prediction.to_dict(),
        "observed": {
            "canon_delta": None if observed is None else [observed[0], observed[1]],
            "moved": (row.get("observed") or {}).get("moved"),
            "blocked": (row.get("observed") or {}).get("blocked"),
        },
        "anomaly_kind": anomaly_kind,
        "local_context": context,
        "provenance": row.get("provenance"),
        "truth_boundary": truth_boundary(),
    }, "anomaly"


def _corpus_paths(corpus_dir: Path) -> tuple[Path, Path]:
    root = corpus_dir if corpus_dir.is_absolute() else ROOT / corpus_dir
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    corpus_path = resolve_repo_path(str(manifest.get("corpus_jsonl") or root / "corpus.jsonl"))
    return manifest_path, corpus_path


def build_discrepancy_ledger(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    out_dir: Path = DEFAULT_DISCREPANCY_DIR,
) -> dict[str, Any]:
    manifest_path, corpus_path = _corpus_paths(corpus_dir)
    corpus_rows = read_jsonl(corpus_path)
    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)

    anomalies: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for row in corpus_rows:
        anomaly, status = build_anomaly_row(row)
        status_counts[status] += 1
        if anomaly is not None:
            anomalies.append(anomaly)

    anomalies_path = resolved_out / "anomalies.jsonl"
    with anomalies_path.open("w", encoding="utf-8") as handle:
        for anomaly in anomalies:
            handle.write(json.dumps(anomaly, sort_keys=True) + "\n")

    anomaly_kind_by_primitive: dict[str, dict[str, int]] = {}
    forward_cell_histogram: dict[str, dict[str, int]] = {}
    for anomaly in anomalies:
        kind = str(anomaly.get("anomaly_kind"))
        primitive = str(anomaly.get("primitive"))
        anomaly_kind_by_primitive.setdefault(kind, {})
        anomaly_kind_by_primitive[kind][primitive] = anomaly_kind_by_primitive[kind].get(primitive, 0) + 1
        ray = ((anomaly.get("local_context") or {}).get("forward_ray_3") or [])
        color = str(ray[0]) if ray else "missing"
        forward_cell_histogram.setdefault(kind, {})
        forward_cell_histogram[kind][color] = forward_cell_histogram[kind].get(color, 0) + 1

    ledger_path = resolved_out / "ledger.json"
    ledger = {
        "schema": LEDGER_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "predictor_version": PREDICT_V0_VERSION,
        "corpus_manifest": _rel(manifest_path),
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "corpus_jsonl": _rel(corpus_path),
        "corpus_jsonl_sha256": sha256_file(corpus_path),
        "anomalies_jsonl": _rel(anomalies_path),
        "ledger_json": _rel(ledger_path),
        "transition_count": len(corpus_rows),
        "anomaly_count": len(anomalies),
        "status_counts": dict(sorted(status_counts.items())),
        "anomaly_kind_by_primitive": {
            kind: dict(sorted(counts.items()))
            for kind, counts in sorted(anomaly_kind_by_primitive.items())
        },
        "forward_cell_color_histogram": {
            kind: dict(sorted(counts.items(), key=lambda item: item[0]))
            for kind, counts in sorted(forward_cell_histogram.items())
        },
        "source_hashes": source_hashes(
            [
                "codexlab/contra_head/crucible/forward_v0.py",
                "codexlab/contra_head/crucible/ledger.py",
                "codexlab/crucible_discrepancy_ledger.py",
            ]
        ),
        "truth_boundary": truth_boundary(),
    }
    ledger["ledger_sha256"] = stable_id(
        "crucible_discrepancy_ledger",
        ledger["predictor_version"],
        ledger["corpus_jsonl_sha256"],
        len(anomalies),
        stable_json(ledger["anomaly_kind_by_primitive"]),
    )
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger
