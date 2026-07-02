#!/usr/bin/env python3
"""GapFinder: discover prediction-vs-observation divergences and emit gap_report.json.

The Gap is the ONLY allowed hypothesis seed for the CONTRA loop.  An agent
must NOT form a hypothesis from abstract reasoning; it reads gap_report.json
(produced here) and picks ONE gap to hypothesize about.

Three data sources (all local; no network calls):

  1. CONTRA run dirs  — packets.jsonl frame deltas (observed) vs ranked
     recommendations (predicted).  The primary mining source.
  2. arc_opus48 summary — level_progressions + action_counts from the public
     Opus-4.8 playthroughs; used as a verified action-routing prior BEFORE
     frame analysis.  Ingested read-only; never treated as local evidence.
  3. GroundTruthOracle seam — pluggable interface (win-bank first; engine
     sp80.py step() or raw_recordings fallback).  Stub on day one; activate
     when the win-bank replay API exists (see ground_truth.py TODO).

Boundary (verbatim, on every Gap record):
  "Hypothesis material only; not local CONTRA evidence."

Usage:
    python codexlab/contra_loop/gap_finder.py --run-dir <run_dir> --level 1
    python codexlab/contra_loop/gap_finder.py --run-dir <run_dir> --level 1 --out gap_report.json
    python codexlab/contra_loop/gap_finder.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]   # repo root (codexlab/../)
SCHEMA = "codexlab.contra_loop.gap_report.v1"

# Verbatim boundary string — must appear on EVERY Gap record.
BOUNDARY = "Hypothesis material only; not local CONTRA evidence."

# Default path for the arc_opus48 mining summary (relative to ROOT).
ARC_OPUS48_SUMMARY_REL = (
    "codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/summary.json"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Gap:
    """One prediction-vs-observation divergence — the raw material for a hypothesis.

    Produced by GapFinder.mine_run_dir() and serialised into gap_report.json.
    The agent reads gap_report.json; it must NOT generate a hypothesis from
    prose — it picks ONE Gap here and forms a falsifiable HypothesisEnvelope.
    """

    gap_id: str                     # sha256 of (run_dir, level, action_index, divergence_kind)
    game_id: str
    level_index: int
    action_index: int               # packet index in packets.jsonl
    action_primitive: str           # what primitive was actually executed
    candidate_kind: str             # pipeline's top-ranked candidate kind at this packet
    predicted_effect: str           # human-readable summary of what the pipeline expected
    observed_effect: str            # human-readable summary of what actually happened
    divergence_kind: str            # see _DIVERGENCE_KINDS below
    divergence_score: float         # 0.0–1.0; higher = more surprising gap
    evidence_detail: dict[str, Any] = field(default_factory=dict)
    boundary: str = BOUNDARY        # verbatim — not solve evidence
    source_run_dir: str = ""        # relative to ROOT
    prior_refs: list[dict[str, Any]] = field(default_factory=list)  # arc_opus48 priors


# Canonical divergence_kind labels — used as gap_report tags.
_DIVERGENCE_KINDS = frozenset({
    "cov_zero_every_step",          # forward model predicts 0 fill / 0 pixel change every step
    "premature_activate_flow",      # activate_flow fired but delta=0 (wrong precondition)
    "candidate_kind_thrash",        # top candidate_kind changes every packet (unstable scorer)
    "portband_lock_absent",         # L0 has 0.82 lock; here portband score never reaches it
    "predicted_not_predictive",     # pipeline explicitly says prediction_status=not_predictive
    "arc_opus48_routing_mismatch",  # dominant action in arc_opus48 differs from pipeline's top
    "level_budget_exhausted",       # run ended budget_exhausted with delta=0
    "observation_only",             # packet contributes data but no divergence found
})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sha256_str(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
    return h.hexdigest()[:16]


def _load_npy_frame(rel_path: str) -> "np.ndarray | None":
    """Load a .npy frame stored at ROOT/rel_path.  Returns None if unavailable."""
    if not _NUMPY_OK or not rel_path:
        return None
    abs_path = ROOT / rel_path
    if not abs_path.is_file():
        return None
    try:
        return np.load(str(abs_path))
    except Exception:
        return None


def _pixel_delta(before: "np.ndarray | None", after: "np.ndarray | None") -> dict[str, Any]:
    """Compute pixel-level diff between two raw grid frames.

    Returns a dict with changed_pixel_count, changed_fraction, unique_before,
    unique_after, unique_gained, unique_lost.  All values are 0/empty if frames
    unavailable (numpy not installed or files missing).
    """
    if before is None or after is None:
        return {
            "changed_pixel_count": 0,
            "changed_fraction": 0.0,
            "frames_available": False,
        }
    changed_mask = before != after
    n_changed = int(np.sum(changed_mask))
    total = int(before.size)
    ub = [int(x) for x in np.unique(before).tolist()]
    ua = [int(x) for x in np.unique(after).tolist()]
    gained = sorted(set(ua) - set(ub))
    lost = sorted(set(ub) - set(ua))
    return {
        "changed_pixel_count": n_changed,
        "changed_fraction": round(n_changed / total, 4) if total else 0.0,
        "unique_values_before": ub,
        "unique_values_after": ua,
        "unique_values_gained": gained,
        "unique_values_lost": lost,
        "frames_available": True,
    }


def _extract_top_rec(packet: dict[str, Any]) -> dict[str, Any]:
    """Return the top ranked_recommendation dict, or {} if absent."""
    crp = packet.get("contra_recommendation_packet") or {}
    recs = crp.get("ranked_recommendations") or []
    return recs[0] if recs else {}


def _portband_max_score(packet: dict[str, Any]) -> float:
    """Return the highest portband value_score across all ranked_recommendations."""
    crp = packet.get("contra_recommendation_packet") or {}
    recs = crp.get("ranked_recommendations") or []
    best = 0.0
    for r in recs:
        if "portband" in (r.get("candidate_kind") or ""):
            vs = r.get("value_score") or 0.0
            if vs > best:
                best = vs
    return best


# ---------------------------------------------------------------------------
# arc_opus48 prior ingestion
# ---------------------------------------------------------------------------

def load_arc_opus48_priors(
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Read the arc_opus48 mining summary and return per-game verified priors.

    These are server-verified (Opus 4.8 actually completed those levels).
    They are ingested as priors, not as local CONTRA evidence.

    Returns a dict keyed by game_id:
      {
        "sp80-589a99af": {
          "boundary": "...",
          "dominant_action": "ACTION5",
          "level_progressions": [ {event_index, from, to, action_id, timestamp} ],
          "action_counts": {"ACTION1": 23, ...},
          "zero_score": False,
          "levels_completed": 2,
        },
        ...
      }
    """
    path = summary_path or (ROOT / ARC_OPUS48_SUMMARY_REL)
    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    outer_boundary = data.get("boundary", BOUNDARY)
    priors: dict[str, Any] = {}
    for run in data.get("runs") or []:
        game_id = run.get("game_id") or ""
        if not game_id:
            continue
        ac: dict[str, int] = {k: int(v) for k, v in (run.get("action_counts") or {}).items()}
        # dominant action: action with highest count (excluding RESET)
        dominant: str | None = None
        dom_count = 0
        for act, cnt in ac.items():
            if act != "RESET" and cnt > dom_count:
                dom_count, dominant = cnt, act

        sc_levels = int(run.get("scorecard_levels_completed") or 0)
        sc_actions = int(run.get("scorecard_actions") or 0)
        is_zero = sc_levels == 0 and sc_actions >= 100

        priors[game_id] = {
            "boundary": outer_boundary,
            "dominant_action": dominant,
            "level_progressions": run.get("level_progressions") or [],
            "action_counts": ac,
            "zero_score": is_zero,
            "levels_completed": sc_levels,
            "scorecard_score": run.get("scorecard_score"),
        }
    return priors


# ---------------------------------------------------------------------------
# Per-packet analysis
# ---------------------------------------------------------------------------

def _analyse_packet(
    packet: dict[str, Any],
    packet_idx: int,
    run_dir_rel: str,
    arc_priors: dict[str, Any],
) -> Gap | None:
    """Analyse one packet and return a Gap (or None if nothing notable).

    Checks (in order):
      1. premature_activate_flow    — activate_flow fired but delta==0
      2. portband_lock_absent       — portband score never reaches 0.80 threshold
      3. predicted_not_predictive   — pipeline says prediction_status=not_predictive
      4. arc_opus48_routing_mismatch— dominant action prior disagrees with pipeline
      5. cov_zero_every_step        — caller collects this across packets

    Returns None for observation_only packets (nothing divergent).
    """
    game_id: str = (packet.get("contra_packet") or {}).get("game_id") or ""
    level_index: int = packet.get("contra_packet", {}).get("level_index") or 0
    action: dict = packet.get("action") or {}
    prim: str = packet.get("decision_primitive") or action.get("primitive") or ""
    levels_delta: int = int(packet.get("levels_completed_delta") or 0)

    crp = packet.get("contra_recommendation_packet") or {}
    top_rec = _extract_top_rec(packet)
    top_kind: str = top_rec.get("candidate_kind") or crp.get("top_primitive") or ""
    top_vs: float = float(crp.get("top_value_score") or 0.0)
    pred_status: str = crp.get("prediction_status") or ""

    # Load frames for observed pixel delta
    bf_path = (packet.get("before_frame") or {}).get("raw_grid", {}).get("path", "")
    af_path = (packet.get("after_frame") or {}).get("raw_grid", {}).get("path", "")
    before_frame = _load_npy_frame(bf_path)
    after_frame = _load_npy_frame(af_path)
    px_delta = _pixel_delta(before_frame, after_frame)

    db = packet.get("deltaband_observations") or {}
    surface_events: list = db.get("action_surface_result_events") or []

    pb_score = _portband_max_score(packet)

    # ------------------------------------------------------------------ #
    # 1.  Premature activate_flow
    # ------------------------------------------------------------------ #
    if prim == "activate_flow" and levels_delta == 0:
        return Gap(
            gap_id=_sha256_str(run_dir_rel, level_index, packet_idx, "premature_activate_flow"),
            game_id=game_id,
            level_index=level_index,
            action_index=packet_idx,
            action_primitive=prim,
            candidate_kind=top_kind,
            predicted_effect=(
                f"activate_flow expected to complete level "
                f"(top_kind={top_kind}, vs={top_vs:.3f})"
            ),
            observed_effect=(
                f"activate_flow fired but levels_completed_delta=0; "
                f"changed_pixels={px_delta.get('changed_pixel_count', '?')}"
            ),
            divergence_kind="premature_activate_flow",
            divergence_score=0.85,
            evidence_detail={
                "top_candidate_kind": top_kind,
                "top_value_score": top_vs,
                "pixel_delta": px_delta,
                "surface_events_count": len(surface_events),
                "prediction_status": pred_status,
            },
            source_run_dir=run_dir_rel,
        )

    # ------------------------------------------------------------------ #
    # 2.  PortBand lock absent  (portband score never reaches 0.80)
    # ------------------------------------------------------------------ #
    if pb_score < 0.80 and prim in ("move_up", "move_down", "move_left", "move_right"):
        if top_kind and "portband" not in top_kind:
            # pipeline is NOT using portband even for directional moves → gap
            return Gap(
                gap_id=_sha256_str(run_dir_rel, level_index, packet_idx, "portband_lock_absent"),
                game_id=game_id,
                level_index=level_index,
                action_index=packet_idx,
                action_primitive=prim,
                candidate_kind=top_kind,
                predicted_effect=(
                    f"pipeline expected {top_kind} (vs={top_vs:.3f}) to guide {prim}; "
                    f"no portband lock (pb_score={pb_score:.3f} < 0.80 threshold)"
                ),
                observed_effect=(
                    f"{prim} executed with {px_delta.get('changed_pixel_count', '?')} "
                    f"changed pixels; no stable edge-lock (L0 achieves 0.82)"
                ),
                divergence_kind="portband_lock_absent",
                divergence_score=0.70,
                evidence_detail={
                    "top_candidate_kind": top_kind,
                    "top_value_score": top_vs,
                    "portband_max_score": pb_score,
                    "pixel_delta": px_delta,
                    "surface_events_count": len(surface_events),
                    "l0_reference_vs": 0.818,
                    "note": "L0 holds portband_lineup_matrix_row vs=0.82 stably; L1 never reaches it",
                },
                source_run_dir=run_dir_rel,
            )

    # ------------------------------------------------------------------ #
    # 3.  Pipeline says prediction_status=not_predictive
    # ------------------------------------------------------------------ #
    if pred_status == "not_predictive" and top_vs > 0.3:
        return Gap(
            gap_id=_sha256_str(run_dir_rel, level_index, packet_idx, "predicted_not_predictive"),
            game_id=game_id,
            level_index=level_index,
            action_index=packet_idx,
            action_primitive=prim,
            candidate_kind=top_kind,
            predicted_effect=(
                f"pipeline has no forward model "
                f"(prediction_status=not_predictive, top_kind={top_kind}, vs={top_vs:.3f})"
            ),
            observed_effect=(
                f"{prim} produced {px_delta.get('changed_pixel_count', '?')} changed pixels; "
                f"pipeline could not have predicted this"
            ),
            divergence_kind="predicted_not_predictive",
            divergence_score=0.60,
            evidence_detail={
                "top_candidate_kind": top_kind,
                "top_value_score": top_vs,
                "prediction_status": pred_status,
                "pixel_delta": px_delta,
            },
            source_run_dir=run_dir_rel,
        )

    # ------------------------------------------------------------------ #
    # 4.  arc_opus48 action-routing mismatch
    # ------------------------------------------------------------------ #
    game_prior = arc_priors.get(game_id) or {}
    dominant = game_prior.get("dominant_action")
    if dominant and dominant != (action.get("name") or action.get("id") or ""):
        # pipeline chose something other than the verified dominant action
        return Gap(
            gap_id=_sha256_str(run_dir_rel, level_index, packet_idx, "arc_opus48_routing_mismatch"),
            game_id=game_id,
            level_index=level_index,
            action_index=packet_idx,
            action_primitive=prim,
            candidate_kind=top_kind,
            predicted_effect=(
                f"pipeline scored {top_kind}→{action.get('name', prim)} "
                f"(vs={top_vs:.3f}) as best"
            ),
            observed_effect=(
                f"arc_opus48 prior: dominant_action={dominant} "
                f"(game completed {game_prior.get('levels_completed')} levels with it); "
                f"pipeline chose {action.get('name', prim)} instead"
            ),
            divergence_kind="arc_opus48_routing_mismatch",
            divergence_score=0.45,
            evidence_detail={
                "top_candidate_kind": top_kind,
                "top_value_score": top_vs,
                "pipeline_action": action.get("name") or prim,
                "arc_dominant_action": dominant,
                "arc_levels_completed": game_prior.get("levels_completed"),
                "arc_level_progressions": game_prior.get("level_progressions"),
                "pixel_delta": px_delta,
            },
            source_run_dir=run_dir_rel,
            prior_refs=[{
                "source": "arc_opus48_playthrough_mining",
                "game_id": game_id,
                "field": "dominant_action",
                "value": dominant,
                "boundary": BOUNDARY,
            }],
        )

    return None  # observation_only


# ---------------------------------------------------------------------------
# Run-dir mining
# ---------------------------------------------------------------------------

def _candidate_kind_diversity(packets: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute candidate_kind diversity stats across a packet sequence."""
    kinds: list[str] = []
    for p in packets:
        crp = p.get("contra_recommendation_packet") or {}
        recs = crp.get("ranked_recommendations") or []
        top = recs[0] if recs else {}
        k = top.get("candidate_kind") or ""
        if k:
            kinds.append(k)
    if not kinds:
        return {"unique_count": 0, "kinds": [], "thrash_detected": False}
    unique = list(dict.fromkeys(kinds))   # preserve order, deduplicate
    # thrash: >=4 distinct kinds in a 6-packet window, or overall >=5 unique
    thrash = len(unique) >= 5 or (
        len(kinds) >= 6 and len(set(kinds[-6:])) >= 4
    )
    return {
        "unique_count": len(unique),
        "kinds": unique,
        "sequence": kinds,
        "thrash_detected": thrash,
    }


def mine_run_dir(
    run_dir: Path,
    level_index: int,
    arc_priors: dict[str, Any],
) -> list[Gap]:
    """Mine one level_NN/ dir inside a run_dir and return a list of Gap records.

    Loads packets.jsonl, iterates every packet, runs per-packet analysis, then
    adds run-level gap kinds (cov_zero_every_step, candidate_kind_thrash,
    level_budget_exhausted) if applicable.

    run_dir may be the episode-level root (…episode_levels_XX/Y/) or a
    level_XX/ subdir.  The function normalises to the level_NN/ dir.
    """
    # Resolve level dir
    level_dir = run_dir
    candidate_level_dir = run_dir / f"level_{level_index:02d}"
    if candidate_level_dir.is_dir():
        level_dir = candidate_level_dir

    packets_path = level_dir / "packets.jsonl"
    result_path = level_dir / "result.json"
    if not packets_path.is_file():
        return []

    run_dir_rel = str(level_dir.relative_to(ROOT))

    # Load packets
    packets: list[dict[str, Any]] = []
    try:
        for line in packets_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                packets.append(json.loads(line))
    except Exception as exc:
        return [Gap(
            gap_id=_sha256_str(run_dir_rel, level_index, 0, "parse_error"),
            game_id="unknown",
            level_index=level_index,
            action_index=0,
            action_primitive="unknown",
            candidate_kind="unknown",
            predicted_effect="",
            observed_effect=f"packets.jsonl parse error: {exc}",
            divergence_kind="observation_only",
            divergence_score=0.0,
            source_run_dir=run_dir_rel,
        )]

    if not packets:
        return []

    # Load result to get episode outcome
    result: dict[str, Any] = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    gaps: list[Gap] = []

    # Per-packet analysis
    for idx, packet in enumerate(packets):
        gap = _analyse_packet(packet, idx, run_dir_rel, arc_priors)
        if gap is not None:
            gaps.append(gap)

    # Run-level: candidate_kind thrash
    diversity = _candidate_kind_diversity(packets)
    if diversity.get("thrash_detected"):
        game_id = (packets[0].get("contra_packet") or {}).get("game_id") or ""
        gaps.append(Gap(
            gap_id=_sha256_str(run_dir_rel, level_index, 0, "candidate_kind_thrash"),
            game_id=game_id,
            level_index=level_index,
            action_index=-1,   # run-level, not per-packet
            action_primitive="(multiple)",
            candidate_kind="(multiple)",
            predicted_effect=(
                f"pipeline expected stable solver dominance; "
                f"top candidate changed every packet"
            ),
            observed_effect=(
                f"detected {diversity['unique_count']} distinct candidate_kinds "
                f"across {len(packets)} packets: {diversity['kinds'][:8]}..."
                if diversity['unique_count'] > 8 else
                f"detected {diversity['unique_count']} distinct candidate_kinds "
                f"across {len(packets)} packets: {diversity['kinds']}"
            ),
            divergence_kind="candidate_kind_thrash",
            divergence_score=0.90,
            evidence_detail={
                "diversity": diversity,
                "packet_count": len(packets),
                "levels_completed_delta": int(result.get("levels_completed_delta") or 0),
                "note": (
                    "L0 reference: portband_lineup_matrix_row dominates 7/16 packets "
                    "with vs≥0.62, stable across 5 corroborating runs. "
                    "L1 never achieves this stability."
                ),
            },
            source_run_dir=run_dir_rel,
        ))

    # Run-level: cov_zero_every_step — every packet changed_pixel_count == 0
    # This is the forward-model grounding bug (wrong actor/seed color).
    if _NUMPY_OK and packets:
        zero_count = 0
        for packet in packets:
            bf_path = (packet.get("before_frame") or {}).get("raw_grid", {}).get("path", "")
            af_path = (packet.get("after_frame") or {}).get("raw_grid", {}).get("path", "")
            before = _load_npy_frame(bf_path)
            after = _load_npy_frame(af_path)
            if before is not None and after is not None:
                changed = int(np.sum(before != after))
                if changed == 0:
                    zero_count += 1
        n_frames_checked = sum(
            1 for p in packets
            if _load_npy_frame((p.get("before_frame") or {}).get("raw_grid", {}).get("path", "")) is not None
        )
        if n_frames_checked > 0 and zero_count == n_frames_checked:
            game_id = (packets[0].get("contra_packet") or {}).get("game_id") or ""
            gaps.append(Gap(
                gap_id=_sha256_str(run_dir_rel, level_index, 0, "cov_zero_every_step"),
                game_id=game_id,
                level_index=level_index,
                action_index=-1,
                action_primitive="(all)",
                candidate_kind="(all)",
                predicted_effect=(
                    "forward model predicts pixel changes (non-zero coverage expected "
                    "for any meaningful action on this level)"
                ),
                observed_effect=(
                    f"ALL {n_frames_checked} checked frames show 0 changed pixels — "
                    "classic grounding bug: wrong actor seed color or wrong game physics model. "
                    "NOTE: blue platform is the actor, NOT magenta liquid."
                ),
                divergence_kind="cov_zero_every_step",
                divergence_score=1.0,
                evidence_detail={
                    "zero_pixel_packets": zero_count,
                    "total_checked_packets": n_frames_checked,
                    "cov_rate": 0.0,
                    "warning": (
                        "ACTOR IDENTITY: blue platform is the actor (BLUE, color value 9 "
                        "in SP80); magenta is liquid (color 8). "
                        "Calibration bug: user_supplied_answer seeds magenta as actor."
                    ),
                },
                source_run_dir=run_dir_rel,
            ))

    # Run-level: budget_exhausted with delta=0
    if (result.get("status") == "budget_exhausted"
            and int(result.get("levels_completed_delta") or 0) == 0):
        game_id = (packets[0].get("contra_packet") or {}).get("game_id") or ""
        gaps.append(Gap(
            gap_id=_sha256_str(run_dir_rel, level_index, 0, "level_budget_exhausted"),
            game_id=game_id,
            level_index=level_index,
            action_index=-1,
            action_primitive="(episode)",
            candidate_kind="(episode)",
            predicted_effect=(
                f"CONTRA expected to complete level {level_index} within "
                f"{result.get('max_actions', '?')} actions"
            ),
            observed_effect=(
                f"budget_exhausted after {result.get('action_count', '?')} actions, "
                f"levels_completed_delta=0"
            ),
            divergence_kind="level_budget_exhausted",
            divergence_score=0.80,
            evidence_detail={
                "action_count": result.get("action_count"),
                "max_actions": result.get("max_actions"),
                "final_status": result.get("status"),
                "result_schema": result.get("schema"),
            },
            source_run_dir=run_dir_rel,
        ))

    return gaps


# ---------------------------------------------------------------------------
# GapFinder
# ---------------------------------------------------------------------------

class GapFinder:
    """Discover prediction-vs-observation divergences across CONTRA run dirs.

    Usage:
        gf = GapFinder(run_dirs=[Path("codexlab/runs/contra_sp80_ep.../")],
                       arc_opus48_summary=None)
        gaps = gf.find_gaps(game_id="sp80-589a99af", level_index=1)
        report = gf.emit_gap_report(gaps, out_path=Path("gap_report.json"))
    """

    def __init__(
        self,
        run_dirs: list[Path],
        arc_opus48_summary: Path | None = None,
    ) -> None:
        self.run_dirs = run_dirs
        self._arc_priors = load_arc_opus48_priors(arc_opus48_summary)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def arc_priors_for(self, game_id: str) -> dict[str, Any]:
        """Return the arc_opus48 prior for a game, or {} if not available."""
        return self._arc_priors.get(game_id) or {}

    def find_gaps(
        self,
        game_id: str,
        level_index: int,
    ) -> list[Gap]:
        """Mine all run_dirs for gaps on (game_id, level_index).

        Gaps are deduplicated by gap_id and sorted by divergence_score DESC.
        """
        all_gaps: dict[str, Gap] = {}
        for run_dir in self.run_dirs:
            found = mine_run_dir(run_dir, level_index, self._arc_priors)
            for gap in found:
                if gap.game_id and game_id and gap.game_id != game_id:
                    continue  # skip other games
                if gap.game_id == "" or game_id == "":
                    pass  # unknown game_id — include
                all_gaps[gap.gap_id] = gap

        return sorted(all_gaps.values(), key=lambda g: g.divergence_score, reverse=True)

    def emit_gap_report(
        self,
        gaps: list[Gap],
        out_path: Path | None = None,
    ) -> dict[str, Any]:
        """Serialise gaps to gap_report.json and return the report dict.

        The report is the ONLY allowed hypothesis seed for the CONTRA loop.
        The agent reads this file; it must NOT form a hypothesis from any
        other source of abstract reasoning.

        The boundary string appears both at the top level AND on each gap
        record — it cannot be stripped.
        """
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "boundary": BOUNDARY,
            "gap_count": len(gaps),
            "gap_report_sha256_basis": _sha256_str(
                *(g.gap_id for g in gaps)
            ),
            "agent_instructions": (
                "Read ONE high-divergence_score gap. "
                "Form a HypothesisEnvelope naming: "
                "action_primitive, precondition_signature, predicted_effect, "
                "falsification_condition, and verbatim boundary string. "
                "This report is the ONLY allowed hypothesis seed. "
                "Do NOT form a hypothesis from abstract reasoning."
            ),
            "arc_opus48_priors_loaded": bool(self._arc_priors),
            "arc_opus48_games_with_priors": sorted(self._arc_priors.keys()),
            "gaps": [asdict(g) for g in gaps],
        }

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        return report


# ---------------------------------------------------------------------------
# Convenience: discover CONTRA run dirs automatically
# ---------------------------------------------------------------------------

def discover_contra_run_dirs(
    runs_root: Path | None = None,
    game_short: str = "sp80",
    max_dirs: int = 10,
) -> list[Path]:
    """Glob for CONTRA episode-level run dirs under runs_root.

    Returns the most-recent max_dirs matching dirs, sorted newest first.
    """
    root = runs_root or (ROOT / "codexlab" / "runs")
    if not root.is_dir():
        return []
    pattern = f"*{game_short}*episode*"
    dirs = sorted(
        [d for d in root.glob(pattern) if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[:max_dirs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        metavar="PATH",
        help=(
            "CONTRA episode run dir to mine (e.g. "
            "codexlab/runs/contra_sp80_episode_levels_00_01_YYYYMMDDTHHMMSSZ). "
            "Relative to repo root. May be repeated."
        ),
        action="append",
        default=[],
    )
    p.add_argument(
        "--level",
        type=int,
        default=1,
        metavar="N",
        help="Level index to analyse (default: 1)",
    )
    p.add_argument(
        "--game",
        default="sp80-589a99af",
        help="Game ID to filter gaps (default: sp80-589a99af)",
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        default=None,
        help="Write gap_report.json to this path (default: print to stdout)",
    )
    p.add_argument(
        "--arc-summary",
        metavar="FILE",
        default=None,
        help=(
            f"Path to arc_opus48 summary.json "
            f"(default: {ARC_OPUS48_SUMMARY_REL})"
        ),
    )
    p.add_argument(
        "--discover",
        action="store_true",
        help=(
            "Auto-discover CONTRA run dirs under codexlab/runs/ for --game "
            "and mine all of them."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print what would be analysed without mining any run dirs or "
            "writing any files. Validates arc_opus48 prior loading."
        ),
    )
    return p


def main() -> int:
    p = _build_parser()
    args = p.parse_args()

    arc_path: Path | None = Path(args.arc_summary) if args.arc_summary else None

    # ------------------------------------------------------------------ #
    # Dry-run: validate priors + print plan, exit 0
    # ------------------------------------------------------------------ #
    if args.dry_run:
        priors = load_arc_opus48_priors(arc_path)
        game_short = args.game.split("-")[0]
        run_dirs = discover_contra_run_dirs(game_short=game_short) if args.discover else [
            ROOT / r for r in args.run_dir
        ]
        if not run_dirs and not args.run_dir:
            run_dirs = discover_contra_run_dirs(game_short=game_short, max_dirs=3)

        prior_str = (
            f"arc_opus48 prior loaded for {args.game}: "
            f"dominant_action={priors.get(args.game, {}).get('dominant_action')}, "
            f"levels_completed={priors.get(args.game, {}).get('levels_completed')}"
            if args.game in priors else
            f"arc_opus48 prior NOT found for {args.game} "
            f"(available: {sorted(priors.keys())[:5]}...)"
        )

        print(json.dumps({
            "dry_run": True,
            "schema": SCHEMA,
            "boundary": BOUNDARY,
            "game_id": args.game,
            "level_index": args.level,
            "run_dirs_to_mine": [str(d) for d in run_dirs],
            "run_dirs_found": len(run_dirs),
            "arc_opus48_summary": str(arc_path or ROOT / ARC_OPUS48_SUMMARY_REL),
            "arc_opus48_priors_available": len(priors),
            "arc_prior_summary": prior_str,
            "numpy_available": _NUMPY_OK,
            "out_path": args.out or "(stdout)",
            "note": (
                "Pass --run-dir <path> [--run-dir <path2>] or --discover to mine "
                "real run dirs. This dry-run validates the prior-loading path only."
            ),
        }, indent=2))
        return 0

    # ------------------------------------------------------------------ #
    # Collect run dirs
    # ------------------------------------------------------------------ #
    game_short = args.game.split("-")[0]

    if args.discover:
        run_dirs = discover_contra_run_dirs(game_short=game_short)
        if not run_dirs:
            print(
                f"[gap_finder] No CONTRA run dirs found under "
                f"{ROOT / 'codexlab' / 'runs'} matching *{game_short}*episode*",
                file=sys.stderr,
            )
            return 1
    elif args.run_dir:
        run_dirs = [ROOT / r for r in args.run_dir]
    else:
        # default: use the canonical reference run dir if present
        default_run = ROOT / "codexlab/runs/contra_sp80_episode_levels_00_01_20260608T175142Z"
        if default_run.is_dir():
            run_dirs = [default_run]
        else:
            print(
                "[gap_finder] No run dirs specified. "
                "Use --run-dir <path>, --discover, or --dry-run.",
                file=sys.stderr,
            )
            return 1

    # ------------------------------------------------------------------ #
    # Mine
    # ------------------------------------------------------------------ #
    gf = GapFinder(run_dirs=run_dirs, arc_opus48_summary=arc_path)
    gaps = gf.find_gaps(game_id=args.game, level_index=args.level)

    if not gaps:
        print(
            f"[gap_finder] No gaps found for game={args.game} level={args.level} "
            f"in {len(run_dirs)} run dir(s). "
            "Try --discover to mine more runs, or --level 0 for the solved level.",
            file=sys.stderr,
        )

    out_path: Path | None = Path(args.out) if args.out else None
    report = gf.emit_gap_report(gaps, out_path=out_path)

    if out_path:
        print(
            f"[gap_finder] Wrote {out_path} "
            f"({report['gap_count']} gaps, top divergence_kind="
            f"{gaps[0].divergence_kind if gaps else 'none'})"
        )
    else:
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
