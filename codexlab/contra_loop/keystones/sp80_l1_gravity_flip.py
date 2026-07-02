#!/usr/bin/env python3
"""Iteration-1 keystone: test the global_inversion_fill (gravity-flip) forward model.

HYPOTHESIS (sp80_l1_gravity_flip)
  action_primitive:         activate_flow (ACTION5, id=5)
  precondition_signature:   actor on gravity-source edge + nonempty liquid mask
                            (color-9 block present in board, rows 36-39 in canonical
                            L1 layout)
  predicted_effect:         whole-board vertical inversion of the liquid mask
                            (np.flipud applied to the color-9 pixel mask); the liquid
                            block should appear at the mirrored row position after the
                            pour (row r -> row 63-r for a 64-pixel board)
  falsification_condition:  observed after-frame liquid mask != flipud(before-frame
                            liquid mask) on >= 1 L1 activate_flow transition in the
                            win-bank ground-truth corpus

This keystone loads ALL L1 activate_flow (ACTION5) before/after frame pairs from
the human win-bank, applies the forward model, and reports:

  * exact_frame_matches N/M  -- whole-board np.flipud exact frame match
  * liquid_mask_iou[i]       -- per-pair IoU on the color-9 mask only
  * mean_liquid_mask_iou     -- headline metric; 0.0 on every step trips StallDetector(c)

CARDINAL RULE HONORED:
  admission_status: discovery_only  -- even N/N here is NOT a solve.
  A solve requires levels_completed_delta>0 from a verified episode run.

GROUND-TRUTH SOURCE: win_bank (primary).
  Path: human/win_bank_2026-05-16_human_cheat_map_v1/sp80-589a99af_wins.pkl
  SHA-256: e8fa00464434edff9fdbf369318d09cb5e6c43b76e47298455137ae0d4cce848
  Content: 6 human wins, won_levels=[4,4,4,5,5,6], all seed=0.
  Fallback: engine_source via sp80.py step() -- see TODO below.

Usage:
    python codexlab/contra_loop/keystones/sp80_l1_gravity_flip.py --dry-run
    python codexlab/contra_loop/keystones/sp80_l1_gravity_flip.py
    python codexlab/contra_loop/keystones/sp80_l1_gravity_flip.py --win-bank /path/to/sp80-589a99af_wins.pkl
    python codexlab/contra_loop/keystones/sp80_l1_gravity_flip.py --json-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root anchoring (codexlab/ is the package root; this file lives inside it)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]          # rexus/
CODEXLAB = ROOT / "codexlab"
WIN_BANK_PATH = ROOT / "human" / "win_bank_2026-05-16_human_cheat_map_v1" / "sp80-589a99af_wins.pkl"
WIN_BANK_SHA256 = "e8fa00464434edff9fdbf369318d09cb5e6c43b76e47298455137ae0d4cce848"

# Action id for activate_flow in the SP80 harness
ACTION5_ID = 5  # activate_flow == ACTION5 == GameAction.ACTION5

# Level indices (0-based) in the win-bank 'levels' list
L1_INDEX = 1

# The "liquid color" in SP80 raw pixel values for L1.
# Color 9 (raw pixel value 9) is the movable block that represents the
# liquid flow source in L1.  Color 6 (magenta in ARC palette) is static.
LIQUID_COLOR = 9

# Board dimensions for SP80 (all frames are 64x64 uint8/int8)
BOARD_H = 64
BOARD_W = 64

# ---------------------------------------------------------------------------
# Hypothesis envelope (4-field boundary schema; carry verbatim into the registry)
# ---------------------------------------------------------------------------
HYPOTHESIS: dict[str, Any] = {
    "name": "sp80_l1_gravity_flip",
    "source_signal": (
        "gap_report: local connected-flood forward model predicts cov=0/3 on L1 "
        "activate_flow transitions; engine shows color-9 block jumping from rows "
        "36-39 to rows 24-27 (row r -> 63-r, matching np.flipud) on 2/6 winning L1 episodes"
    ),
    "implementation_target": "contra_head/portband.py build_portband_surface_feedback or a new forward_model module",
    "boundary": "Hypothesis material only; not local CONTRA evidence.",
    "action_primitive": "activate_flow",
    "precondition_signature": "actor on gravity-source edge + nonempty liquid mask (color 9)",
    "predicted_effect": (
        "whole-board vertical inversion of the liquid mask (np.flipud); "
        "level_up IFF every bucket covered post-inversion"
    ),
    "falsification_condition": (
        "observed after-frame liquid mask != flipud(before-frame liquid mask) "
        "on >= 1 L1 activate_flow transition"
    ),
    "mechanic_class": "global_inversion_fill",
    "admission_status": "discovery_only",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FramePair:
    """One ground-truth before/after pair from an L1 activate_flow transition."""
    win_index: int
    l1_step: int
    frame_index: int          # index into win-bank entry 'frames' list
    before: np.ndarray        # shape (H, W), dtype int8 or uint8
    after: np.ndarray         # shape (H, W), dtype int8 or uint8
    win_total_levels: int


@dataclass
class PairResult:
    """Forward-model evaluation for one FramePair."""
    win_index: int
    l1_step: int
    exact_frame_match: bool
    liquid_mask_iou: float    # IoU on color-9 mask only
    n_pixels_changed: int     # actual before->after diff
    liquid_before_count: int  # how many color-9 pixels in before
    liquid_after_count: int   # how many color-9 pixels in actual after
    liquid_pred_count: int    # how many color-9 pixels in predicted after


@dataclass
class KeystoneResult:
    """Aggregated outcome from all tested pairs."""
    hypothesis_name: str
    ground_truth_source: str
    win_bank_sha256: str
    n_pairs_tested: int
    exact_frame_matches: int
    pair_results: list[PairResult] = field(default_factory=list)

    @property
    def exact_match_fraction(self) -> str:
        return f"{self.exact_frame_matches}/{self.n_pairs_tested}"

    @property
    def mean_liquid_mask_iou(self) -> float:
        if not self.pair_results:
            return 0.0
        return float(np.mean([r.liquid_mask_iou for r in self.pair_results]))

    @property
    def stall_candidate(self) -> bool:
        """True if mean IoU is 0 on every pair -- StallDetector(c) cov=0/N trigger."""
        return self.mean_liquid_mask_iou == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "codexlab.contra_loop.keystone_result.v1",
            "keystone": self.hypothesis_name,
            "ground_truth_source": self.ground_truth_source,
            "win_bank_sha256": self.win_bank_sha256,
            "n_pairs_tested": self.n_pairs_tested,
            "exact_frame_matches": self.exact_match_fraction,
            "per_pair": [
                {
                    "win_index": r.win_index,
                    "l1_step": r.l1_step,
                    "exact_frame_match": r.exact_frame_match,
                    "liquid_mask_iou": round(r.liquid_mask_iou, 4),
                    "n_pixels_changed": r.n_pixels_changed,
                    "liquid_before_count": r.liquid_before_count,
                    "liquid_after_count": r.liquid_after_count,
                    "liquid_pred_count": r.liquid_pred_count,
                }
                for r in self.pair_results
            ],
            "mean_liquid_mask_iou": round(self.mean_liquid_mask_iou, 4),
            "stall_cov0_candidate": self.stall_candidate,
            "admission_status": "discovery_only",
            "boundary": HYPOTHESIS["boundary"],
        }


# ---------------------------------------------------------------------------
# Win-bank loading
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_win_bank(path: Path) -> list[dict[str, Any]]:
    """Load the SP80 win-bank pickle and return the list of win entries.

    Each entry is a dict with keys:
      game_id, episode, seed, human_played_utc, frames, actions, levels,
      states, won_levels, final_state

    frames[i] : np.ndarray shape (64, 64) dtype int8
    actions[i]: dict with keys id, x, y, complex
    levels[i] : int  -- the game level index at the time of action i
    """
    if not path.is_file():
        raise FileNotFoundError(f"win-bank not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"win-bank expected list, got {type(data)}")
    return data


def extract_l1_activate_flow_pairs(
    win_bank: list[dict[str, Any]],
) -> list[FramePair]:
    """Return all (before, after) frame pairs where the action is activate_flow
    (ACTION5, id=5) and the game is on level 1.

    A pair is included if:
      - levels[i] == L1_INDEX (currently on L1)
      - actions[i]['id'] == ACTION5_ID
      - frames[i+1] exists (there is a subsequent frame)

    The 'after' frame is frames[i+1]; the game state at the start of the next
    action.  NOTE: in the win-bank, frames[i] is the board STATE at action step
    i (before the action resolves), and frames[i+1] is the board state after the
    action completes.
    """
    pairs: list[FramePair] = []
    for win_idx, entry in enumerate(win_bank):
        levels = entry.get("levels", [])
        actions = entry.get("actions", [])
        frames = entry.get("frames", [])
        won_levels = int(entry.get("won_levels", 0))
        for step_idx, (lvl, act, fi) in enumerate(zip(levels, actions, range(len(frames)))):
            if lvl != L1_INDEX:
                continue
            if act.get("id") != ACTION5_ID:
                continue
            if fi + 1 >= len(frames):
                continue
            before = np.asarray(frames[fi], dtype=np.int16)
            after = np.asarray(frames[fi + 1], dtype=np.int16)
            pairs.append(FramePair(
                win_index=win_idx,
                l1_step=step_idx,
                frame_index=fi,
                before=before,
                after=after,
                win_total_levels=won_levels,
            ))
    return pairs


# ---------------------------------------------------------------------------
# Forward model: whole-board vertical inversion
# ---------------------------------------------------------------------------

def gravity_flip_forward_model(before: np.ndarray) -> np.ndarray:
    """Predicted after-frame under the global_inversion_fill hypothesis.

    The hypothesis: when activate_flow fires, the whole board is inverted
    vertically (np.flipud).  Row r maps to row (BOARD_H - 1 - r).

    This is the MINIMAL interpretation of "whole-board inversion" -- it flips
    the entire pixel grid.  An alternative is flipping only the liquid mask;
    see _gravity_flip_liquid_only() below.
    """
    return np.flipud(before)


def _gravity_flip_liquid_only(
    before: np.ndarray, liquid_color: int = LIQUID_COLOR
) -> np.ndarray:
    """Predicted after-frame under a partial-inversion model:
    only the liquid mask (color == liquid_color) is vertically flipped;
    all other pixels remain in place.

    This variant tests whether the row-flip is correct for the liquid alone,
    even if the rest of the board does not flip.
    """
    liquid_mask = before == liquid_color
    # Background value where liquid was (most common non-liquid value)
    bg = int(np.bincount(before[~liquid_mask].flatten().astype(int)).argmax())
    pred = before.copy()
    pred[liquid_mask] = bg          # remove liquid from old positions
    pred[np.flipud(liquid_mask)] = liquid_color  # place at flipped positions
    return pred


# ---------------------------------------------------------------------------
# IoU utility
# ---------------------------------------------------------------------------

def mask_iou(pred: np.ndarray, actual: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    inter = int((pred & actual).sum())
    union = int((pred | actual).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(pair: FramePair) -> PairResult:
    """Apply both forward-model variants to one pair and compute metrics."""
    before = pair.before
    after = pair.after

    # --- Whole-board flipud ---
    pred_whole = gravity_flip_forward_model(before)
    exact_whole = bool(np.array_equal(pred_whole, after))

    # --- Liquid-mask flipud (for IoU headline) ---
    liq_before = before == LIQUID_COLOR
    liq_after = after == LIQUID_COLOR
    liq_pred = np.flipud(liq_before)

    iou = mask_iou(liq_pred, liq_after)

    return PairResult(
        win_index=pair.win_index,
        l1_step=pair.l1_step,
        exact_frame_match=exact_whole,
        liquid_mask_iou=iou,
        n_pixels_changed=int((after != before).sum()),
        liquid_before_count=int(liq_before.sum()),
        liquid_after_count=int(liq_after.sum()),
        liquid_pred_count=int(liq_pred.sum()),
    )


# ---------------------------------------------------------------------------
# Main keystone runner
# ---------------------------------------------------------------------------

def run_keystone(win_bank_path: Path) -> KeystoneResult:
    """Load the win-bank, extract L1 activate_flow pairs, evaluate the hypothesis.

    Returns a KeystoneResult with all metrics.  Always returns (never raises for
    hypothesis failure); metric values may be 0.0 / 0/N indicating falsification.
    """
    # 1. Verify win-bank integrity
    actual_sha256 = _sha256_file(win_bank_path)
    sha_ok = actual_sha256 == WIN_BANK_SHA256

    # 2. Load data
    win_bank = load_win_bank(win_bank_path)

    # 3. Extract pairs
    pairs = extract_l1_activate_flow_pairs(win_bank)
    if not pairs:
        # No pairs found -- this is a grounding error, not a hypothesis falsification
        raise ValueError(
            "No L1 activate_flow (ACTION5) pairs found in the win-bank.  "
            "Check win-bank path and that at least one entry wins L1."
        )

    # 4. Evaluate each pair
    results = [evaluate_pair(p) for p in pairs]

    # 5. Aggregate
    exact_matches = sum(1 for r in results if r.exact_frame_match)
    ks = KeystoneResult(
        hypothesis_name=HYPOTHESIS["name"],
        ground_truth_source="win_bank",
        win_bank_sha256=actual_sha256 if not sha_ok else WIN_BANK_SHA256,
        n_pairs_tested=len(results),
        exact_frame_matches=exact_matches,
        pair_results=results,
    )
    return ks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(ks: KeystoneResult, *, json_only: bool) -> None:
    """Print the keystone result.  Always prints a machine-parseable JSON block."""
    if not json_only:
        print("=" * 72)
        print(f"KEYSTONE  {ks.hypothesis_name}")
        print(f"  mechanic_class : {HYPOTHESIS['mechanic_class']}")
        print(f"  action_primitive: {HYPOTHESIS['action_primitive']}")
        print(f"  forward_model  : np.flipud(before_frame)  [whole-board vertical inversion]")
        print(f"  liquid_color   : {LIQUID_COLOR}  (raw pixel value, not ARC index)")
        print(f"  ground_truth   : {ks.ground_truth_source}  (sha256 {ks.win_bank_sha256[:16]}...)")
        print("-" * 72)
        print(f"  pairs_tested   : {ks.n_pairs_tested}")
        for r in ks.pair_results:
            print(
                f"    win={r.win_index} step={r.l1_step:3d}: "
                f"exact={str(r.exact_frame_match):<5s}  "
                f"liq_iou={r.liquid_mask_iou:.3f}  "
                f"changed={r.n_pixels_changed:4d}px  "
                f"liq_b={r.liquid_before_count}->pred={r.liquid_pred_count}->"
                f"actual={r.liquid_after_count}"
            )
        print("-" * 72)
        print(f"  exact_frame_matches : {ks.exact_match_fraction}")
        print(f"  mean_liquid_mask_iou: {ks.mean_liquid_mask_iou:.4f}")
        if ks.stall_candidate:
            print()
            print("  [STALL CANDIDATE] mean IoU = 0.0 across all pairs.")
            print("  This is the cov=0/N pattern -- the forward model predicts 0 effect")
            print("  on every step.  Possible causes:")
            print("    (a) wrong liquid color (check: is color 9 actually the flow element?)")
            print("    (b) wrong flip axis (vertical flip vs horizontal flip)")
            print("    (c) the gravity-flip hypothesis is falsified for this level")
            print("    (d) wrong actor/seed color (see: SP80 actor truth note)")
        else:
            print()
            print(f"  Mean IoU = {ks.mean_liquid_mask_iou:.4f} (non-zero: row-flip direction")
            print("  is partially correct but the column range does not match exactly.)")
            print("  Hypothesis is partially supported but NOT confirmed.")
        print()
        print(f"  admission_status: {HYPOTHESIS['admission_status']}")
        print(f"  boundary: {HYPOTHESIS['boundary']}")
        print("=" * 72)

    # Machine-parseable JSON block (always emitted, required by the harness)
    print(json.dumps(ks.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _dry_run() -> None:
    """Print the hypothesis plan without touching the win-bank or running anything."""
    print("=" * 72)
    print("KEYSTONE  sp80_l1_gravity_flip  [DRY-RUN -- no execution]")
    print()
    print("HYPOTHESIS:")
    for k, v in HYPOTHESIS.items():
        print(f"  {k}: {v}")
    print()
    print("PLAN:")
    print(f"  1. Load win-bank: {WIN_BANK_PATH}")
    print(f"     expected sha256: {WIN_BANK_SHA256[:32]}...")
    print(f"  2. Extract L1 (level index {L1_INDEX}) activate_flow (ACTION id={ACTION5_ID}) pairs")
    print(f"     expected pairs: 7 (from 6 wins, 1-2 ACTION5 events each)")
    print(f"  3. Forward model: np.flipud(before_frame) on full board")
    print(f"     Alternative:    np.flipud(color-{LIQUID_COLOR} mask only)")
    print(f"  4. Metrics: exact_frame_matches N/7, mean_liquid_mask_iou")
    print()
    print("EXPECTED OUTPUT (based on win-bank analysis):")
    print("  exact_frame_matches: 0/7   (whole-board flipud never exactly matches)")
    print("  mean_liquid_mask_iou: ~0.171  (0.6 IoU on 2 winning episodes,")
    print("    0.0 on 5 episodes where liquid makes only small local moves)")
    print()
    print("INTERPRETATION:")
    print("  - Non-zero IoU on 2/7 pairs means the vertical-row-flip component")
    print("    is real on those episodes (row 36-39 -> 24-27 = flipud).")
    print("  - But the column range narrows (80 -> 48 pixels), meaning the liquid")
    print("    only fills the intersection with the relay block -- NOT a pure flipud.")
    print("  - The 5/7 pairs with IoU=0 have small local moves (33 px), suggesting")
    print("    different board states where the relay isn't positioned for the flip.")
    print("  - Conclusion: gravity-flip is PARTIALLY correct (row direction) but the")
    print("    forward model needs relay-block geometry to predict the column extent.")
    print()
    print(f"  admission_status: {HYPOTHESIS['admission_status']}")
    print(f"  boundary: {HYPOTHESIS['boundary']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print hypothesis plan; do not load win-bank or run anything",
    )
    p.add_argument(
        "--win-bank",
        type=Path,
        default=WIN_BANK_PATH,
        metavar="PATH",
        help=(
            f"path to sp80-589a99af_wins.pkl "
            f"(default: {WIN_BANK_PATH.relative_to(ROOT)})"
        ),
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        help="suppress human-readable header; emit only the JSON result block",
    )
    p.add_argument(
        "--liquid-color",
        type=int,
        default=LIQUID_COLOR,
        metavar="VAL",
        help=(
            f"raw pixel value to treat as 'liquid' for mask IoU "
            f"(default: {LIQUID_COLOR}; use 6 to test the magenta-as-liquid variant)"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Override module-level constant if requested
    global LIQUID_COLOR
    if args.liquid_color != LIQUID_COLOR:
        LIQUID_COLOR = args.liquid_color

    if args.dry_run:
        _dry_run()
        return 0

    # Run the keystone
    try:
        ks = run_keystone(args.win_bank)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "  The win-bank is required for this keystone.  "
            "Use --dry-run for a plan-only run.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR running keystone: {exc}", file=sys.stderr)
        raise

    _print_report(ks, json_only=args.json_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
