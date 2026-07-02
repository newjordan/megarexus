#!/usr/bin/env python3
"""Keystone: test the Crucible promoted forward model on SP80 L1 move transitions.

HYPOTHESIS (sp80_l1_surface_redirect_chain)
  action_primitive:         move_up/move_down/move_left/move_right (ACTION1-4)
  precondition_signature:   actor block present in the L1 board; a move is issued
  predicted_effect:         CrucibleModel.predict() (grammar.v0 promoted rules,
                            composed over the v0 forward baseline) predicts the
                            ACTOR's unit displacement for the issued move; on L1
                            the surface redirects/blocks some moves
  falsification_condition:  predicted_canon_delta != observed actor unit delta on
                            the L1 move transitions in the win-bank ground truth

This keystone loads ALL SP80 L1 (level index 1) move transitions (ACTION1-4)
from the human win-bank, derives the actor block's unit displacement, applies
CrucibleModel.predict(), and reports:

  * exact_actor_delta_matches N/M  -- predicted_canon_delta == observed unit delta
                                      AND the model gave an opinion ("prediction")

CARDINAL RULE HONORED:
  admission_status: discovery_only  -- even N/N here is NOT a solve.
  A solve requires levels_completed_delta>0 from a verified episode run.

  Near-zero matches is a VALID go/no-go outcome, not a failure to hide. The plan
  pre-authorizes it: if matches are near zero, stop live wiring and iterate
  offline grammar v1 with connected-component / largest-object features.

REPRESENTATION CAVEATS (reported, not hidden):
  * The win-bank L1 actor is the rigidly-translating block (data-detected,
    empirically color 9) at a 4-pixel cell pitch. The Crucible corpus keys local
    context on the harness `selected_coord`, which in the mined rows sits on the
    color-12 background. predict() was induced from that corpus, so its promoted
    rules condition on background-cell colors, not the moving block.
  * The corpus selected-coord axis convention (move_left=(-1,0), move_up=(0,-1))
    may be transposed relative to the win-bank render axes. A diagnostic count
    under an axis swap is reported separately; it is NOT the headline.
  These gaps are exactly what a near-zero headline should motivate fixing.

GROUND-TRUTH SOURCE: win_bank (primary), located exactly as sp80_l1_gravity_flip.
  Path: human/win_bank_2026-05-16_human_cheat_map_v1/sp80-589a99af_wins.pkl
  SHA-256: e8fa00464434edff9fdbf369318d09cb5e6c43b76e47298455137ae0d4cce848

Usage:
    python codexlab/contra_loop/keystones/sp80_l1_surface_redirect_chain.py --dry-run
    python codexlab/contra_loop/keystones/sp80_l1_surface_redirect_chain.py
    python codexlab/contra_loop/keystones/sp80_l1_surface_redirect_chain.py --json-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # rexus/
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

from contra_head.crucible.model import CrucibleModel, DEFAULT_LEDGER_PATH  # noqa: E402

WIN_BANK_PATH = ROOT / "human" / "win_bank_2026-05-16_human_cheat_map_v1" / "sp80-589a99af_wins.pkl"
WIN_BANK_SHA256 = "e8fa00464434edff9fdbf369318d09cb5e6c43b76e47298455137ae0d4cce848"

# Level index (0-based) in the win-bank 'levels' list.
L1_INDEX = 1

# SP80 action ids -> control primitives (authoritative sp80_base_controller map:
# up=ACTION1, down=ACTION2, left=ACTION3, right=ACTION4). A swapped left/right
# convention appears in sp80_play_known_solutions; recorded here as a caveat.
ID_TO_PRIMITIVE = {1: "move_up", 2: "move_down", 3: "move_left", 4: "move_right"}

BOARD_H = 64
BOARD_W = 64

HYPOTHESIS: dict[str, Any] = {
    "name": "sp80_l1_surface_redirect_chain",
    "source_signal": (
        "Crucible grammar.v0 promoted forward model (blocked/forward/slide rules, "
        "composed over the v0 forward baseline) induced from the admitted SP80 "
        "transition corpus; tested for whether it predicts the L1 actor's unit "
        "displacement, including surface redirects/blocks, on human win-bank moves"
    ),
    "implementation_target": "contra_head/crucible/model.py CrucibleModel.predict",
    "boundary": "Hypothesis material only; not local CONTRA evidence.",
    "action_primitive": "move_up|move_down|move_left|move_right",
    "precondition_signature": "actor block present; a move (ACTION1-4) is issued",
    "predicted_effect": "CrucibleModel.predict() predicted_canon_delta == actor unit delta",
    "falsification_condition": (
        "predicted_canon_delta != observed actor unit delta on >= 1 L1 move transition"
    ),
    "mechanic_class": "surface_redirect_chain",
    "admission_status": "discovery_only",
}


@dataclass
class MovePair:
    win_index: int
    l1_step: int
    primitive: str
    before: np.ndarray
    after: np.ndarray
    actor_anchor: tuple[int, int]            # (row, col) of actor block top-left in before
    observed_unit_delta: tuple[int, int]     # (sign drow, sign dcol)


@dataclass
class PairResult:
    win_index: int
    l1_step: int
    primitive: str
    observed_unit_delta: tuple[int, int]
    predicted_canon_delta: tuple[int, int] | None
    opinion: str
    matched_rule_id: str | None
    exact_match: bool
    axis_swap_match: bool


@dataclass
class KeystoneResult:
    hypothesis_name: str
    ground_truth_source: str
    win_bank_sha256: str
    mechanic_ledger_loaded: bool
    mechanic_ledger_sha256: str | None
    actor_color: int | None
    n_pairs_tested: int
    exact_actor_delta_matches: int
    pair_results: list[PairResult] = field(default_factory=list)

    @property
    def exact_match_fraction(self) -> str:
        return f"{self.exact_actor_delta_matches}/{self.n_pairs_tested}"

    @property
    def n_opinions(self) -> int:
        return sum(1 for r in self.pair_results if r.opinion == "prediction")

    @property
    def n_no_opinion(self) -> int:
        return sum(1 for r in self.pair_results if r.opinion != "prediction")

    @property
    def n_axis_swap_matches(self) -> int:
        return sum(1 for r in self.pair_results if r.axis_swap_match)

    @property
    def near_zero(self) -> bool:
        if self.n_pairs_tested == 0:
            return True
        return (self.exact_actor_delta_matches / self.n_pairs_tested) <= 0.05

    def to_dict(self) -> dict[str, Any]:
        observed_hist = Counter(str(tuple(r.observed_unit_delta)) for r in self.pair_results)
        predicted_hist = Counter(
            str(tuple(r.predicted_canon_delta)) if r.predicted_canon_delta is not None else "None"
            for r in self.pair_results
        )
        return {
            "schema": "codexlab.contra_loop.keystone_result.v1",
            "keystone": self.hypothesis_name,
            "mechanic_class": HYPOTHESIS["mechanic_class"],
            "ground_truth_source": self.ground_truth_source,
            "win_bank_sha256": self.win_bank_sha256,
            "mechanic_ledger_loaded": self.mechanic_ledger_loaded,
            "mechanic_ledger_sha256": self.mechanic_ledger_sha256,
            "actor_color_detected": self.actor_color,
            "n_pairs_tested": self.n_pairs_tested,
            "exact_actor_delta_matches": self.exact_match_fraction,
            "exact_actor_delta_match_count": self.exact_actor_delta_matches,
            "diagnostics": {
                "n_model_opinions": self.n_opinions,
                "n_no_opinion": self.n_no_opinion,
                "n_axis_swap_matches": self.n_axis_swap_matches,
                "observed_unit_delta_histogram": dict(sorted(observed_hist.items())),
                "predicted_canon_delta_histogram": dict(sorted(predicted_hist.items())),
                "axis_swap_is_diagnostic_only": True,
            },
            "near_zero_match_rate": self.near_zero,
            "go_no_go": (
                "no_go_iterate_grammar_v1_connected_component_largest_object_features"
                if self.near_zero
                else "candidate_signal_present_review_before_live_wiring"
            ),
            "admission_status": "discovery_only",
            "boundary": HYPOTHESIS["boundary"],
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_win_bank(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"win-bank not found: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"win-bank expected list, got {type(data)}")
    return data


def _rigid_translation(before_mask: np.ndarray, after_mask: np.ndarray) -> tuple[int, int] | None:
    """Return the (drow, dcol) shift if after is a pure translation of before."""
    bcount = int(before_mask.sum())
    acount = int(after_mask.sum())
    if bcount == 0 or bcount != acount:
        return None
    br, bc = np.where(before_mask)
    ar, ac = np.where(after_mask)
    drow = int(ar.min() - br.min())
    dcol = int(ac.min() - bc.min())
    shifted = np.zeros_like(before_mask)
    rr = br + drow
    cc = bc + dcol
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= before_mask.shape[0] or cc.max() >= before_mask.shape[1]:
        return None
    shifted[rr, cc] = True
    if not np.array_equal(shifted, after_mask):
        return None
    return (drow, dcol)


def detect_actor_color(win_bank: list[dict[str, Any]]) -> int | None:
    """Data-driven L1 actor color = the non-background color that rigidly
    translates in the most L1 move transitions."""
    translate_counts: Counter[int] = Counter()
    for entry in win_bank:
        frames = entry.get("frames", [])
        actions = entry.get("actions", [])
        levels = entry.get("levels", [])
        for i in range(min(len(actions), len(levels), len(frames) - 1)):
            if levels[i] != L1_INDEX or actions[i].get("id") not in ID_TO_PRIMITIVE:
                continue
            before = np.asarray(frames[i], dtype=np.int16)
            after = np.asarray(frames[i + 1], dtype=np.int16)
            background = int(np.bincount(before.flatten().clip(min=0)).argmax())
            for color in sorted(set(before.flatten().tolist())):
                if color in (0, background):
                    continue
                shift = _rigid_translation(before == color, after == color)
                if shift is not None and shift != (0, 0):
                    translate_counts[int(color)] += 1
    if not translate_counts:
        return None
    return translate_counts.most_common(1)[0][0]


def extract_l1_move_pairs(win_bank: list[dict[str, Any]], actor_color: int) -> list[MovePair]:
    pairs: list[MovePair] = []
    for win_idx, entry in enumerate(win_bank):
        frames = entry.get("frames", [])
        actions = entry.get("actions", [])
        levels = entry.get("levels", [])
        for i in range(min(len(actions), len(levels), len(frames) - 1)):
            if levels[i] != L1_INDEX:
                continue
            primitive = ID_TO_PRIMITIVE.get(actions[i].get("id"))
            if primitive is None:
                continue
            before = np.asarray(frames[i], dtype=np.int16)
            after = np.asarray(frames[i + 1], dtype=np.int16)
            before_mask = before == actor_color
            after_mask = after == actor_color
            if int(before_mask.sum()) == 0 or int(after_mask.sum()) == 0:
                continue
            shift = _rigid_translation(before_mask, after_mask)
            if shift is None:
                # Not a clean actor translation/stationary state (paint/flow event).
                continue
            br, bc = np.where(before_mask)
            anchor = (int(br.min()), int(bc.min()))
            observed = (int(np.sign(shift[0])), int(np.sign(shift[1])))
            pairs.append(
                MovePair(
                    win_index=win_idx,
                    l1_step=i,
                    primitive=primitive,
                    before=before,
                    after=after,
                    actor_anchor=anchor,
                    observed_unit_delta=observed,
                )
            )
    return pairs


def evaluate_pair(model: CrucibleModel, pair: MovePair) -> PairResult:
    # actor_coord follows the corpus convention coord=[x, y] indexing grid[x, y];
    # x is the first numpy axis (row). rotation_quarters=0 (win-bank display frame).
    actor_coord = [pair.actor_anchor[0], pair.actor_anchor[1]]
    prediction = model.predict(pair.before, pair.primitive, actor_coord, rotation_quarters=0)
    predicted = prediction.predicted_canon_delta
    observed = pair.observed_unit_delta
    exact = prediction.opinion == "prediction" and predicted is not None and tuple(predicted) == observed
    # Diagnostic only: does the prediction match the observed delta with axes swapped?
    axis_swap = (
        prediction.opinion == "prediction"
        and predicted is not None
        and (tuple(predicted)[1], tuple(predicted)[0]) == observed
    )
    return PairResult(
        win_index=pair.win_index,
        l1_step=pair.l1_step,
        primitive=pair.primitive,
        observed_unit_delta=observed,
        predicted_canon_delta=None if predicted is None else (int(predicted[0]), int(predicted[1])),
        opinion=prediction.opinion,
        matched_rule_id=prediction.matched_rule_id,
        exact_match=exact,
        axis_swap_match=axis_swap,
    )


def run_keystone(win_bank_path: Path, ledger_path: Path) -> KeystoneResult:
    actual_sha256 = _sha256_file(win_bank_path)
    win_bank = load_win_bank(win_bank_path)

    model = CrucibleModel.load(ledger_path)
    actor_color = detect_actor_color(win_bank)
    if actor_color is None:
        raise ValueError("could not detect an L1 actor color (no rigidly-translating block found)")
    pairs = extract_l1_move_pairs(win_bank, actor_color)
    if not pairs:
        raise ValueError("no clean L1 move transitions found in the win-bank")

    results: list[PairResult] = []
    if model is not None:
        results = [evaluate_pair(model, p) for p in pairs]
    else:
        # No mechanic ledger: every move is a no-opinion (fail-open) result.
        for p in pairs:
            results.append(
                PairResult(
                    win_index=p.win_index,
                    l1_step=p.l1_step,
                    primitive=p.primitive,
                    observed_unit_delta=p.observed_unit_delta,
                    predicted_canon_delta=None,
                    opinion="no_opinion",
                    matched_rule_id=None,
                    exact_match=False,
                    axis_swap_match=False,
                )
            )

    exact = sum(1 for r in results if r.exact_match)
    return KeystoneResult(
        hypothesis_name=HYPOTHESIS["name"],
        ground_truth_source="win_bank",
        win_bank_sha256=actual_sha256,
        mechanic_ledger_loaded=model is not None,
        mechanic_ledger_sha256=getattr(model, "ledger_sha256", None) if model is not None else None,
        actor_color=actor_color,
        n_pairs_tested=len(results),
        exact_actor_delta_matches=exact,
        pair_results=results,
    )


def _print_report(ks: KeystoneResult, *, json_only: bool) -> None:
    if not json_only:
        print("=" * 72)
        print(f"KEYSTONE  {ks.hypothesis_name}")
        print(f"  mechanic_class  : {HYPOTHESIS['mechanic_class']}")
        print(f"  forward_model   : CrucibleModel.predict() over the promoted mechanic ledger")
        print(f"  ledger_loaded   : {ks.mechanic_ledger_loaded}  sha={ks.mechanic_ledger_sha256}")
        print(f"  actor_color     : {ks.actor_color}  (data-detected rigidly-translating L1 block)")
        print(f"  ground_truth    : {ks.ground_truth_source}  (sha256 {ks.win_bank_sha256[:16]}...)")
        print("-" * 72)
        print(f"  pairs_tested    : {ks.n_pairs_tested}")
        for r in ks.pair_results[:24]:
            print(
                f"    win={r.win_index} step={r.l1_step:3d} {r.primitive:<10s}: "
                f"obs={str(tuple(r.observed_unit_delta)):<8s} "
                f"pred={str(tuple(r.predicted_canon_delta)) if r.predicted_canon_delta is not None else 'None':<8s} "
                f"opinion={r.opinion:<10s} exact={r.exact_match}"
            )
        if len(ks.pair_results) > 24:
            print(f"    ... ({len(ks.pair_results) - 24} more)")
        print("-" * 72)
        print(f"  exact_actor_delta_matches : {ks.exact_match_fraction}")
        print(f"  model_opinions / no_opinion: {ks.n_opinions} / {ks.n_no_opinion}")
        print(f"  axis_swap_matches (diag)  : {ks.n_axis_swap_matches}")
        print()
        if ks.near_zero:
            print("  [NEAR-ZERO] exact match rate <= 5%.")
            print("  This is a VALID go/no-go outcome (not hidden): the corpus keys local")
            print("  context on the harness selected_coord (color-12 background), while the")
            print("  L1 actor is the color-%s block at a 4px cell pitch. Per the plan, the" % ks.actor_color)
            print("  next step is offline grammar v1 with connected-component / largest-object")
            print("  features BEFORE any live wiring trusts Crucible L1 predictions.")
        else:
            print("  Candidate signal present; review against the representation caveats")
            print("  before any live wiring.")
        print()
        print(f"  admission_status: {HYPOTHESIS['admission_status']}  (NOT a solve)")
        print(f"  boundary: {HYPOTHESIS['boundary']}")
        print("=" * 72)
    print(json.dumps(ks.to_dict(), indent=2, sort_keys=True))


def _dry_run() -> None:
    print("=" * 72)
    print("KEYSTONE  sp80_l1_surface_redirect_chain  [DRY-RUN -- no execution]")
    print()
    print("HYPOTHESIS:")
    for k, v in HYPOTHESIS.items():
        print(f"  {k}: {v}")
    print()
    print("PLAN:")
    print(f"  1. Load win-bank: {WIN_BANK_PATH}")
    print(f"     expected sha256: {WIN_BANK_SHA256[:32]}...")
    print(f"  2. Load mechanic ledger: {DEFAULT_LEDGER_PATH.relative_to(ROOT)} (fail-open if absent)")
    print(f"  3. Detect L1 actor color (rigidly-translating block; empirically color 9)")
    print(f"  4. Extract L1 (level {L1_INDEX}) move (ACTION1-4) transitions with a clean")
    print(f"     actor translation; observed unit delta = sign of the block shift")
    print(f"  5. Apply CrucibleModel.predict() and count exact_actor_delta_matches N/M")
    print()
    print("EXPECTED OUTPUT (representation-gap analysis):")
    print("  exact_actor_delta_matches: near 0/N is the expected outcome -- the corpus")
    print("  keys context on the selected_coord (color-12 background), not the moving")
    print("  block, so predict() mostly returns no_opinion on win-bank L1 moves.")
    print("  This maps to the plan's NO-GO path: iterate grammar v1 with")
    print("  connected-component / largest-object features before live wiring.")
    print()
    print(f"  admission_status: {HYPOTHESIS['admission_status']}")
    print(f"  boundary: {HYPOTHESIS['boundary']}")
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the hypothesis plan; do not run")
    p.add_argument("--win-bank", type=Path, default=WIN_BANK_PATH, metavar="PATH")
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH, metavar="PATH")
    p.add_argument("--json-only", action="store_true", help="emit only the JSON result block")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        _dry_run()
        return 0
    try:
        ks = run_keystone(args.win_bank, args.ledger)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("  The win-bank is required. Use --dry-run for a plan-only run.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR running keystone: {exc}", file=sys.stderr)
        raise
    _print_report(ks, json_only=args.json_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
