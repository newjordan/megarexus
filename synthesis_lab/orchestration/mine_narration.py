"""SCOUT_EXPERIMENT: mine narration events from the sp80_liquid_pour corpus.

Turns the human's spoken narration (already transcribed, aligned, and
usefulness-reviewed in the corpus) into frame-aligned event spans that
future training elements can consume:

- outcome_spill / outcome_success / fuel — human-annotated outcome
  moments ("I spilled", "Got it", "ran out of...")
- intent — pre-action statements ("I'm going to move to the right")
- mechanic — general regime statements ("butting a piece stops flow")
- planning — routing/symmetry monologue

Label boundary (per corpus/ALIGNMENT.md): narration is human-generated
demonstration data — legitimate experience under the lab rules. The tag
rules below classify SPEECH, not game state; they are transparent,
auditable data curation (like the existing usefulness review), NOT a
game-mechanics model. Every emitted row carries the corpus's own risk
scalars (narration_window_scalar) so trainers can downweight fragile
windows, and the output stays below user-audited ground truth.

Usage:
    python synthesis_lab/orchestration/mine_narration.py \
        --corpus corpus/sp80_liquid_pour \
        --out synthesis_lab/experiments/megarexus_cnn/data/narration_events.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Ordered: first match wins as the PRIMARY tag; all matches land in tags.
TAG_RULES = [
    ("outcome_spill", re.compile(r"\bspill", re.I)),
    ("fuel", re.compile(r"run out|ran out|fuel|conservation|barely made", re.I)),
    ("outcome_success", re.compile(
        r"\bgot it\b|\bit worked\b|\bmade it\b|that's what happened"
        r"|that was a solution", re.I)),
    ("outcome_fail", re.compile(
        r"doesn't work|doesn't do anything|wasn't the case|technique was wrong"
        r"|couldn't fill|I lost", re.I)),
    ("intent", re.compile(
        r"I'm going to|I will\b|I envision|if I jet|I'm trying to"
        r"|I'm hoping|I have to make", re.I)),
    ("mechanic", re.compile(
        r"flow|isolated m\w+|faucet|branch|butt a piece|gravity|waterfall"
        r"|straight down|moving piece|stops? flow", re.I)),
    ("planning", re.compile(
        r"rout|symmetr|configur|lin(e|ing) up|calculat|assum|goal|extra piece"
        r"|pre-calc", re.I)),
]


def tag_text(text: str) -> tuple[str | None, list[str]]:
    tags = [name for name, rx in TAG_RULES if rx.search(text)]
    return (tags[0] if tags else None), tags


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    review = json.loads(
        (args.corpus / "transcript_usefulness_review.json").read_text())
    useful_ids = {r["segment_id"] for r in review["segments"]
                  if r["agent_review_label"] == "useful"}
    rows = [json.loads(l) for l in
            (args.corpus / "manifest.jsonl").read_text().splitlines()]

    # per-frame tagging: a frame gets a segment's tags iff the segment is
    # useful-reviewed and overlaps the frame's audio window
    frame_rows = []
    for r in rows:
        seg_ids = [s for s in (r.get("transcript_segment_ids") or [])
                   if s in useful_ids]
        text = (r.get("transcript") or "").strip()
        if not seg_ids or not text:
            continue
        primary, tags = tag_text(text)
        if primary is None:
            continue
        frame_rows.append({
            "episode_id": r["episode_id"],
            "episode_frame_index": r["episode_frame_index"],
            "level": r.get("level"),
            "image": r["image"],
            "primary": primary,
            "tags": tags,
            "text": text,
            "segment_ids": seg_ids,
            "narration_window_scalar": r.get("narration_window_scalar"),
            "alignment_confidence": r.get("alignment_confidence"),
        })

    # group contiguous same-primary frames into event spans
    spans = []
    for row in frame_rows:
        last = spans[-1] if spans else None
        if (last and last["episode_id"] == row["episode_id"]
                and last["primary"] == row["primary"]
                and row["episode_frame_index"] == last["frame_end"] + 1):
            last["frame_end"] = row["episode_frame_index"]
            last["n_frames"] += 1
            last["mean_scalar"] += row["narration_window_scalar"] or 0.0
            if row["text"] not in last["texts"]:
                last["texts"].append(row["text"])
        else:
            spans.append({
                "episode_id": row["episode_id"],
                "level": row["level"],
                "primary": row["primary"],
                "frame_start": row["episode_frame_index"],
                "frame_end": row["episode_frame_index"],
                "n_frames": 1,
                "mean_scalar": row["narration_window_scalar"] or 0.0,
                "texts": [row["text"]],
            })
    for s in spans:
        s["mean_scalar"] = round(s["mean_scalar"] / s["n_frames"], 4)

    out = {
        "schema": "synthesis_lab.narration_events.v0",
        "evidence": ("human narration segments (useful-reviewed) aligned to "
                     "win-bank frames; risk scalars from corpus ALIGNMENT.md; "
                     "SCOUT_EXPERIMENT data curation, below user-audited "
                     "ground truth"),
        "corpus": str(args.corpus),
        "tag_rules": {name: rx.pattern for name, rx in TAG_RULES},
        "n_frames_tagged": len(frame_rows),
        "frames": frame_rows,
        "spans": spans,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    from collections import Counter
    by_primary = Counter(r["primary"] for r in frame_rows)
    print(f"[narr] {len(frame_rows)} frames tagged, {len(spans)} spans "
          f"→ {args.out}")
    print(f"[narr] frames by primary tag: {dict(by_primary)}")
    print("[narr] event spans:")
    for s in spans:
        if s["primary"].startswith(("outcome", "fuel", "intent")):
            print(f"  {s['episode_id'][:28]:<28} lvl{s['level']} "
                  f"f{s['frame_start']:3d}-{s['frame_end']:3d} "
                  f"scalar={s['mean_scalar']:.2f} {s['primary']:<15} "
                  f"“{s['texts'][0][:60]}”")


if __name__ == "__main__":
    main()
