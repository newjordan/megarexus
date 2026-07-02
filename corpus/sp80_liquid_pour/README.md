# Liquid Pour - sp80-589a99af

Frame-level human narration corpus for SP80 / Liquid Pour.

## Game Identity

- **game_id**: `sp80-589a99af`
- **name**: Liquid Pour
- **corpus_id**: `sp80_liquid_pour`
- **source win-bank**: `human/win_bank_2026-05-16_human_cheat_map_v1/sp80-589a99af_wins.pkl`

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Frames | 1245 |
| Win-bank episodes | 6 |
| Sessions | night (`liquid_pour.m4a`) and morning (`liquid_pour_2.m4a`) |
| Transcript segments | 186 |
| Level-boundary anchors | 39 |
| Final states | GAME_OVER, GAME_OVER, GAME_OVER, GAME_OVER, GAME_OVER, WIN |

## Files

- `manifest.jsonl` - one row per win-bank frame.
- `frames/episode_*/frame_*.png` - canonical ARC palette renderings.
- `segments.json` - normalized transcript segments from the SRT/Whisper JSON inputs.
- `alignment.json` - machine-readable source hashes, anchor table, and limitations.
- `ALIGNMENT.md` - human-readable alignment assessment.
- `transcript_usefulness_review.json` - agent review labels for useful vs non-useful transcript segments, if generated after the builder.
- `NON_USEFUL_TRANSCRIPT_SEGMENTS.md` - readable list of non-useful transcript segments, if generated after the builder.
- `SHA256SUMS` - generated corpus artifact hashes.

## Label Boundary

Transcript text is copied only from audio-derived segment files. The corpus does
not contain agent-authored visual/mechanic ground-truth labels. Scalar support
fields (`alignment_support_scalar`, `narration_usefulness_scalar`,
`narration_window_scalar`) are risk controls for downstream filtering/weighting,
not ground truth. Any future visual review labels should use
`agent_review_label` until user-audited.
