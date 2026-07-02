# Corpus Specification v1

The standard format for time-locked human playthrough corpora. Each game gets
one directory under `corpus/`. The s5i5 Snake Modulo corpus is the reference
implementation.

## Directory Structure

```
corpus/{game_short}_{game_name}/
├── manifest.jsonl       # One JSON object per frame, all sources aligned
├── frames/               # 0000.png – {N-1:04d}.png
├── segments.json         # Raw Whisper segments (from *_whisper.json)
├── alignment.json        # Anchor points and per-level stats (machine-readable)
├── ALIGNMENT.md          # Alignment assessment with evidence and confidence
├── README.md             # Human-readable game summary
```

## manifest.jsonl

One JSON object per line.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `frame` | int | Zero-based frame index |
| `ts_s` | float | Estimated wall-clock timestamp in seconds |
| `level` | int | Level number at this frame (from win-bank) |
| `action` | object | Action taken after this frame was observed |
| `action.id` | int | Action type (1-7, 6 for complex) |
| `action.x` | int | X coordinate (0 for non-complex) |
| `action.y` | int | Y coordinate (0 for non-complex) |
| `image` | string | Relative path to rendered frame PNG |
| `transcript` | string | Human narration text for this frame's time window |
| `seg_start_s` | float or null | Start of earliest overlapping whisper segment |
| `seg_end_s` | float or null | End of latest overlapping whisper segment |
| `alignment` | string | "anchor" or "interpolated" — how this frame's timestamp was derived |

## Transcript Alignment Rule

**Method: anchor-interpolated.** Proportional mapping (frame/total_frames) is
invalid — narration rate is not constant.

1. **Identify anchors.** Cross-reference the transcript against win-bank level
   boundaries. Find verbal markers for level transitions (e.g. "it gets harder",
   "third level", "game over"). Record confidence per anchor.
2. **Assign anchor timestamps.** The whisper segment containing the level
   transition marker gives the anchor frame's wall-clock time.
3. **Interpolate between anchors.** For frames between anchors, timestamps are
   linearly interpolated.
4. **Span transcript segments.** A whisper segment is assigned to ALL frames
   whose estimated time window overlaps with `[seg.start, seg.end]`.

Each game directory MUST contain an `ALIGNMENT.md` documenting:
- Each anchor point with whisper segment text, frame, timestamp, confidence
- Per-level pacing statistics
- Silent gap documentation
- Known limitations

## Frame Rendering

- Source: raw frame array from win-bank `.pkl` file (shape `[64, 64]`, dtype `int8`)
- Palette: `ARC_CANONICAL_RGB_PALETTE` from `architecture/src/lewm/data/arc_agi_3.py`
- Format: PNG, 64×64 pixels, RGB
- Naming: `{frame_index:04d}.png`, zero-padded

The canonical palette:

| Index | R   | G   | B   | Approximate name |
|-------|-----|-----|-----|------------------|
| 0     | 255 | 255 | 255 | White |
| 1     | 204 | 204 | 204 | Light grey |
| 2     | 153 | 153 | 153 | Medium grey |
| 3     | 102 | 102 | 102 | Dark grey |
| 4     |  51 |  51 |  51 | Very dark grey |
| 5     |   0 |   0 |   0 | Black |
| 6     | 229 |  58 | 163 | Pink |
| 7     | 255 | 123 | 204 | Light pink |
| 8     | 249 |  60 |  49 | Red |
| 9     |  30 | 147 | 255 | Blue |
| 10    | 136 | 216 | 241 | Light blue |
| 11    | 255 | 220 |   0 | Yellow |
| 12    | 255 | 133 |  27 | Orange |
| 13    | 146 |  18 |  49 | Maroon |
| 14    |  79 | 204 |  48 | Green |
| 15    | 163 |  86 | 214 | Purple |

Values outside [0, 15] are clamped.

## Adding a New Game

1. Identify the game from `game_names.json`
2. Locate the win-bank: `human/win_bank_.../{game_id}_wins.pkl`
3. Locate the transcript: `human/human_goal_discovery_.../transcript_{NNN}_{short}.txt`
4. Locate the whisper JSON: `human/raw_incoming_audio_and_sidecars/{name}_game.json`
5. Create `corpus/{short}_{name}/` following this spec
6. Render frames using the canonical palette
7. **Find anchor points** — cross-reference transcript against level boundaries
8. Interpolate timestamps between anchors, span transcript segments
9. Write `ALIGNMENT.md` with evidence and confidence per anchor
10. Write `README.md` with mechanical summary from transcript
11. Verify: `wc -l manifest.jsonl` matches frame count
