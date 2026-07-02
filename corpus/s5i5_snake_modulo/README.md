# Snake Modulo — s5i5-18d95033

Human playthrough corpus. 1 episode, 3 levels cleared.

## Game Identity

- **game_id**: `s5i5-18d95033`
- **name**: Snake Modulo (aliases: snake, snake modulo game)
- **tags**: click-only, monodirectional body parts, modular motion

## Mechanical Summary (from human narration)

Snake Modulo is a click-only puzzle in which the player controls modular
body-part segments ("snakes"). Each segment is *monodirectional* — it extends
or shrinks along exactly one axis. The player discovers which button controls
which body part, then uses the parts to navigate obstacles and reach targets.

Key mechanics discovered:

1. **Modular body control.** Body parts extend/shrink in one direction.
   "Every modular part is monodirectional."

2. **Obstacles as movable objects.** Some obstacles are also controllable
   bodies. "Find out what you control. Find out what you don't."

3. **Self-blocking.** Body parts can block each other. Body length is linked.

4. **Mirrored targets.** Moving one body part can affect another target.

5. **Exploration cost.** Clicking costs fuel. "I clicked too many times.
   Game over."

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Frames | 215 |
| Actions | 215 (all ACTION6 complex) |
| Levels | 4 (0–3), won 3 |
| FPS | 15 |
| Transcript duration | 520.1s |
| Whisper segments | 52 |
| Level boundaries | frames 17, 51, 114 |
| Final state | GAME_OVER |

## Per-Level Pacing (from ALIGNMENT.md)

| Level | Frames | Duration | s/frame |
|-------|--------|----------|---------|
| 0     | 17     | 111.7s   | 6.6     |
| 1     | 34     | 65.4s    | 1.9     |
| 2     | 63     | 154.4s   | 2.5     |
| 3     | 100    | 171.4s   | 1.7     |

## Files

- `manifest.jsonl` — 215 records, anchor-interpolated timestamps
- `frames/0000.png` – `frames/0214.png` — canonical ARC-AGI-3 palette
- `segments.json` — raw Whisper segments with timestamps
- `alignment.json` — anchor points and per-level stats (machine-readable)
- `ALIGNMENT.md` — alignment assessment with evidence and confidence
