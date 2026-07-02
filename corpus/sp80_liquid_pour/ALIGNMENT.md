# Alignment Assessment - sp80-589a99af Liquid Pour

## Method

Manual anchor table plus linear interpolation between anchors. Pairing is
based on file timing, win-bank episode order, and transcript anchors.
Frame-level timestamps are estimates because there is no synced gameplay
video in the source bundle.

Confidence is per anchor and per interpolated row. Low-confidence spans
remain marked low, especially late retry sections where the transcript
contains repeated non-specific speech.

## Narration Window Scalars

Each manifest row carries numeric support fields so downstream consumers
can downweight fragile windows instead of accepting every transcript
overlap equally:

- `alignment_confidence_scalar`: HIGH=1.0, MEDIUM=0.65, LOW=0.25.
- `interpolation_distance_scalar`: 1.0 at anchors and decays toward
  0.65 at the midpoint between adjacent anchors.
- `narration_usefulness_scalar`: duration-weighted fraction of overlapping
  transcript segments marked useful by the agent review table.
- `narration_window_scalar`: product of the three supports above.

These scalars are risk controls, not new evidence. The usefulness table is
agent-reviewed and must remain below user-audited ground truth.

## Source Pairing

### night_liquid_pour_2026-05-21

- **audio**: `human/raw_incoming_audio_and_sidecars/liquid_pour.m4a`
- **transcript**: `human/human_goal_discovery_calculus_2026-05-19/transcript_027_sp80.srt`
- **episodes**: [0, 1, 2]
- **pairing_status**: admitted_with_limitations
- **basis**: Night audio file timing and transcript_027 SP80 content align with win-bank episodes 0-2; early episode has strong mechanical anchors, later retry sections are low confidence.

### morning_liquid_pour_2_2026-05-21

- **audio**: `human/raw_incoming_audio_and_sidecars/liquid_pour_2.m4a`
- **transcript**: `human/raw_incoming_audio_and_sidecars/liquid_pour_2.json`
- **episodes**: [3, 4, 5]
- **pairing_status**: admitted_with_limitations
- **basis**: Morning audio file timing, liquid_pour_2 Whisper segments, and explicit morning transcript text align with win-bank episodes 3-5; late repeated-okay retry tail is low confidence.

## Anchor Points

### Episode 0 (night_liquid_pour_2026-05-21)

- **win_bank_episode_field**: 1
- **played_utc**: 2026-05-21T07:28:05Z
- **frames**: 230
- **final_state**: GAME_OVER

| Local frame | Level | Timestamp | Segment | Confidence | Quote |
|-------------|-------|-----------|---------|------------|-------|
| 0 | 0 | 0.000s | night_srt_001 | HIGH | Okay we're doing liquid pour. |
| 8 | 1 | 68.840s | night_srt_005 | HIGH | Yep, that's what happened. |
| 50 | 2 | 225.080s | night_srt_011 | HIGH | So this has three spouts |
| 64 | 3 | 397.720s | night_srt_019 | MEDIUM | One has a gap |
| 129 | 4 | 625.720s | night_srt_030 | MEDIUM | The challenge is routing. |
| 229 | 4 | 795.720s | night_srt_036 | HIGH | Yeah, I lost |

### Episode 1 (night_liquid_pour_2026-05-21)

- **win_bank_episode_field**: 2
- **played_utc**: 2026-05-21T07:33:57Z
- **frames**: 190
- **final_state**: GAME_OVER

| Local frame | Level | Timestamp | Segment | Confidence | Quote |
|-------------|-------|-----------|---------|------------|-------|
| 0 | 0 | 815.720s | night_srt_037 | MEDIUM | Let's do it again. |
| 5 | 1 | 835.720s | night_srt_038 | LOW | Let's do it again. |
| 27 | 2 | 855.720s | night_srt_039 | LOW | Let's do it again. |
| 43 | 3 | 875.720s | night_srt_040 | MEDIUM | I'm rushing through it. |
| 95 | 4 | 1009.720s | night_srt_046 | LOW | No, it doesn't work like that. |
| 189 | 4 | 1199.720s | night_srt_055 | LOW | Oh. |

Limitations:
- episode 1 level 1 boundary: No level-specific verbal description in this section.
- episode 1 level 2 boundary: No level-specific verbal description in this section.
- episode 1 game over: No explicit game-over phrase at this boundary.

### Episode 2 (night_liquid_pour_2026-05-21)

- **win_bank_episode_field**: 3
- **played_utc**: 2026-05-21T07:40:16Z
- **frames**: 174
- **final_state**: GAME_OVER

| Local frame | Level | Timestamp | Segment | Confidence | Quote |
|-------------|-------|-----------|---------|------------|-------|
| 0 | 0 | 1227.720s | night_srt_056 | MEDIUM | Let's do it again. |
| 4 | 1 | 1251.720s | night_srt_057 | LOW | Let's see. |
| 26 | 2 | 1281.720s | night_srt_058 | LOW | Let's do it again. |
| 43 | 3 | 1321.720s | night_srt_060 | LOW | Let's do it again. |
| 82 | 4 | 1345.720s | night_srt_061 | LOW | Let's do it again. |
| 173 | 4 | 1393.332s | night_srt_062 | MEDIUM | I'm tired. |

Limitations:
- episode 2 level 1 boundary: No level-specific verbal description in this section.
- episode 2 level 2 boundary: No level-specific verbal description in this section.
- episode 2 level 3 boundary: No level-specific verbal description in this section.
- episode 2 level 4 boundary: No level-specific verbal description in this section.
- episode 2 game over: SRT tail extends slightly beyond ffprobe duration.

### Episode 3 (morning_liquid_pour_2_2026-05-21)

- **win_bank_episode_field**: 1
- **played_utc**: 2026-05-21T14:52:56Z
- **frames**: 301
- **final_state**: GAME_OVER

| Local frame | Level | Timestamp | Segment | Confidence | Quote |
|-------------|-------|-----------|---------|------------|-------|
| 0 | 0 | 0.000s | morning_json_000 | HIGH | Okay, a new game. This is going to be Liquid Pour. |
| 4 | 1 | 16.000s | morning_json_003 | MEDIUM | So, it's easy to match. |
| 32 | 2 | 46.000s | morning_json_004 | MEDIUM | routing based upon flow |
| 44 | 3 | 386.000s | morning_json_025 | MEDIUM | the isolated movement technique |
| 88 | 4 | 473.000s | morning_json_037 | HIGH | There's a piece that's a faucet. |
| 180 | 5 | 898.000s | morning_json_068 | HIGH | This is the last level. |
| 300 | 5 | 1089.000s | morning_json_078 | HIGH | I gotta do it again. |

### Episode 4 (morning_liquid_pour_2_2026-05-21)

- **win_bank_episode_field**: 2
- **played_utc**: 2026-05-21T15:08:37Z
- **frames**: 198
- **final_state**: GAME_OVER

| Local frame | Level | Timestamp | Segment | Confidence | Quote |
|-------------|-------|-----------|---------|------------|-------|
| 0 | 0 | 1111.000s | morning_json_080 | MEDIUM | I had to make an L with a vertical piece. |
| 6 | 1 | 1145.000s | morning_json_082 | LOW | Okay. |
| 30 | 2 | 1230.000s | morning_json_085 | LOW | Okay. |
| 43 | 3 | 1318.000s | morning_json_088 | LOW | Okay. |
| 85 | 4 | 1403.000s | morning_json_091 | LOW | That is life for me right now, defined. |
| 111 | 5 | 1516.000s | morning_json_095 | LOW | Okay. |
| 197 | 5 | 1801.000s | morning_json_106 | LOW | Okay. |

Limitations:
- episode 4 level 1 boundary: No level-specific verbal description in this section.
- episode 4 level 2 boundary: No level-specific verbal description in this section.
- episode 4 level 3 boundary: No level-specific verbal description in this section.
- episode 4 level 5 boundary: No level-specific verbal description in this section.
- episode 4 game over: No explicit game-over phrase at this boundary.

### Episode 5 (morning_liquid_pour_2_2026-05-21)

- **win_bank_episode_field**: 3
- **played_utc**: 2026-05-21T15:22:57Z
- **frames**: 152
- **final_state**: WIN

| Local frame | Level | Timestamp | Segment | Confidence | Quote |
|-------------|-------|-----------|---------|------------|-------|
| 0 | 0 | 1801.000s | morning_json_106 | LOW | Okay. |
| 4 | 1 | 1874.000s | morning_json_110 | LOW | Okay. |
| 24 | 2 | 1956.000s | morning_json_113 | LOW | Okay. |
| 36 | 3 | 2014.000s | morning_json_115 | LOW | Okay. |
| 72 | 4 | 2101.000s | morning_json_118 | LOW | Okay. |
| 105 | 5 | 2176.000s | morning_json_121 | LOW | Okay. |
| 151 | 6 | 2250.000s | morning_json_123 | LOW | Okay. |

Limitations:
- episode 5 level 0 start: No explicit restart phrase at this boundary.
- episode 5 level 1 boundary: No level-specific verbal description in this section.
- episode 5 level 2 boundary: No level-specific verbal description in this section.
- episode 5 level 3 boundary: No level-specific verbal description in this section.
- episode 5 level 4 boundary: No level-specific verbal description in this section.
- episode 5 level 5 boundary: No level-specific verbal description in this section.
- episode 5 win: Win is from win-bank state, not verbal confirmation.

## Global Limitations

- No synchronized gameplay video was available; frame times are manual-anchor estimates.
- Late night retry sections mostly say 'Let's do it again' and are intentionally low confidence.
- Late morning retry sections mostly say 'Okay' and are intentionally low confidence.
- Win/fail states come from the win-bank pickle, not from transcript speech unless an anchor says so.
- Agent-authored visual/mechanic labels are not ground truth and are left null.

## Validation

- Source checksum gate: `sha256sum --quiet -c SHA256SUMS` passed before generation.
- Builder command: `python scripts/build_sp80_liquid_pour_corpus.py`.
- Reviewer command: `python scripts/review_sp80_liquid_pour_corpus.py`.
