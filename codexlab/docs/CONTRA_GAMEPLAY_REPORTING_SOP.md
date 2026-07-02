# CONTRA Gameplay Reporting SOP

Date: 2026-05-31
Scope: Angel reporting structure for CONTRA, Oracle sidecars, SP80 gameplay harness runs, and future game harnesses.
Status: SOP and implementation contract. This document is not run evidence.

## Purpose

Create a reporting harness that records gameplay runs as an inspectable visual process, not just as JSON logs. Every important function or stage must produce a clean image, every action must preserve before/action/after state, and every review page must make evidence boundaries obvious.

The reporting target is:

- one traceable run directory per condition
- one readable `index.html` review surface per run
- one image slide per meaningful function, pass, action, or decision point
- one manifest that links every slide to raw inputs, raw outputs, hashes, source files, and commands
- no claim of control truth, goal truth, action choice, prediction, or solve evidence unless the required controlled artifact exists

## CONTRA Shorthand

When the user says `CONTRA`, read it as the full integrated CONTRA system, not a
single layer, single packet field, or partial report. A static CONTRA run must
call `contra.run_contra_view(...)` and preserve all required pass reports:
`config_pass`, `oracle_pass`, `contra_pass`, and `view_system_passes`.

If the user asks for a CONTRA playthrough, level run, gameplay run, solve
attempt, or action run, static view packets are not enough. The report must come
from a controlled harness that records actions, before/after frames,
`packets.jsonl`, `brain_calls.jsonl`, verifier output, and a review surface. If
the full requested mode cannot be produced, report `cannot-run-full-CONTRA` or
`cannot-report-yet` and name the missing artifact instead of substituting a
partial static packet.

## Current Code Anchors

Use these existing repo surfaces as the baseline:

| Area | Current path | Reporting role |
| --- | --- | --- |
| Gameplay harness | `codexlab/sp80_player_harness.py` | Runs SP80 control cases, records `result.json`, `packets.jsonl`, `brain_calls.jsonl`, frame PNGs, raw NPY grids, DeltaBand, ActionModBand, and episode memory. |
| Packet recorder | `codexlab/players/base.py` | Standard JSONL/result writer for gameplay packets and brain calls. |
| Static CONTRA packet API | `codexlab/contra/api.py` | Stable entrypoint for `run_contra_view(...)` over a raw frame. |
| CONTRA review page | `codexlab/contra_sp80_review_page.py` | Builds `index.html` and one large PNG per layer/pass from a packet. This is the current visual-slide pattern. |
| Review helpers | `codexlab/review_helpers.py` | Shared repo-relative link and HTML escaping helpers. |
| Oracle sidecar | `codexlab/oracle.py` | Emits review-only assumptions. Oracle must remain sidecar-only and must not control action choice. |
| Package boundary | `codexlab/docs/PACKAGE_GUIDE.md` | `codexlab/` is canonical source; `codexlab/runs/` is artifacts only. |
| Pass image procedure | `codexlab/docs/CONTRA_PASS_REVIEW_SURFACE_SPO.md` | Requires a pass image ledger, visible pass counts, and frames-first review pages. |

## Evidence Boundary Rules

Every report must label itself as one of these categories:

| Category | Meaning | May claim |
| --- | --- | --- |
| `static_review` | A packet or frame was inspected without executing a controlled action. | Visual hypotheses, layer counts, admission status, missing inputs. |
| `controlled_run` | A real game harness action sequence was executed and recorded. | The recorded action happened, with before/after frames and action logs. |
| `proxy` | A substitute condition was intentionally used. | Only the proxy result, never the requested condition. |
| `historical` | Existing artifact reviewed without regeneration. | What the historical artifact says, with stale boundary. |
| `invalid` | Missing provenance, missing raw inputs, or failed verification. | Nothing beyond the invalid reason. |

Hard rules:

- Do not show raw JSON as the only reporting surface.
- Do not call an image "evidence" unless it is linked to a recorded condition, command, and source input.
- Do not blur static CONTRA hypotheses with controlled gameplay action evidence.
- Do not let Oracle outputs become control truth, goal truth, action choice, prediction, or solve evidence.
- Do not omit zero counts. A missing Oracle sidecar count must be shown as `0`.
- Do not bury the first meaningful frame below tables.
- Do not use tiny thumbnails for state frames or pass images.
- Gameplay reports are image-first artifacts. The Angel report must open on a
  meaningful image section, normally PortBand for SP80, and each recorded action
  must have image frames for brain decision, action/click overlay, raw gameplay
  before/after frames, delta, action sequence, DeltaBand, ActionModBand, and
  episode memory. Each recorded action packet must also expose Config, CONTRA
  pass, View System passes, raw color components, control isolation, scene
  surfaces, symmetry, actor/target/support affordances, StateCon, GeoBand
  boundary queues, ExploreBand, action semantics, layer handoffs,
  debug/admission, Oracle, Ghost, relationship graph, relationship ladder, and
  PortBand images before dense tables are acceptable.
- A gameplay report is not reportable unless `angel_report_verify.py` passes
  with `--strict-png-ledger --require-gameplay-visual-contract`.

## Standard Run Directory

New reporting harness outputs should use this structure:

```text
codexlab/runs/<run_id>/
  index.html
  report_manifest.json
  report_trace.jsonl
  result.json
  packets.jsonl
  brain_calls.jsonl
  slides/
    000_run_condition.png
    010_level_00_raw_frame.png
    020_level_00_contra_raw_color_components.png
    030_level_00_contra_control_isolation.png
    ...
    300_level_00_action_000_before.png
    301_level_00_action_000_decision.png
    302_level_00_action_000_after.png
    303_level_00_action_000_delta.png
    304_level_00_action_000_sequence_contact_sheet.png
    900_provenance.png
  assets/
    pass_images/
    action_images/
    timelines/
    thumbnails/
  levels/
    level_00/
      frames/
      raw/
      overlays/
      action_000/
  oracle/
    *.json
  screenshots/
    index_desktop.png
    index_mobile.png
```

`codexlab/runs/` remains artifact-only. Reusable reporting code belongs in a source file such as `codexlab/gameplay_report_harness.py`, not inside a run directory.

## Required Trace IDs

Every packet row, slide row, action row, and manifest entry must carry stable IDs:

| Field | Required for | Format |
| --- | --- | --- |
| `run_id` | all artifacts | `<system>_<game>_<condition>_<utc>` |
| `level_run_id` | per-level rows | `<run_id>_level_<nn>` |
| `frame_id` | every frame | `<level_run_id>_frame_<phase>_<index>` |
| `function_call_id` | every instrumented function/stage | `<level_run_id>_fn_<ordinal>_<function_name>` |
| `layer_pass_id` | CONTRA internal layer/pass slides | source packet pass id when present |
| `action_invocation_id` | every gameplay action | existing harness id such as `<case_id>_<grounded_action_name>` |
| `brain_call_id` | brain calls | `<action_invocation_id>_brain` |
| `oracle_sidecar_id` | Oracle sidecars | source Oracle packet id or file stem |
| `packet_sha256` | packet-backed slides | SHA-256 of the packet or raw JSON row basis |
| `image_sha256` | every generated image | SHA-256 of the rendered PNG |

## Required Manifest Fields

`report_manifest.json` is the top-level contract for a run:

```json
{
  "schema": "angel.contra_gameplay_report_manifest.v1",
  "created_at_utc": "2026-05-31T00:00:00Z",
  "report_status": "static_review",
  "run_id": "contra_sp80_example_20260531T000000Z",
  "game_id": "sp80-589a99af",
  "working_directory": "/home/frosty40/rexus",
  "git_commit": "<commit>",
  "git_dirty": true,
  "command": "<exact command>",
  "source_paths": [
    {"path": "codexlab/sp80_player_harness.py", "sha256": "<hash>"}
  ],
  "raw_inputs": [
    {"path": "codexlab/runs/.../frame.npy", "sha256": "<hash>"}
  ],
  "raw_outputs": [
    {"path": "codexlab/runs/.../result.json", "sha256": "<hash>"}
  ],
  "slide_count": 0,
  "function_slide_count": 0,
  "action_count": 0,
  "oracle_sidecar_count": 0,
  "verification": {
    "image_count_matches_manifest": false,
    "index_reachable": false,
    "browser_screenshot_checked": false
  },
  "boundaries": [
    "Not solve evidence unless a controlled solve verifier is named.",
    "Oracle sidecars are review-only."
  ]
}
```

## Required Trace Row

`report_trace.jsonl` is the chronological process record. Each row is one function, pass, action, verifier, or rendered slide:

```json
{
  "schema": "angel.report_trace_event.v1",
  "run_id": "<run_id>",
  "event_index": 12,
  "event_type": "function_slide",
  "level_index": 0,
  "function_call_id": "contra_sp80_fn_004_symmetry_hypotheses",
  "function_name": "symmetry_hypotheses",
  "function_family": "contra_layer",
  "input_refs": [{"path": "codexlab/runs/.../result.json", "sha256": "<hash>"}],
  "output_refs": [{"path": "codexlab/runs/.../slides/040_symmetry.png", "sha256": "<hash>"}],
  "source_packet_refs": ["layer_output_packets.outputs[3]"],
  "admission_status": "not_admitted",
  "truth_boundary": "static_review_not_action_choice",
  "status": "rendered"
}
```

## Required Visual Slide Types

Every slide must be a standalone PNG that remains understandable outside the HTML page. Use the current `contra_sp80_review_page.py` pattern: large raw/overlay image on the left, calculation/result panel on the right, title and boundary text on the image itself.

| Slide type | Required content |
| --- | --- |
| Run condition | Run ID, condition, command, source commit/hash, evidence category, boundaries. |
| Raw frame | Raw frame rendered large, frame hash, level, source path. |
| Function slide | Function/stage name, input count, output count, missing inputs, admission status, reason codes, overlay boxes/edges when available. |
| Config pass | First-click/config status, selected click, allowance counts, and explicit not-control/goal/prediction/solve flags. |
| CONTRA pass | Unified pass summary, required input refs, layer output counts, relationship ladder status, and review-only boundary. |
| View System passes | Required report passes, pass status, pass count, and no-truth-admission flags. |
| Raw color components | Raw frame with component boxes, color counts, component counts, and source frame hash. |
| Control isolation | Raw frame with control-isolation component boxes and packet status fields. |
| Scene surfaces | Raw frame with surface candidate boxes, role counts, status/admission, and not-truth boundary. |
| Symmetry hypotheses | Raw frame with symmetry group/component boxes, scores/counts, and static-hypothesis boundary. |
| Actor/target/support affordances | Raw frame with candidate actor, goal/target, and support boxes, ranks/scores/counts, and not-action-authority boundary. |
| StateCon | Relationship condition rows, member boxes when present, features/counts, and no prediction/solve admission. |
| GeoBand boundary queues | Boundary queue rows, unresolved reasons, candidate boxes, and relationship-review boundary. |
| ExploreBand/action semantics | Budget or semantic ledger status, scores/counts, and control-feedback boundary. |
| Layer handoffs/debug/admission | Layer handoff status, recovered/skipped/fail-closed counts, admission flags, and missing-input reasons. |
| Oracle sidecar | Selected missing element, assumption text, candidate boxes, explicit not-admitted truth flags. |
| Relationship graph | Before frame with recorded StateCon/GeoBand nodes and edges, plus counts and not-action-authority boundary. |
| Relationship ladder | CONTRA relationship admission ladder, complete/incomplete stages, missing inputs, and not-control/goal/prediction/solve boundary. |
| Brain decision | Allowed primitives, request summary, chosen primitive, confidence/error, latency, expected primitive when controlled. |
| Action overlay | Before frame with selected source bbox and click marker when click coordinates are recorded. |
| Gameplay frame | Individual recorded before, after, and returned action frames exposed as first-class Angel frames, not only as a contact sheet. |
| Action before | Before frame, selected object/state, goal condition, action intent. |
| Action after | After frame, selected object/state, returned frame count, level-completion delta. |
| Action delta | Before/after or overlay delta, changed pixel count, selected coordinate delta, null-action status. |
| Action sequence | Contact sheet of returned frames when the game returns multiple frames. |
| DeltaBand | Pre-action ladder, post-action ladder, raw frame hashes, descriptor changes, null-action signal. |
| ActionModBand | Action response summary, downstream result, level-completion delta, repeat/no-op guard if present. |
| Episode memory | Before memory id, after-round memory id, action taken, added/updated observations. |
| Admission ladder | Stage name, present inputs, missing inputs, completeness, admission status. |
| Provenance | Command, cwd, host, source hashes, raw input hashes, output hashes, verifier results. |

For gameplay reports, these action-level image types are mandatory for every
recorded action: `config_pass`, `contra_pass`, `view_system_passes`,
`raw_color_components`, `control_isolation`, `scene_surfaces`,
`symmetry_hypotheses`, `actor_affordances`, `target_affordances`,
`support_action_affordances`, `state_condition_hypotheses`,
`geoband_boundary_queues`, `exploreband_budget`, `action_semantics_ledger`,
`layer_handoffs`, `debug_summary`, `admission`, `oracle_pass`, `ghost_pass`,
`portband_alignment`, `relationship_graph`, `relationship_ladder`,
`brain_decision`, `action_overlay`, `gameplay_frame`, `action_before`,
`action_after`, `action_delta`, `action_sequence`, `deltaband_gate`,
`actionmodband`, and `episode_memory`. Level-level summary images remain
mandatory for `oracle_pass`, either `ghost_pass` or `ghost_memory`,
`relationship_graph`, `relationship_ladder`, and `portband_alignment`.
`gameplay_frame` must include at least the recorded `before` and `after` phases
for every action, plus all returned frame-sequence PNGs when present. The Angel
`contraview_report.json` must expose these as image frames with reachable
`image_url` assets under `/data/<run_id>/assets/`.
The manifest action timeline must record `returned_frame_count` and
`expected_gameplay_frame_count`, and the verifier must reject mismatches between
those counts and the actual `gameplay_frame` before/after/returned-frame rows.
The manifest action timeline must also record `contra_layer_visual_map` from
the packet's `layer_output_packets.outputs`; every non-hidden layer output must
map to a visual stage present for that same action. Hidden layer outputs must be
explicitly named in `hidden_layer_output_ids` with a hidden reason.
When an action packet records click coordinates, the corresponding
`action_overlay` frame must carry machine-checkable `click_xy` metadata and
`has_click_marker: true`; the verifier must reject mismatches between the action
timeline and rendered overlay frame.
The local `angel_assets/` directory must be an exact asset ledger for the
current `contraview_report.json`: every referenced image must exist, and
unreferenced stale image files are invalid.

## Per-Function Image Policy

"One clean image per function" means one image per recorded system stage, not one image for every Python helper. A function is reportable when it changes the packet, makes an admission decision, calls an external brain/model, executes a game action, or creates a sidecar assumption.

Required reportable function families:

| Family | Examples | Image count rule |
| --- | --- | --- |
| CONTRA layers | raw colors, control isolation, scene surfaces, symmetry, actor, target, support, StateCon, GeoBand, boundary queue, DeltaBand gate, debug summary, admission | one PNG per `layer_output_packets.outputs` row |
| Internal passes | symmetry passes, GeoBand passes, PortBand alignments, PortBandFleet candidates | one PNG per pass row, with zero counts shown when absent |
| Oracle | direct sidecar packet, first-pass actor/target rows | one PNG per sidecar or Oracle row |
| Brain | tiny MLP, LoRA adapter, Qwen HTTP | one PNG per brain call |
| Game action | grounded primitive, action execution, returned frame sequence | before, after, delta, and sequence contact sheet |
| Memory/control feedback | DeltaBand, ActionModBand, episode memory | one PNG per packet or action result |
| Verifier | syntax, packet verifier, image count check, URL/browser check | one provenance slide plus manifest rows |

## HTML Review Page Layout

Every `index.html` must follow this order:

1. Header: condition, run id, evidence category, and boundary sentence.
2. Top cards: status, slide counts, action counts, pass counts, Oracle sidecar count, verifier status.
3. Source links: raw packet, manifest, trace JSONL, prior review pages, raw frame paths.
4. Visual deck: raw frame, before/action/after/delta images, then function slides.
5. Pass image ledger: family counts and all generated pass PNGs full-width or near full-width.
6. Gameplay action timeline: one row per action with links to before, decision, after, delta, and sequence slides.
7. Object and relation tables: actors, targets, supports, StateCons, GeoBands, boundary queues.
8. Brain/model table: backend, request schema, decision, error, latency, allowed primitives.
9. Oracle sidecar section: assumptions, selected missing elements, truth flags, admission status.
10. Provenance and verification: command, cwd, host, source hashes, input/output hashes, browser checks.

Frames and visual decks come before dense tables.

## Verification Gate

A report is not ready to send unless these checks are recorded in the manifest:

```bash
PYTHONPATH=codexlab python -m py_compile codexlab/contra_sp80_review_page.py
PYTHONPATH=codexlab python codexlab/contra_sp80_review_page.py --packet <packet> --out-dir <run-dir>
find <run-dir>/assets/pass_images -type f -name '*.png' | wc -l
curl -I http://127.0.0.1:<port>/<run-dir>/index.html
npx playwright screenshot --viewport-size=1600,1000 http://127.0.0.1:<port>/<run-dir>/index.html <run-dir>/screenshots/index_desktop.png
```

For a gameplay run, also verify:

```bash
PYTHONPATH=codexlab python codexlab/sp80_verify_run.py <run-dir>
```

The exact verifier command may differ by run family. The manifest must record the actual command used, exit status, and output path.

## Implementation Plan

The reporting harness should be added in small steps:

1. Add a source generator such as `codexlab/gameplay_report_harness.py`.
2. Make it ingest an existing `sp80_player_harness.py` run directory.
3. Emit `report_manifest.json` and `report_trace.jsonl` without changing the original run artifacts.
4. Render action slides from recorded `packets.jsonl` frame references.
5. Reuse the existing `pass_card(...)` visual pattern for function slides.
6. Build `index.html` with the required layout.
7. Add a verifier that checks manifest rows, image counts, source paths, hashes, and missing image references.
8. Only after static ingestion is stable, wire the generator into live gameplay harness execution.

## Minimum Acceptance Criteria

The first acceptable implementation must produce:

- `index.html`
- `report_manifest.json`
- `report_trace.jsonl`
- at least one raw frame slide
- one function slide per existing CONTRA `layer_output_packets.outputs` row
- one before/after/delta slide set per executed action packet
- one brain decision slide per `brain_calls.jsonl` row
- one provenance slide
- a manifest check that fails if any slide file is missing or has no hash
- a review page screenshot proving the page is readable

## Naming Rules

Use stable, sortable names:

```text
slides/000_run_condition.png
slides/010_level_00_raw_frame.png
slides/020_level_00_contra_<layer_id>.png
slides/100_level_00_oracle_<sidecar_id>.png
slides/200_level_00_brain_<action_index>.png
slides/300_level_00_action_<action_index>_before.png
slides/301_level_00_action_<action_index>_after.png
slides/302_level_00_action_<action_index>_delta.png
slides/303_level_00_action_<action_index>_sequence.png
slides/900_provenance.png
```

Do not overwrite an existing run directory unless the command explicitly says `--overwrite`. Prefer timestamped run IDs.

## Report Language

Use these terms consistently:

- `reported`: artifact exists and is linked.
- `controlled`: exact gameplay condition was executed and recorded.
- `static_review`: visual/static packet only.
- `not_admitted`: present for review but not control truth.
- `missing_input`: stage could not run because required input was absent.
- `recovered_error`: layer failed but produced a fail-closed packet.
- `invalid`: report cannot be used as evidence because provenance or verification failed.

Avoid these terms unless a verifier and artifact support them:

- `solved`
- `passed`
- `working`
- `ready`
- `proves`
- `evidence` without category and provenance

## Immediate Next Step

Start by implementing an ingest-only report generator against an existing run, such as:

```bash
PYTHONPATH=codexlab python codexlab/gameplay_report_harness.py \
  --run-dir codexlab/runs/sp80_player_harness_tiny_mlp_levels/level_00 \
  --contra-packet codexlab/runs/contra_sp80_level_00_latest/result.json \
  --out-dir codexlab/runs/sp80_player_harness_tiny_mlp_levels/level_00/report
```

This first version must not rerun gameplay. It should convert existing recorded packets and frames into a visual slide/report surface, then verify its own manifest.
