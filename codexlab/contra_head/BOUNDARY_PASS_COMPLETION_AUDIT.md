# CONTRA Boundary-Pass Completion Audit

Audit date: 2026-05-27

This audit maps the active CONTRA boundary-pass objective to current evidence.
It is intentionally fail-closed: a requirement is not complete unless a file,
manifest, verifier, or controlled run proves it.

## Status Summary

| Requirement | Status | Evidence |
| --- | --- | --- |
| Boundary-pass spec | Complete | `codexlab/contra/BOUNDARY_PASS_SPEC.md` |
| First-class GeoBand passes | Complete | `codexlab/contra/boundary_pass.py`; verified in `sp80_verify_color_board_surfaces.py` |
| Boundary queue | Complete | `codexlab/contra/boundary_pass.py`; rendered and verified in SP80/cross-game manifests |
| Viewer layer split | Complete | `sp80_color_board_viewer.py` renders raw board, primary GeoBand field, subgroups, boundary queue, actor/target/supports, and StateCon layers |
| No dropped objects verifier | Complete | `sp80_verify_color_board_surfaces.py` enforces primary support coverage, subgroup refinement, unresolved boundary reasons, and Y-object metadata |
| Manual SP80 calibration table | Exported, not human-reviewed | `codexlab/runs/contra_v001_sp80_manual_calibration_table/manifest.json` has `review_status: not_reviewed` and `admission_status: not_admitted` |
| DeltaBand controlled event run | Complete as controlled observation | `codexlab/runs/contra_v001_sp80_deltaband_controlled_run/manifest.json` has `status: controlled` and 6 verified levels |

## Verified Commands

```bash
python3 -m py_compile codexlab/contra/*.py codexlab/sp80_color_board_viewer.py codexlab/cross_game_color_board_viewer.py codexlab/sp80_verify_color_board_surfaces.py codexlab/sp80_manual_calibration_table.py codexlab/sp80_verify_manual_calibration_table.py codexlab/sp80_deltaband_controlled_run.py codexlab/sp80_verify_deltaband_run.py
git diff --check
python3 codexlab/sp80_verify_color_board_surfaces.py codexlab/runs/contra_v001_sp80_boundary_pass_review
python3 codexlab/sp80_verify_color_board_surfaces.py codexlab/runs/contra_v001_cross_game_boundary_pass_review
python3 codexlab/sp80_verify_manual_calibration_table.py codexlab/runs/contra_v001_sp80_manual_calibration_table
python3 codexlab/sp80_verify_deltaband_run.py codexlab/runs/contra_v001_sp80_deltaband_controlled_run
```

## Static Review Artifacts

- SP80 boundary-pass viewer:
  `codexlab/runs/contra_v001_sp80_boundary_pass_review/index.html`
- Cross-game boundary-pass viewer:
  `codexlab/runs/contra_v001_cross_game_boundary_pass_review/index.html`
- Manual calibration review table:
  `codexlab/runs/contra_v001_sp80_manual_calibration_table/index.html`
- Human signoff scaffold:
  `codexlab/runs/contra_v001_sp80_manual_calibration_table/human_signoff_template.json`
- DeltaBand controlled run:
  `codexlab/runs/contra_v001_sp80_deltaband_controlled_run/index.html`

## Human-Review Boundary

The manual calibration table is not complete as reviewed calibration. It is a
human-review table with seed candidates only. The remaining required human
review fields are:

- `actor_object_ids`
- `target_object_ids`
- `known_action_support_object_ids`
- `connector_object_ids`
- `known_y_object_ids`
- `unresolved_object_ids`
- `image_hash`

Until a human-reviewed artifact fills those fields and a verifier accepts it,
the calibration table remains `not_reviewed`, `not_admitted`, and cannot be
used as semantic truth, control truth, goal truth, action choice, or solve
truth.

The reviewed acceptance gate is:

```bash
python3 codexlab/sp80_apply_manual_calibration_signoff.py --signoff codexlab/runs/contra_v001_sp80_manual_calibration_table/human_signoff_template.json --out-dir codexlab/runs/contra_v001_sp80_manual_calibration_reviewed
python3 codexlab/sp80_verify_manual_calibration_table.py codexlab/runs/contra_v001_sp80_manual_calibration_reviewed --require-reviewed
```

That gate is expected to fail until the calibration rows carry
`review_status: human_reviewed`, reviewer metadata, reviewed expectation lists,
and an `image_hash` matching each row's `source_raw_frame_sha256`.

Current expected failure reasons include pending manifest status, pending row
status, empty reviewer metadata, non-final reviewed image hashes, and non-empty
`fields_pending_human_review`.

## DeltaBand Controlled Evidence

The DeltaBand run used the real SP80 `activate_flow` primitive (`ACTION5`) and
recorded before frames, action frame sequences, after frames, changed boundary
ids, fixed boundary ids, and null-action event lists for all six SP80 slices.

Current verified summary:

| Level | Action frames | Changed pixels | Changed boundaries | Fixed boundaries | Null events |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 22 | 2 | 2 | 0 | 0 |
| 1 | 22 | 1 | 0 | 21 | 0 |
| 2 | 24 | 1 | 0 | 81 | 0 |
| 3 | 33 | 1 | 57 | 0 | 0 |
| 4 | 30 | 1 | 0 | 70 | 0 |
| 5 | 31 | 1 | 51 | 0 | 0 |

DeltaBand remains observation metadata only. It is not predictive and is not
admitted as control truth.
