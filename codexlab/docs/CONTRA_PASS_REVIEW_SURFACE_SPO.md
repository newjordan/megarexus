# CONTRA Pass Review Surface SPO

This is the required review-surface procedure for CONTRA/paint evidence pages.

## Template

Use `codexlab/runs/contra_single_game_test_viewer/ar25_level_05/index.html` as the page template family:

- dark review surface
- top count cards
- large overlay screens before raw tables
- Oracle section when Oracle sidecars exist
- raw packet link
- computed/skipped output tables
- no claim that static hypotheses are control truth, goal truth, action choice, prediction, or solve evidence

## Required Pass Image Ledger

Every generated CONTRA review page must include a pass image ledger before raw data tables.

For each packet:

1. Count `layer_output_packets.outputs`.
2. Count `symmetry_hypotheses.symmetry_passes`.
3. Count every `state_condition_hypotheses[*].geoband_passes`.
4. Count Oracle sidecar rows only when actor/target rows have `source` beginning with `oracle_first_pass`.
5. Generate one readable PNG per counted pass.
6. Put the calculation result on the PNG itself, not only in HTML.
7. Render those PNGs full-width or near full-width in the page.
8. Verify the number of generated pass PNGs equals the expected count.

If a family has zero passes, show `0` explicitly. Do not imply missing Oracle sidecars exist.

## SP80 Level 00 Current Count

Current packet:

`codexlab/runs/contra_sp80_level_00_latest/result.json`

Expected pass images:

- CONTRA layer outputs: `13`
- Oracle sidecar pass images: `0`
- symmetry internal passes: `8`
- GeoBand internal passes: `4`
- generated PNG total: `25`

The page must show these counts in the top cards and in the pass-image ledger.

## Generator

Use:

```bash
python3 codexlab/contra_sp80_review_page.py \
  --packet codexlab/runs/contra_sp80_level_00_latest/result.json \
  --out-dir codexlab/runs/contra_sp80_level_00_latest
```

Generated pass images live under:

`codexlab/runs/contra_sp80_level_00_latest/assets/pass_images/`

Generated page:

`codexlab/runs/contra_sp80_level_00_latest/index.html`

## Verification

Before sending a link:

```bash
python3 -m py_compile codexlab/contra_sp80_review_page.py
python3 codexlab/contra_sp80_review_page.py --packet codexlab/runs/contra_sp80_level_00_latest/result.json --out-dir codexlab/runs/contra_sp80_level_00_latest
find codexlab/runs/contra_sp80_level_00_latest/assets/pass_images -type f -name '*.png' | wc -l
curl -I http://192.168.1.176:8766/codexlab/runs/contra_sp80_level_00_latest/index.html
npx playwright screenshot --viewport-size=1600,1000 http://127.0.0.1:8766/codexlab/runs/contra_sp80_level_00_latest/index.html /tmp/contra_sp80_pass_review.png
```

Required checks:

- page returns `200 OK`
- no missing image references
- pass image count matches packet-derived expected count
- screenshot shows readable large images
- the boundary sentence remains visible: static review only, not control truth, not goal truth, not action choice, not solve evidence
