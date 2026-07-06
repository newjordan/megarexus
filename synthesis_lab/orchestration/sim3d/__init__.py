"""sim3d — 3D pour-simulator imagination space for SP80 (2026-07-05, user's design).

Extract a 3D scene from the 2D board (scene.py), then literally pour spheres out of
the bucket (engine.py, Lane P) or trace the hydraulic streams (raycast.py, Lane R);
render the rollout back to frames and grade it with the proven per-target scorer
(metrics.py). decisive_test.py is the go/no-go; calibrate.py fits params on legal
corpora only. Plan-of-record: synthesis_lab/notes/SIM3D_EXECUTION_PLAN_20260705.md
"""
