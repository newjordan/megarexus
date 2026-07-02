"""Keystones — per-iteration standalone forward-model validators.

Each script in this folder must:
  - exit 0 with a parseable quantitative metric in stdout (N/N exact-match, mean IoU).
  - validate against win-bank or engine-source ground-truth frames (never synthetic-only).
  - declare  admission_status: discovery_only  -- a passing keystone is NOT solve-evidence.

A keystone is authored at STEP 4 (KEYSTONE-FIRST) in the CONTRA Loop Harness protocol.
See codexlab/contra_loop/CONTRA_LOOP.md for the full per-iteration contract.
"""
