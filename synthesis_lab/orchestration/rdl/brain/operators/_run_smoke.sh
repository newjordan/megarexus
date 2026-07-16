#!/usr/bin/env bash
# Optional convenience wrapper for WP-A smoke (same as smoke_physics.py).
exec /home/frosty40/miniconda3/bin/python3 -B \
  "$(dirname "$0")/smoke_physics.py" "$@"
