#!/usr/bin/env bash
# Optional convenience wrapper for WP-A smoke (same as smoke_physics.py).
exec python3 -B   "$(dirname "$0")/smoke_physics.py" "$@"
