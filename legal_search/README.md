# legal_search — operator backends (not the primary brain)

**Primary brain:** MegaMind — see [`docs/PRIMARY.md`](../docs/PRIMARY.md) and `from brain import MegaMind`.

This package provides **mined discrete algorithms** (geometry, symmetry, ports, crucible, features, …) that MegaMind operators import. Source is kept; run banks are not.

Do not treat `legal_search` train loops as the primary learning path. That is `train_flywheel.py` under the brain package.
