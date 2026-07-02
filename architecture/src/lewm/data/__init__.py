"""Dataset adapters for LeWM.

Each module wraps an external benchmark or simulator and exposes a
``torch.utils.data.Dataset`` that yields the batch fields LeWM consumes
(``obs`` of shape ``(T, C, H, W)`` plus benchmark-specific per-step action
fields, all assembled by a DataLoader into ``(B, T, ...)`` tensors that
match paper Algorithm 1's pseudocode). Adapters are kept independent so
we can develop one without disturbing another.
"""
