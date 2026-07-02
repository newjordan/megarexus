"""SP80 player controller entrypoint."""

from __future__ import annotations

from sp80_base_controller import SP80BaseController


class SP80PlayerController(SP80BaseController):
    """Concrete controller for the SP80 player package."""

    player_id = "sp80_liquid_pour"

