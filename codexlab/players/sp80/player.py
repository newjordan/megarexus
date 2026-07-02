"""Descriptor for the SP80/Liquid Pour player package."""

from __future__ import annotations

from players.base import GamePlayerDescriptor


SP80_PLAYER = GamePlayerDescriptor(
    game_id="sp80_liquid_pour",
    package="players.sp80",
    interface_schema="codexlab.game_player.v1",
    result_schema="codexlab.sp80_player_harness.v1",
    packet_schema="codexlab.sp80_harness_packet.v1",
    brain_request_schema="codexlab.sp80_brain_request.v1",
    controller_source="codexlab/players/sp80/controller.py",
    registry_source="codexlab/players/sp80/registry.py",
    verifier_source="codexlab/sp80_verify_run.py",
)

