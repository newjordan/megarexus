"""Primitive registry and calibration cases for SP80/Liquid Pour."""

from __future__ import annotations

from players.base import ControlCase, PrimitiveRegistry, PrimitiveSpec


CONTROL_PRIMITIVE_SPECS = [
    PrimitiveSpec(
        name="select_object",
        action_id=6,
        action_name="ACTION6",
        requires_click_target=True,
        effect_tags=("selection_changed",),
        description="Click a live selectable object and make it selected.",
    ),
    PrimitiveSpec(
        name="move_up",
        action_id=1,
        action_name="ACTION1",
        requires_selection=True,
        effect_tags=("selected_display_delta_0_-1",),
        description="Move the currently selected object one visible cell up when unblocked.",
    ),
    PrimitiveSpec(
        name="move_down",
        action_id=2,
        action_name="ACTION2",
        requires_selection=True,
        effect_tags=("selected_display_delta_0_1",),
        description="Move the currently selected object one visible cell down when unblocked.",
    ),
    PrimitiveSpec(
        name="move_left",
        action_id=3,
        action_name="ACTION3",
        requires_selection=True,
        effect_tags=("selected_display_delta_-1_0",),
        description="Move the currently selected object one visible cell left when unblocked.",
    ),
    PrimitiveSpec(
        name="move_right",
        action_id=4,
        action_name="ACTION4",
        requires_selection=True,
        effect_tags=("selected_display_delta_1_0",),
        description="Move the currently selected object one visible cell right when unblocked.",
    ),
    PrimitiveSpec(
        name="activate_flow",
        action_id=5,
        action_name="ACTION5",
        effect_tags=("flow_changed", "multi_frame_effect"),
        description="Activate the liquid/flow mechanic and record resulting frames.",
    ),
    PrimitiveSpec(
        name="no_selected_motion",
        action_id=1,
        action_name="ACTION1",
        requires_selection=True,
        effect_tags=("selected_display_delta_0_0", "blocked_motion"),
        description="Issue an upward move in a blocked setup and verify no selected motion.",
    ),
]

PRIMITIVE_ALIASES = {
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
    "click": "select_object",
    "select": "select_object",
    "activate": "activate_flow",
    "flow": "activate_flow",
    "no_motion": "no_selected_motion",
    "blocked": "no_selected_motion",
    "action1": "move_up",
    "action2": "move_down",
    "action3": "move_left",
    "action4": "move_right",
    "action5": "activate_flow",
    "action6": "select_object",
    "1": "move_up",
    "2": "move_down",
    "3": "move_left",
    "4": "move_right",
    "5": "activate_flow",
    "6": "select_object",
}

PRIMITIVE_REGISTRY = PrimitiveRegistry(CONTROL_PRIMITIVE_SPECS, PRIMITIVE_ALIASES)
CONTROL_PRIMITIVES = PRIMITIVE_REGISTRY.names()
SP80_ACTION_SET = PRIMITIVE_REGISTRY.action_set()

INTENT_TRAINING_EXAMPLES = {
    "select_object": [
        "i need to select the object",
        "select this object",
        "click the object",
        "choose the movable object",
    ],
    "move_up": [
        "i need to go up",
        "move up",
        "go upward",
        "push the selected object up",
    ],
    "move_down": [
        "i need to go down",
        "move down",
        "go downward",
        "push the selected object down",
    ],
    "move_left": [
        "i need to go left",
        "move left",
        "go leftward",
        "push the selected object left",
    ],
    "move_right": [
        "i need to go right",
        "move right",
        "go rightward",
        "push the selected object right",
    ],
    "activate_flow": [
        "i need to activate flow",
        "pour now",
        "test the liquid",
        "start the flow",
    ],
    "no_selected_motion": [
        "i need to test no motion",
        "try a blocked move",
        "check that the object cannot move",
        "confirm no selected object movement",
    ],
}

ACTION_SET_CASES = [
    ControlCase(
        case_id="select_object",
        intent="I need to select the object",
        expected_primitive="select_object",
        setup="fresh",
        effect_tags=("selection_changed",),
    ),
    ControlCase(
        case_id="move_up",
        intent="I need to go up",
        expected_primitive="move_up",
        setup="selected",
        effect_tags=("selected_display_delta_0_-1",),
    ),
    ControlCase(
        case_id="move_down",
        intent="I need to go down",
        expected_primitive="move_down",
        setup="selected",
        effect_tags=("selected_display_delta_0_1",),
    ),
    ControlCase(
        case_id="move_left",
        intent="I need to go left",
        expected_primitive="move_left",
        setup="selected",
        effect_tags=("selected_display_delta_-1_0",),
    ),
    ControlCase(
        case_id="move_right",
        intent="I need to go right",
        expected_primitive="move_right",
        setup="selected",
        effect_tags=("selected_display_delta_1_0",),
    ),
    ControlCase(
        case_id="activate_flow",
        intent="I need to activate flow",
        expected_primitive="activate_flow",
        setup="fresh",
        effect_tags=("flow_changed", "multi_frame_effect"),
    ),
    ControlCase(
        case_id="no_selected_motion",
        intent="I need to test no motion",
        expected_primitive="no_selected_motion",
        setup="blocked_up",
        effect_tags=("selected_display_delta_0_0", "blocked_motion"),
    ),
    ControlCase(
        case_id="approach_nearest_bucket",
        intent="I need to approach the nearest bucket",
        expected_primitive="goal_nearest_bucket",
        setup="selected",
        effect_tags=("goal_conditioned", "nearest_bucket_distance_reduced"),
    ),
]

SP80_EXPECTED_CASES = {
    "select_object": {
        "primitive": "select_object",
        "action_id": 6,
        "action_name": "ACTION6",
    },
    "move_up": {
        "primitive": "move_up",
        "action_id": 1,
        "action_name": "ACTION1",
        "selected_display_delta": [0, -1],
    },
    "move_down": {
        "primitive": "move_down",
        "action_id": 2,
        "action_name": "ACTION2",
        "selected_display_delta": [0, 1],
    },
    "move_left": {
        "primitive": "move_left",
        "action_id": 3,
        "action_name": "ACTION3",
        "selected_display_delta": [-1, 0],
    },
    "move_right": {
        "primitive": "move_right",
        "action_id": 4,
        "action_name": "ACTION4",
        "selected_display_delta": [1, 0],
    },
    "activate_flow": {
        "primitive": "activate_flow",
        "action_id": 5,
        "action_name": "ACTION5",
        "min_action_frames": 2,
    },
    "no_selected_motion": {
        "primitive": "no_selected_motion",
        "action_id": 1,
        "action_name": "ACTION1",
        "selected_delta": [0, 0],
        "selected_display_delta": [0, 0],
    },
    "approach_nearest_bucket": {
        "primitive": "goal_nearest_bucket",
        "dynamic_goal_primitive": True,
        "allowed_primitives": ["move_up", "move_down", "move_left", "move_right"],
        "requires_goal_condition": "pour_into_buckets",
        "requires_goal_progress": "distance_reduced",
    },
}


def action_set_cases_for_harness() -> list[dict[str, object]]:
    return [case.to_harness_case() for case in ACTION_SET_CASES]


def normalize_primitive(value: object) -> str | None:
    return PRIMITIVE_REGISTRY.normalize(value)
