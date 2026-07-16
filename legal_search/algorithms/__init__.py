"""Pure legal algorithms — no game-specific win knowledge.

Mined from codexlab/contra_head, sim3d, goose, loom — stripped of answers.
"""

from .affordances import mine_affordance_candidates
from .calibrate import mine_calibration
from .cem_plan import discrete_cem_plan
from .components import change_regions, connected_components, enrich_components, frame_changed
from .crucible_mine import mine_crucible
from .dynamics import mine_dynamics
from .explore import ChangeExplorer
from .geometry_rel import mine_geo_relations
from .local_symmetry import mine_symmetry
from .memory import EpisodicMemory
from .perception_trust import gate_frame_structure
from .ports import mine_ports_and_alignments, mine_ports_full
from .program_search import hill_climb_programs, random_program
from .role_induce import induce_movable_colors, induce_static_colors
from .saliency import edge_map, symmetry_score_horizontal
from .score import env_score_signals
from .shape_traces import mine_shape_traces
from .statecon_graph import mine_statecon
from .worldmodel import mine_worldmodel

__all__ = [
    "ChangeExplorer",
    "EpisodicMemory",
    "change_regions",
    "connected_components",
    "discrete_cem_plan",
    "edge_map",
    "enrich_components",
    "env_score_signals",
    "frame_changed",
    "gate_frame_structure",
    "hill_climb_programs",
    "induce_movable_colors",
    "induce_static_colors",
    "mine_affordance_candidates",
    "mine_calibration",
    "mine_crucible",
    "mine_dynamics",
    "mine_geo_relations",
    "mine_ports_and_alignments",
    "mine_ports_full",
    "mine_shape_traces",
    "mine_statecon",
    "mine_symmetry",
    "mine_worldmodel",
    "random_program",
    "symmetry_score_horizontal",
]
