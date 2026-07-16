"""Short action-program search — algorithm mined from DW genome_search.

Legal form: hill-climb over action programs scored by a *plug-in* scorer.
No shipped win genomes. No level-id strategy tables.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from legal_search.types import Action, Proposal

MOVES = (1, 2, 3, 4)  # up/down/left/right convention in ARC agents


def random_program(
    rng: np.random.Generator,
    *,
    max_genes: int = 6,
    include_click: bool = True,
    include_activate: bool = True,
    grid: int = 64,
) -> list[Action]:
    k = int(rng.integers(1, max_genes + 1))
    prog: list[Action] = []
    for _ in range(k):
        choices = list(MOVES)
        if include_activate:
            choices.append(5)
        if include_click:
            choices.append(6)
        aid = int(rng.choice(choices))
        if aid == 6:
            prog.append(
                Action(6, x=int(rng.integers(0, grid)), y=int(rng.integers(0, grid)), source="program_search")
            )
        else:
            # gene magnitude: repeat move 1..8 times
            reps = int(rng.integers(1, 5))
            for _r in range(reps):
                prog.append(Action(aid, source="program_search"))
    return prog


def mutate_program(
    prog: list[Action],
    rng: np.random.Generator,
    *,
    grid: int = 64,
) -> list[Action]:
    out = list(prog)
    if not out:
        return random_program(rng, grid=grid)
    op = int(rng.integers(0, 4))
    if op == 0 and out:
        i = int(rng.integers(0, len(out)))
        a = out[i]
        new_id = int(rng.choice(MOVES))
        out[i] = Action(new_id, source="program_search")
    elif op == 1 and out:
        i = int(rng.integers(0, len(out)))
        a = out[i]
        if a.action_id == 6:
            out[i] = Action(
                6,
                x=int(rng.integers(0, grid)),
                y=int(rng.integers(0, grid)),
                source="program_search",
            )
        else:
            out.append(Action(a.action_id, source="program_search"))
    elif op == 2 and len(out) < 24:
        out.extend(random_program(rng, max_genes=2, grid=grid))
    elif len(out) > 1:
        del out[int(rng.integers(0, len(out)))]
    return out


def hill_climb_programs(
    scorer: Callable[[list[Action]], float],
    *,
    restarts: int = 8,
    steps: int = 16,
    seed: int = 0,
    grid: int = 64,
    topk: int = 3,
) -> list[Proposal]:
    """Mine of DW genome_search_candidates — scorer is injected (env or model)."""
    rng = np.random.default_rng(seed)
    found: list[Proposal] = []
    for r in range(restarts):
        cur = random_program(rng, grid=grid)
        best_score = float(scorer(cur))
        best = cur
        for _ in range(steps):
            cand = mutate_program(best, rng, grid=grid)
            s = float(scorer(cand))
            if s > best_score:
                best_score = s
                best = cand
        found.append(
            Proposal(
                actions=best,
                score=best_score,
                kind="program_hill_climb",
                evidence={"restart": r, "steps": steps},
                algorithm_id="search.program_hill_climb.v1",
            )
        )
    found.sort(key=lambda p: -p.score)
    # dedupe by action signature
    seen: set[tuple] = set()
    out: list[Proposal] = []
    for p in found:
        sig = tuple((a.action_id, a.x, a.y) for a in p.actions)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(p)
        if len(out) >= topk:
            break
    return out
