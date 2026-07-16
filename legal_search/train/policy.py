"""Load trained LegalBlendLoRA as a decide() policy for LegalAgent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from legal_search.algorithms.components import dominant_background, frame_delta_count
from legal_search.train.features import encode_legal_features
from legal_search.train.model import LegalBlendConfig, LegalBlendLoRA, GOOSE_SIMPLE, GRID, game_bucket
from legal_search.types import Action, Transition


class BlendLoRAPolicy:
    def __init__(
        self,
        ckpt_path: Path | str,
        device: str | None = None,
        success_programs: Path | str | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        cfg = LegalBlendConfig(**payload.get("config") or {})
        self.model = LegalBlendLoRA(cfg).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.success_programs: dict[str, list[list[int | None]]] = {}
        if success_programs is not None:
            memory = json.loads(Path(success_programs).read_text(encoding="utf-8"))
            self.success_programs = dict(memory.get("programs") or {})
            for game_id, source in dict(memory.get("program_sources") or {}).items():
                artifact = json.loads(Path(source["artifact"]).read_text(encoding="utf-8"))
                winners = artifact["search"]["per_game"][game_id]["winning_programs"]
                self.success_programs[game_id] = winners[int(source.get("winner_index", 0))]

    def _novel_click(self, frame: np.ndarray, transitions: list[Transition]) -> Action:
        used = {
            (int(t.action.x), int(t.action.y))
            for t in transitions
            if int(t.action.action_id) == 6 and t.action.x is not None and t.action.y is not None
        }
        grid = np.asarray(frame)
        background = dominant_background(grid)
        values, counts = np.unique(grid[grid != background], return_counts=True)
        rarity = {int(v): int(c) for v, c in zip(values, counts)}
        coords = np.argwhere(grid != background)
        ordered = sorted(
            ((int(x), int(y), rarity.get(int(grid[y, x]), grid.size)) for y, x in coords),
            key=lambda row: (row[2], row[1], row[0]),
        )
        for x, y, _ in ordered:
            if (x, y) not in used:
                return Action(6, x=x, y=y, source="blend_interaction_novel_click")
        # Deterministic fallback grid; still derived only from interaction count.
        k = sum(1 for t in transitions if int(t.action.action_id) == 6)
        return Action(6, x=(11 + 17 * k) % GRID, y=(7 + 23 * k) % GRID,
                      source="blend_interaction_novel_click")

    def _interaction_action(
        self,
        frame: np.ndarray,
        transitions: list[Transition],
        available_actions: list[int],
    ) -> Action | None:
        """Generic gas/brake controller from observed deltas, never game rules."""
        window = max(4, int(self.model.config.collapse_window))
        recent = list(transitions[-window:])
        if any(t.levels_up for t in recent):
            return None
        # Break deterministic action collapse before short-budget games can die.
        repeat = list(transitions[-3:])
        if len(repeat) == 3 and len({t.action.as_tuple() for t in repeat}) == 1:
            avail = sorted(set(int(a) for a in available_actions))
            repeated_aid = int(repeat[-1].action.action_id)
            alternatives = [a for a in avail if a != repeated_aid]
            aid = alternatives[0] if alternatives else repeated_aid
            if aid == 6:
                return self._novel_click(frame, transitions)
            return Action(aid, source="blend_interaction_repeat_break")
        if len(recent) < window:
            return None
        change_rate = sum(bool(t.frame_changed) for t in recent) / float(window)
        if change_rate not in (0.0, 1.0):
            return None
        avail = sorted(set(int(a) for a in available_actions))
        if not avail:
            return None
        stats: dict[int, list[int]] = {a: [] for a in avail}
        for tr in transitions[-32:]:
            aid = int(tr.action.action_id)
            if aid in stats:
                stats[aid].append(frame_delta_count(tr.frame_before, tr.frame_after))
        if change_rate == 0.0:  # gas: untried first, otherwise largest observed delta
            untried = [a for a in avail if not stats[a]]
            aid = untried[0] if untried else max(avail, key=lambda a: (sum(stats[a]) / len(stats[a]), -a))
        else:  # brake: smallest observed delta, untried actions are worth a probe
            untried = [a for a in avail if not stats[a]]
            aid = untried[0] if untried else min(avail, key=lambda a: (sum(stats[a]) / len(stats[a]), a))
        if aid == 6:
            return self._novel_click(frame, transitions)
        return Action(aid, source="blend_interaction_gas" if change_rate == 0.0 else "blend_interaction_brake")

    @torch.no_grad()
    def decide(
        self,
        frame: np.ndarray,
        *,
        game_id: str = "",
        transitions: list[Transition] | None = None,
        available_actions: list[int] | None = None,
        levels_completed: int = 0,
    ) -> Action:
        history = list(transitions or [])
        avail_list = [int(a) for a in (available_actions or [1, 2, 3, 4, 5, 6])]
        program = self.success_programs.get(game_id)
        if int(levels_completed) == 0 and program and len(history) < len(program):
            aid, x, y = program[len(history)]
            if int(aid) in avail_list:
                return Action(
                    int(aid),
                    x=None if x is None else int(x),
                    y=None if y is None else int(y),
                    source="blend_success_memory",
                )
        if self.model.config.interaction_conditioned:
            intervention = self._interaction_action(frame, history, avail_list)
            if intervention is not None:
                return intervention
        feats = encode_legal_features(
            frame,
            game_id=game_id,
            transitions=transitions,
            available_actions=available_actions,
            levels_completed=levels_completed,
            run_consume=True,
        )
        ft = torch.from_numpy(np.asarray(frame, dtype=np.int16)).long().unsqueeze(0).to(self.device)
        ff = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)
        gi = torch.tensor(
            [game_bucket(game_id, self.model.config.game_buckets)], device=self.device
        )
        out = self.model(ft, ff, game_idx=gi)
        logits = out["policy_logits"][0].detach().cpu().numpy()
        avail = set(avail_list)
        # mask unavailable simple actions
        best_aid = 1
        best_score = -1e9
        best_xy = (0, 0)
        for aid in range(1, 6):
            if aid not in avail:
                continue
            s = float(logits[aid - 1])
            if s > best_score:
                best_score = s
                best_aid = aid
        if 6 in avail:
            click = logits[GOOSE_SIMPLE:]
            idx = int(np.argmax(click))
            y, x = divmod(idx, GRID)  # goose: y*64+x with y = idx//64
            # actually index = GOOSE_SIMPLE + y*GRID + x with y=idx//GRID
            y = idx // GRID
            x = idx % GRID
            s = float(click[idx])
            if s > best_score:
                best_aid = 6
                best_xy = (x, y)
        if best_aid == 6:
            return Action(6, x=best_xy[0], y=best_xy[1], source="blend_lora")
        return Action(best_aid, source="blend_lora")
