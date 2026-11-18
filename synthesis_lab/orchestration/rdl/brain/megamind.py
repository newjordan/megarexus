"""MegaMind facade: observe → judge_plan → act_logits → train_transition.

Composes InternalHemisphereFull, ExternalHemisphereFull, CentralCortex, and
OscillationTrainer. I dreams/proposes; E judges; C commits actions.
"""

from __future__ import annotations

from typing import Any

import torch

from .ada_online import CentralAdaOnline
from .central import CentralConfig, CentralCortex
from .hemisphere_e import ExternalHemisphereFull
from .hemisphere_i import EXTRA_FEAT_DIM, InternalHemisphereFull, pack_operator_feats
from .hemisphere_i import _as_frame_tensor
from .hemispheres import ExternalHemisphere, HemisphereBundle, InternalHemisphere
from .oscillate import OscillationConfig, OscillationTrainer
from .brain_types import BrainState, TruthFlags


class MegaMind:
    """Bi-hemispherical brain facade (v0).

    Methods:
      observe(frame) -> BrainState          # map + I propose, no act
      judge_plan(frame, action, plan) -> …  # E judges I dream
      act_logits(state) -> logits           # central only
      train_transition(s, a, s')            # oscillate one cycle
    """

    def __init__(
        self,
        emb: int = 128,
        n_actions: int = 8,
        device: str | torch.device = "cpu",
        lr_E: float = 1e-3,
        lr_I: float = 1e-3,
        lr_C: float = 2e-3,
    ) -> None:
        self.emb = emb
        self.n_actions = n_actions
        self.device = torch.device(device)

        # Shared nets so OscillationTrainer and full hemispheres stay aligned
        self._net_I = InternalHemisphere(emb=emb, n_actions=n_actions)
        self._net_E = ExternalHemisphere(emb=emb, n_actions=n_actions)
        self.I = InternalHemisphereFull(
            net=self._net_I, emb=emb, n_actions=n_actions, device=self.device
        )
        self.E = ExternalHemisphereFull(
            net=self._net_E,
            emb=emb,
            n_actions=n_actions,
            device=self.device,
            lr=lr_E,
            use_world_model=True,
        )
        self.bundle = HemisphereBundle(internal=self._net_I, external=self._net_E)
        # keep central pad == hemisphere operator garden size (no silent truncation)
        self.central = CentralCortex(
            CentralConfig(emb=emb, n_actions=n_actions, extra_feats=int(EXTRA_FEAT_DIM))
        )
        self.central.to(self.device)

        self.trainer = OscillationTrainer(
            hemis=self.bundle,
            central=self.central,
            cfg=OscillationConfig(
                emb=emb,
                n_actions=n_actions,
                lr_E=lr_E,
                lr_I=lr_I,
                lr_C=lr_C,
                device=str(self.device),
            ),
        )
        self.ada = CentralAdaOnline(
            central=self.central,
            net_I=self._net_I,
            net_E=self._net_E,
            device=self.device,
        )
        self._last_state: BrainState | None = None
        self._last_ada: dict[str, Any] = {}

    def reset_episode(self, game_id: str | None = None) -> None:
        """Wipe episode memory + Ada buffer (new game / tabula rasa)."""
        self.I.reset_episode(game_id=game_id)
        self.ada.set_game(game_id if game_id is not None else f"ep_{id(self)}")
        self._last_state = None

    def observe(
        self,
        frame: Any,
        available_actions: list[int] | None = None,
    ) -> BrainState:
        """Map + Internal propose. No action commit."""
        self.I.net.eval()
        with torch.no_grad():
            out_I = self.I.run(frame, available_actions=available_actions)

        state = BrainState(
            frame=frame,
            trusted=bool(out_I.get("trusted", False)),
            gate_failures=list(out_I.get("gate_failures") or []),
            h_I=out_I["h_I"],
            h_E=None,
            dream_summary=dict(out_I.get("dream_summary") or {}),
            world_summary={},
            disagreement=0.0,
            judge_I=0.0,
            action_logits=None,
            propose_logits=out_I.get("propose_logits"),
            trust=None,
            mode_logits=None,
            truth_flags=TruthFlags(),  # all False
            extra_feats=out_I.get("extra_feats"),
            status=str((out_I.get("dream_summary") or {}).get("map_status", "ok")),
            map_view=out_I.get("map_view"),
        )
        # stash dream tensor for judge_plan (not JSON-serialized)
        state.dream_summary["_dream_next"] = out_I.get("dream_next")
        state.dream_summary["_action_for_dream"] = out_I.get("action_for_dream")
        self._last_state = state
        return state

    def judge_plan(
        self,
        frame: Any,
        action: int,
        plan: Any = None,
        state: BrainState | None = None,
    ) -> BrainState:
        """External judges Internal dream for the given action.

        `plan` is an optional hypothesis packet (never truth-admitted).
        E scores I's dream vs world pred for (frame, action).
        """
        prev = state or self._last_state
        self.I.net.eval()
        self.E.net.eval()
        self.central.eval()

        with torch.no_grad():
            frames = _as_frame_tensor(frame, self.device)
            act_t = torch.tensor(
                [int(action) % self.n_actions], device=self.device, dtype=torch.long
            )
            h_I = self._net_I.encode(frames)
            dream = self._net_I.dream_next(frames, act_t)
            propose = self._net_I.propose_logits(frames)

            if prev is not None and prev.dream_summary:
                dream_summary = {
                    k: v
                    for k, v in prev.dream_summary.items()
                    if not str(k).startswith("_")
                }
                trusted = prev.trusted
                gate_failures = list(prev.gate_failures)
                extra = prev.extra_feats
            else:
                out_I = self.I.run(frame, action_for_dream=int(action))
                dream_summary = {
                    k: v
                    for k, v in (out_I.get("dream_summary") or {}).items()
                    if not str(k).startswith("_")
                }
                trusted = bool(out_I.get("trusted", False))
                gate_failures = list(out_I.get("gate_failures") or [])
                extra = out_I.get("extra_feats")

            if plan is not None and isinstance(plan, dict):
                dream_summary["plan_hyp"] = {
                    k: v
                    for k, v in plan.items()
                    if isinstance(v, (int, float, str, bool, list, type(None)))
                }

            out_E = self.E.run(frame, action, h_I=h_I, dream=dream)
            h_E = out_E["h_E"]
            judge = out_E.get("judge_I")
            disagree = out_E.get("disagreement")

            if extra is None:
                extra = pack_operator_feats(
                    dream_summary, dim=EXTRA_FEAT_DIM, device=self.device
                )

            out_C = self.central(
                h_I.detach(),
                h_E.detach(),
                judge_I=judge.detach() if judge is not None else None,
                disagreement=disagree.detach() if disagree is not None else None,
                extra=extra.detach() if isinstance(extra, torch.Tensor) else None,
            )

        new_state = BrainState(
            frame=frame,
            trusted=trusted,
            gate_failures=gate_failures,
            h_I=h_I,
            h_E=h_E,
            dream_summary=dream_summary,
            world_summary=dict(out_E.get("world_summary") or {}),
            disagreement=float(out_E.get("disagreement_scalar", 0.0)),
            judge_I=float(out_E.get("judge_I_scalar", 0.0)),
            action_logits=out_C["action_logits"],
            propose_logits=propose,
            trust=out_C["trust"],
            mode_logits=out_C["mode_logits"],
            truth_flags=TruthFlags(),
            extra_feats=extra,
            status="ok",
        )
        if plan is not None:
            new_state.world_summary["plan_provided"] = True
        new_state.dream_summary["_dream_next"] = dream
        new_state.dream_summary["_action_for_dream"] = int(action)
        self._last_state = new_state
        return new_state

    def act_logits(self, state: BrainState | None = None) -> torch.Tensor:
        """Central-only action logits. Requires h_I+h_E or prior judge_plan."""
        state = state or self._last_state
        if state is None:
            raise RuntimeError("act_logits requires a prior observe/judge_plan state")
        if state.action_logits is not None:
            return state.action_logits
        if state.h_I is None or state.h_E is None:
            raise RuntimeError(
                "act_logits needs h_I and h_E — call judge_plan first"
            )
        self.central.eval()
        with torch.no_grad():
            judge = torch.tensor(
                [state.judge_I], device=self.device, dtype=torch.float32
            )
            disagree = torch.tensor(
                [state.disagreement], device=self.device, dtype=torch.float32
            )
            extra = state.extra_feats
            out_C = self.central(
                state.h_I.to(self.device),
                state.h_E.to(self.device),
                judge_I=judge,
                disagreement=disagree,
                extra=extra.to(self.device) if extra is not None else None,
            )
        state.action_logits = out_C["action_logits"]
        state.trust = out_C["trust"]
        state.mode_logits = out_C["mode_logits"]
        return out_C["action_logits"]

    def train_transition(
        self,
        s: Any,
        a: int | torch.Tensor,
        s_prime: Any,
        extra: torch.Tensor | None = None,
        steps_per_phase: int = 1,
        game_id: str | None = None,
        ada: bool = True,
    ) -> list[dict[str, Any]]:
        """One oscillation cycle on a real (s, a, s′) transition.

        Always auto-fuses operator features from Internal.observe path into
        Phase C (creativity in the train path). Optional explicit `extra`
        overrides. Then AdaJEPA-style LoRA adapt on central.
        """
        frames = _as_frame_tensor(s, self.device)
        frames_next = _as_frame_tensor(s_prime, self.device)
        if isinstance(a, torch.Tensor):
            actions = a.long().reshape(-1).to(self.device)
        else:
            actions = torch.tensor(
                [int(a) % self.n_actions], device=self.device, dtype=torch.long
            )
        if frames.shape[0] != actions.shape[0]:
            if actions.shape[0] == 1:
                actions = actions.expand(frames.shape[0])
            else:
                n = min(frames.shape[0], actions.shape[0])
                frames = frames[:n]
                frames_next = frames_next[:n]
                actions = actions[:n]

        # Auto-fuse creativity: pack ops from first (or each) frame
        if extra is None:
            extras = []
            n = int(frames.shape[0])
            # Cap operator runs for batch efficiency (still always non-zero path)
            idxs = list(range(n)) if n <= 4 else [0, n // 3, 2 * n // 3, n - 1]
            cache: dict[int, torch.Tensor] = {}
            for i in idxs:
                fr = frames[i].detach().cpu().numpy()
                with torch.no_grad():
                    out_I = self.I.run(fr, action_for_dream=int(actions[i].item()))
                cache[i] = out_I.get("extra_feats")
                if cache[i] is None:
                    cache[i] = pack_operator_feats(
                        out_I.get("dream_summary") or {},
                        dim=EXTRA_FEAT_DIM,
                        device=self.device,
                    )
            # fill all batch rows (repeat nearest cached)
            filled = []
            for i in range(n):
                if i in cache:
                    filled.append(cache[i].reshape(1, -1))
                else:
                    # nearest
                    j = min(idxs, key=lambda k: abs(k - i))
                    filled.append(cache[j].reshape(1, -1))
            extra = torch.cat(filled, dim=0).to(self.device)

        logs = self.trainer.train_cycle(
            frames,
            actions,
            frames_next,
            steps_per_phase=steps_per_phase,
            extra=extra,
        )
        # episode memory seeds (ghost/delta/explore) grow across steps
        try:
            for i in range(int(frames.shape[0])):
                self.I.remember_transition(
                    frames[i].detach().cpu().numpy(),
                    int(actions[i].item()),
                    frames_next[i].detach().cpu().numpy(),
                )
        except Exception:
            pass

        if ada:
            self._last_ada = self.ada.observe_transition(
                frames, actions, frames_next, extra=extra, game_id=game_id
            )
            logs.append({"phase": "ada_online", **self._last_ada})
        # coverage breadcrumb for audits
        op_status = {}
        try:
            sample = self.I.run(frames[0].detach().cpu().numpy())
            ds = sample.get("dream_summary") or {}
            for k, v in ds.items():
                if isinstance(v, dict) and "status" in v:
                    op_status[k] = v.get("status")
        except Exception:
            pass
        logs.append(
            {
                "phase": "coverage",
                "auto_extra": True,
                "extra_dim": int(extra.shape[-1]) if extra is not None else 0,
                "extra_nonzero": float((extra.abs() > 0).float().mean())
                if extra is not None
                else 0.0,
                "wm": (self.E.world.summary() if self.E.world else {"status": "off"}),
                "op_status": op_status,
                "n_ops_ok": sum(1 for s in op_status.values() if s == "ok"),
                "n_ops": len(op_status),
            }
        )
        return logs
