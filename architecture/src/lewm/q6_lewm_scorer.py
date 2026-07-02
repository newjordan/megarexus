#!/usr/bin/env python3
"""Q6-in-LeWM Scorer Wrapper — T1 Interface Schema.

Wraps the Q6 rejection adapter as a LeWM/Rex candidate scorer.
Consumes LeWM candidate transition records and emits structured
Q6 decisions in target-fit format.

Target-fit contract (from QWEN_LEWM_TARGET_FIT_MATRIX):
  Input:  {game_id, episode_id, step_id, candidate_id, candidate_action,
           lewm_predicted_outcome, lewm_state_id, state_encoding_ref, ...}
  Output: {q6_decision: accept|reject|rank, q6_score, q6_reason,
           sao_refs, candidate_id, ...}

This module does NOT do online action selection — it is a scorer only.
The LeWM/Rex loop calls this and uses the returned scores.
"""
from __future__ import annotations

import json
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ── Scorer configuration ───────────────────────────────────────────────────

@dataclass
class Q6ScorerConfig:
    """Configuration for the Q6-in-LeWM scorer."""
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: str = "experiments/2026-05-23_q6_rejection_dataset_v1/q6_rejection_v2_0p5b_adapter/best_adapter"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    torch_dtype: str = "float16"
    # Future outcome policy: never expose future O_true in online prompts
    future_outcome_policy: str = "offline-label-only"


# ── Target-fit record types ────────────────────────────────────────────────

@dataclass
class CandidateRecord:
    """A single LeWM/Rex candidate transition that Q6 will score."""
    game_id: str
    game_full_id: str = ""
    episode_id: str = ""
    step_id: int = 0
    runner: str = "lewm_cem_discrete"
    lewm_checkpoint: str = ""
    qwen_adapter: str = ""
    candidate_id: str = ""          # unique per candidate
    lewm_state_id: str = ""         # state encoding reference
    state_encoding_ref: str = ""    # latent z or frame hash
    candidate_action: str = ""      # human-readable action description
    candidate_action_id: int = 0    # numeric action id
    lewm_predicted_outcome: str = ""  # what LeWM predicts will happen
    lewm_cost: float = 0.0           # LeWM's internal cost for this candidate
    observed_outcome: str | None = None  # offline-only, null in online
    future_outcome_policy: str = "offline-label-only"
    source_path: str = ""
    source_hash: str = ""
    truth_status: str = "recorded"  # recorded|shadow|online
    split: str = "test"             # train|val|test|live


@dataclass
class Q6DecisionRecord:
    """Q6's scoring decision for a LeWM candidate."""
    # Inherited from candidate
    game_id: str = ""
    episode_id: str = ""
    step_id: int = 0
    runner: str = ""
    lewm_checkpoint: str = ""
    qwen_adapter: str = ""
    candidate_id: str = ""
    lewm_state_id: str = ""
    state_encoding_ref: str = ""
    candidate_action: str = ""
    candidate_action_id: int = 0
    lewm_predicted_outcome: str = ""
    lewm_cost: float = 0.0
    observed_outcome: str | None = None
    future_outcome_policy: str = "offline-label-only"
    source_path: str = ""
    source_hash: str = ""
    truth_status: str = ""
    split: str = ""

    # Q6 scoring output
    q6_input_hash: str = ""
    q6_raw_response_hash: str = ""
    q6_raw_response: str = ""
    q6_decision: str = ""           # accept | reject | rank
    q6_score: float | None = None   # 0.0-1.0 confidence
    q6_reason: str = ""             # grounded reason
    sao_refs: list[str] = field(default_factory=list)
    q6_rejection_reason: str = ""   # rejection_reason tag if rejected

    # Metadata
    elapsed_ms: float = 0.0
    error: str = ""


# ── Prompt builder ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Q6 reasoning/scoring module inside the LeWM/Rex action system.
You consume LeWM candidate S/A/O transitions and emit accept/reject/rank decisions.

Rules:
1. REJECT if the candidate action is impossible (invalid coordinates, invalid action id, action after terminal state).
2. REJECT if the transition appears locally valid but the trajectory context suggests it will fail (myopic transition).
3. ACCEPT if the candidate action is valid and grounded in the S/A/O window.
4. Always cite S/A/O evidence for your decision.
5. Respond with the decision keyword first: REJECTED, ACCEPTED, or RANK.
6. Include rejection_reason: tag if rejecting."""


def build_candidate_prompt(candidate: CandidateRecord) -> list[dict[str, str]]:
    """Build a Q6-compatible prompt from a LeWM candidate record."""
    user_msg_parts = [
        f"Game: {candidate.game_full_id or candidate.game_id}",
        f"Episode: {candidate.episode_id}",
        f"Step: {candidate.step_id}",
        f"Candidate ID: {candidate.candidate_id}",
        f"Candidate Action: {candidate.candidate_action}",
        f"LeWM Predicted Outcome: {candidate.lewm_predicted_outcome}",
        f"LeWM Cost: {candidate.lewm_cost:.4f}",
        f"State Encoding Ref: {candidate.state_encoding_ref}",
    ]
    # Only include observed outcome in offline mode
    if candidate.future_outcome_policy == "offline-label-only" and candidate.observed_outcome:
        user_msg_parts.append(f"Observed Outcome (offline label): {candidate.observed_outcome}")
    elif candidate.future_outcome_policy == "online-visible":
        # In online mode, observed_outcome MUST be null
        pass

    user_content = "\n".join(user_msg_parts)
    user_content += "\n\nScore this candidate transition. Is it valid or should it be rejected?"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── Response parser ─────────────────────────────────────────────────────────

def parse_q6_response(raw_response: str) -> dict[str, Any]:
    """Parse Q6's text response into structured fields."""
    result: dict[str, Any] = {
        "q6_decision": "",
        "q6_score": None,
        "q6_reason": "",
        "sao_refs": [],
        "q6_rejection_reason": "",
    }

    text = raw_response.strip()

    # Decision detection
    if text.upper().startswith("REJECTED"):
        result["q6_decision"] = "reject"
    elif text.upper().startswith("ACCEPTED"):
        result["q6_decision"] = "accept"
    elif text.upper().startswith("RANK"):
        result["q6_decision"] = "rank"
        # Try to extract a numeric score
        score_match = re.search(r"score[:\s]*([\d.]+)", text, re.IGNORECASE)
        if score_match:
            result["q6_score"] = float(score_match.group(1))
    else:
        # Heuristic: if it contains rejection_reason, it's a reject
        if "rejection_reason:" in text.lower():
            result["q6_decision"] = "reject"
        else:
            result["q6_decision"] = "accept"  # default: no rejection language

    # Extract rejection reason
    reason_match = re.search(r"rejection_reason:\s*(\S+)", text, re.IGNORECASE)
    if reason_match:
        result["q6_rejection_reason"] = reason_match.group(1)

    # Extract S/A/O references
    sao_pattern = re.compile(r"(?:S/A/O|State/Action/Outcome|t\d+|Evidence ID:\s*\S+)", re.IGNORECASE)
    result["sao_refs"] = sao_pattern.findall(text)[:10]

    # Extract reason (everything after the first sentence)
    lines = text.split("\n")
    if len(lines) > 1:
        result["q6_reason"] = " ".join(lines[1:])[:500]
    else:
        result["q6_reason"] = text[:500]

    return result


# ── Scorer class ────────────────────────────────────────────────────────────

class Q6LewmScorer:
    """Q6 reasoning/scoring module wired into the LeWM/Rex candidate loop.

    Usage:
        scorer = Q6LewmScorer(config)
        scorer.load()

        for candidate in lewm_candidates:
            decision = scorer.score(candidate)
            # decision.q6_decision is "accept", "reject", or "rank"
    """

    def __init__(self, config: Q6ScorerConfig | None = None):
        self.config = config or Q6ScorerConfig()
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        print(f"Loading Q6 scorer: {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True,
        )
        dtype = getattr(torch, self.config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, torch_dtype=dtype,
            device_map=self.config.device, trust_remote_code=True,
        )

        adapter_path = Path(self.config.adapter_path)
        if adapter_path.exists():
            print(f"  Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self._loaded = True
        print(f"  Device: {self.model.device}")

    def score(self, candidate: CandidateRecord) -> Q6DecisionRecord:
        """Score a single LeWM candidate transition through Q6."""
        if not self._loaded:
            self.load()

        t0 = time.time()

        # Build prompt
        messages = build_candidate_prompt(candidate)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Hash input for traceability
        q6_input_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

        # Generate
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw_response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        raw_hash = hashlib.sha256(raw_response.encode()).hexdigest()[:16]

        # Parse response
        parsed = parse_q6_response(raw_response)

        # Build decision record
        decision = Q6DecisionRecord(
            game_id=candidate.game_id,
            episode_id=candidate.episode_id,
            step_id=candidate.step_id,
            runner=candidate.runner,
            lewm_checkpoint=candidate.lewm_checkpoint,
            qwen_adapter=self.config.adapter_path,
            candidate_id=candidate.candidate_id,
            lewm_state_id=candidate.lewm_state_id,
            state_encoding_ref=candidate.state_encoding_ref,
            candidate_action=candidate.candidate_action,
            candidate_action_id=candidate.candidate_action_id,
            lewm_predicted_outcome=candidate.lewm_predicted_outcome,
            lewm_cost=candidate.lewm_cost,
            observed_outcome=candidate.observed_outcome,
            future_outcome_policy=candidate.future_outcome_policy,
            source_path=candidate.source_path,
            source_hash=candidate.source_hash,
            truth_status=candidate.truth_status,
            split=candidate.split,
            q6_input_hash=q6_input_hash,
            q6_raw_response_hash=raw_hash,
            q6_raw_response=raw_response,
            q6_decision=parsed["q6_decision"],
            q6_score=parsed["q6_score"],
            q6_reason=parsed["q6_reason"],
            sao_refs=parsed["sao_refs"],
            q6_rejection_reason=parsed.get("q6_rejection_reason", ""),
            elapsed_ms=(time.time() - t0) * 1000,
        )

        return decision

    def score_batch(
        self, candidates: list[CandidateRecord],
    ) -> list[Q6DecisionRecord]:
        """Score a batch of candidates."""
        return [self.score(c) for c in candidates]


# ── Convenience: convert decision to dict ──────────────────────────────────

def decision_to_target_fit_dict(decision: Q6DecisionRecord) -> dict[str, Any]:
    """Convert a Q6DecisionRecord to the target-fit JSON schema."""
    d = asdict(decision)
    # Ensure all required fields are present
    required = [
        "game_id", "episode_id", "step_id", "runner", "lewm_checkpoint",
        "qwen_adapter", "candidate_id", "lewm_state_id", "state_encoding_ref",
        "candidate_action", "lewm_predicted_outcome", "observed_outcome",
        "future_outcome_policy", "q6_input_hash", "q6_raw_response_hash",
        "q6_decision", "q6_score", "q6_reason", "sao_refs",
        "truth_status", "split", "source_path", "source_hash",
    ]
    return {k: d.get(k) for k in required if k in d}


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Q6-in-LeWM Scorer — Smoke Test")
    print("=" * 60)

    # Create a dummy candidate
    candidate = CandidateRecord(
        game_id="g50t",
        game_full_id="g50t-5849a774",
        episode_id="bad_runs_v2/g50t-5849a774_bad/episode_00001",
        step_id=42,
        candidate_id="cand_g50t_00001_0042_a3",
        lewm_state_id="z_g50t_00001_0042",
        state_encoding_ref="frame_hash:a1b2c3d4",
        candidate_action="click(65,65)",
        candidate_action_id=9,
        lewm_predicted_outcome="NOT_FINISHED",
        lewm_cost=12.5,
        observed_outcome="GAME_OVER (terminal)",  # offline label
        future_outcome_policy="offline-label-only",
        source_path="experiments/remote_bad_runs/bad_runs_v2/g50t-5849a774_bad.pkl",
        source_hash="abcd1234",
        truth_status="recorded",
        split="test",
    )

    print(f"\nCandidate: {candidate.candidate_id}")
    print(f"  Game: {candidate.game_full_id}")
    print(f"  Action: {candidate.candidate_action}")
    print(f"  Observed: {candidate.observed_outcome}")

    # Build prompt (no model load needed for this test)
    prompt = build_candidate_prompt(candidate)
    print(f"\nPrompt ({len(str(prompt))} chars):")
    for m in prompt:
        print(f"  [{m['role']}]: {m['content'][:200]}...")

    # Test response parser
    print("\nResponse parsing tests:")
    tests = [
        ("REJECTED: Impossible transition at t42.\nrejection_reason: impossible_transition_action_after_terminal\nEvidence: t41: GAME_OVER, t42: click(65,65) is illegal.",
         {"decision": "reject", "reason_tag": "impossible_transition_action_after_terminal"}),
        ("ACCEPTED. The candidate action click(30,30) is within the valid 64x64 grid. State encoding shows NOT_FINISHED, action is legal.",
         {"decision": "accept", "reason_tag": ""}),
        ("REJECTED: Myopic transition — locally valid S/A/O window but trajectory outcome is loss.\nrejection_reason: myopic_transition_looks_valid_but_trajectory_lost",
         {"decision": "reject", "reason_tag": "myopic_transition_looks_valid_but_trajectory_lost"}),
    ]

    for raw, expected in tests:
        parsed = parse_q6_response(raw)
        ok = (
            parsed["q6_decision"] == expected["decision"]
            and parsed.get("q6_rejection_reason", "") == expected["reason_tag"]
        )
        status = "✅" if ok else "❌"
        print(f"  {status} '{raw[:80]}...'")
        print(f"     decision={parsed['q6_decision']}, reason={parsed.get('q6_rejection_reason','')}")

    print(f"\n✅ Scorer wrapper smoke test complete")
    print(f"   To use: scorer = Q6LewmScorer(); scorer.load(); decision = scorer.score(candidate)")
