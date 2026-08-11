"""Competition-safe hybrid: heuristic proposals ranked by a BC CNN checkpoint.

Imports only ``generals_bot.models.*`` and ``generals_bot.policies.*`` (no training).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from generals_bot.action import PASS_ACTION, Action
from generals_bot.legal import is_legal_action
from generals_bot.models.action_index import PASS_INDEX, action_to_index
from generals_bot.models.checkpoint import apply_state_dict, load_checkpoint_payload
from generals_bot.models.factory import build_model
from generals_bot.models.heads import STRATEGIC_OPTIONS
from generals_bot.models.legal_mask import legal_mask_observation
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.models.observation_encoder import encode_globals, encode_observation
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, Proposal, TraceLevel
from generals_bot.policies.heuristic_v2_ablations import (
    V2F_PLANNER_TERMINAL,
    create_ablation,
)
from generals_bot.risk.shield import SurvivalShield, proposal_rank_key

logger = logging.getLogger(__name__)

OPTION_TO_IDX = {name: i for i, name in enumerate(STRATEGIC_OPTIONS)}
# Heuristic option aliases → mixture vocabulary.
_OPTION_ALIASES = {
    "CASTLE": "BUILD",
    "WAIT": "WAIT",
    "IMMEDIATE_TERMINAL_WIN": "DEATHTOUCH",
    "GENERAL_HUNT": "GENERAL_HUNT",
}


@dataclass(frozen=True)
class HybridConfidenceConfig:
    """Provisional confidence gate (calibrate before challenger eval seeds)."""

    min_top2_margin: float = 0.05
    max_normalised_entropy: float = 0.98
    min_support_size: int = 1


def option_to_index(option: str | None) -> int:
    if not option:
        return OPTION_TO_IDX["WAIT"]
    key = _OPTION_ALIASES.get(option, option)
    return OPTION_TO_IDX.get(key, OPTION_TO_IDX["WAIT"])


def canonicalize_proposals(proposals: list[Proposal]) -> list[Proposal]:
    """Map to flat action IDs; keep highest hard_priority meta; PASS at most once."""
    best: dict[int, Proposal] = {}
    pass_prop: Proposal | None = None
    for p in proposals:
        idx = action_to_index(p.action)
        if idx == PASS_INDEX:
            if pass_prop is None or proposal_rank_key(p) < proposal_rank_key(pass_prop):
                pass_prop = p
            continue
        existing = best.get(idx)
        if existing is None or proposal_rank_key(p) < proposal_rank_key(existing):
            best[idx] = p
    out = list(best.values())
    if pass_prop is not None:
        out.append(pass_prop)
    elif not out:
        out.append(
            Proposal(
                action=PASS_ACTION,
                option="WAIT",
                module="canonicalize_pass",
                hard_priority=0,
                score=0.0,
                confidence=1.0,
                explanation_code="canonicalize_empty_pass",
            )
        )
    return out


def _candidate_confidence(
    logits: Tensor,
    candidate_indices: list[int],
    legal_mask: Tensor,
) -> tuple[int | None, float, float, int, bool]:
    """Return (best_idx, top2_margin, normalised_entropy, support_size, ok_support)."""
    if not candidate_indices:
        return None, 0.0, 1.0, 0, False
    device = logits.device
    mask = legal_mask.to(device=device, dtype=torch.bool).reshape(-1)
    flat = logits.reshape(-1)
    # Restrict to candidates that are legal under the observation mask.
    legal_cands = [i for i in candidate_indices if bool(mask[i].item())]
    support = len(legal_cands)
    if support == 0:
        return None, 0.0, 1.0, 0, False
    idx_t = torch.tensor(legal_cands, dtype=torch.long, device=device)
    cand_logits = flat.index_select(0, idx_t)
    probs = F.softmax(cand_logits, dim=-1)
    order = torch.argsort(probs, descending=True)
    best_local = int(order[0].item())
    best_idx = legal_cands[best_local]
    if support == 1:
        margin = 1.0
        norm_entropy = 0.0
    else:
        top1 = float(probs[order[0]].item())
        top2 = float(probs[order[1]].item())
        margin = top1 - top2
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        norm_entropy = ent / math.log(support)
    return best_idx, margin, norm_entropy, support, True


class HybridBcRankerPolicy:
    """Heuristic proposal generators + BC CNN ranker + SurvivalShield fallback."""

    policy_id = "hybrid_bc_ranker"

    def __init__(
        self,
        checkpoint_json: Path | str,
        *,
        fallback_policy_name: str = V2F_PLANNER_TERMINAL,
        device: str = "cpu",
        confidence: HybridConfidenceConfig | None = None,
    ) -> None:
        self.checkpoint_json = Path(checkpoint_json)
        self.fallback_policy_name = fallback_policy_name
        self.device = torch.device(device)
        self.confidence = confidence or HybridConfidenceConfig()
        self._fallback = create_ablation(fallback_policy_name)
        self._model = None
        self._architecture: str | None = None
        self._load_failed = False
        self._load_failure_logged = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            payload = load_checkpoint_payload(self.checkpoint_json)
            arch = str(payload.get("architecture") or payload.get("config", {}).get("architecture"))
            model = build_model(arch, payload.get("config")).to(self.device)
            apply_state_dict(model, self.checkpoint_json, map_location=self.device)
            model.eval()
            self._model = model
            self._architecture = arch
        except Exception as exc:  # noqa: BLE001 — competition: heuristic-only forever
            self._model = None
            self._load_failed = True
            self._log_load_failure(exc)

    def _log_load_failure(self, exc: BaseException) -> None:
        if self._load_failure_logged:
            return
        self._load_failure_logged = True
        logger.warning(
            "HybridBcRankerPolicy: checkpoint load failed (%s); heuristic-only forever",
            exc,
        )

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and not self._load_failed

    def initial_state(self, context: GameContext) -> PolicyState:
        state = self._fallback.initial_state(context)
        self._reset_neural(state)
        return state

    def _reset_neural(self, state: PolicyState) -> None:
        state.data["hybrid_hidden"] = None
        state.data["hybrid_cell"] = None
        state.data["previous_option_index"] = 0
        state.data["previous_action"] = PASS_ACTION
        state.data["hybrid_forward_count"] = 0

    def _ensure_hidden(self, state: PolicyState) -> tuple[Tensor, Tensor | None]:
        assert self._model is not None
        hidden = state.data.get("hybrid_hidden")
        if hidden is None:
            hidden = self._model.initial_hidden(1, device=self.device)
            state.data["hybrid_hidden"] = hidden
        cell = state.data.get("hybrid_cell")
        if cell is None and hasattr(self._model, "initial_cell_memory"):
            cell = self._model.initial_cell_memory(1, device=self.device)
            state.data["hybrid_cell"] = cell
        return hidden, cell

    def _forward_once(
        self,
        observation: Observation,
        state: PolicyState,
    ) -> tuple[Tensor, Tensor]:
        """Exactly one recurrent forward; always advances hidden/cell."""
        assert self._model is not None
        cells = encode_observation(observation, device=self.device).unsqueeze(0)
        glob = encode_globals(observation, device=self.device).unsqueeze(0)
        hidden, cell = self._ensure_hidden(state)
        prev_opt = int(state.data.get("previous_option_index") or 0)
        previous_option = torch.tensor([prev_opt], dtype=torch.long, device=self.device)
        kwargs: dict = {
            "deterministic": True,
            "previous_option": previous_option,
        }
        with torch.no_grad():
            if cell is not None:
                raw = self._model.forward_tensors(cells, glob, hidden, cell, **kwargs)
            else:
                raw = self._model.forward_tensors(cells, glob, hidden, **kwargs)
            fwd = adapt_forward_output(raw)
        state.data["hybrid_hidden"] = fwd.hidden.detach()
        if fwd.cell_memory is not None:
            state.data["hybrid_cell"] = fwd.cell_memory.detach()
        state.data["hybrid_forward_count"] = int(state.data.get("hybrid_forward_count") or 0) + 1
        return fwd.logits.reshape(-1), fwd.hidden

    def _heuristic_only(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        trace: TraceLevel,
        deadline: float | None,
        reason: str,
    ) -> ActionDecision:
        decision = self._fallback.act(
            observation, state, deterministic=True, trace=trace, deadline=deadline
        )
        decision.policy_id = self.policy_id
        decision.fallback_used = True
        decision.model_id = None
        shield = dict(decision.shield_result or {})
        shield["hybrid"] = "heuristic_only"
        shield["hybrid_reason"] = reason
        decision.shield_result = shield
        state = decision.new_state
        state.data["previous_option_index"] = option_to_index(decision.strategic_option)
        state.data["previous_action"] = decision.action
        decision.new_state = state
        return decision

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        del deterministic
        if not self.model_loaded:
            return self._heuristic_only(
                observation, state, trace=trace, deadline=deadline, reason="load_failed"
            )

        proposals, state, legal = self._fallback.generate_proposals(
            observation, state, deadline=deadline
        )
        canon = canonicalize_proposals(proposals)
        legal_mask = legal_mask_observation(observation, device=self.device)
        logits, _hidden = self._forward_once(observation, state)

        cand_indices = [action_to_index(p.action) for p in canon]
        best_idx, margin, norm_ent, support, ok_support = _candidate_confidence(
            logits, cand_indices, legal_mask
        )
        conf = self.confidence
        confidence_ok = (
            ok_support
            and support >= conf.min_support_size
            and margin >= conf.min_top2_margin
            and norm_ent <= conf.max_normalised_entropy
            and best_idx is not None
        )

        chosen: Proposal | None = None
        path = "fallback_shield"
        if confidence_ok and best_idx is not None:
            for p in canon:
                if action_to_index(p.action) == best_idx:
                    chosen = p
                    break
            if chosen is not None:
                shielded = SurvivalShield().select(observation, [chosen], legal)
                if (
                    shielded is not None
                    and shielded.action.as_tuple() == chosen.action.as_tuple()
                    and is_legal_action(observation, chosen.action)
                ):
                    path = "bc_rank"
                else:
                    chosen = None
                    path = "bc_safety_fail"

        if chosen is None:
            chosen = SurvivalShield().select(observation, canon, legal)
            path = "fallback_shield" if path != "bc_safety_fail" else path

        assert chosen is not None
        if not is_legal_action(observation, chosen.action):
            # Last resort: full proposal set then PASS.
            chosen = SurvivalShield().select(observation, proposals or canon, legal)
            if chosen is None or not is_legal_action(observation, chosen.action):
                chosen = Proposal(
                    action=PASS_ACTION,
                    option="WAIT",
                    module="hybrid_illegal_pass",
                    hard_priority=0,
                    score=0.0,
                    confidence=1.0,
                    explanation_code="hybrid_illegal_pass",
                )
            path = "illegal_recover"

        state.data["previous_option_index"] = option_to_index(chosen.option)
        state.data["previous_action"] = chosen.action

        diag = dict(state.data.get("diagnostics") or {})
        diag["hybrid_path"] = path
        diag["hybrid_margin"] = margin
        diag["hybrid_norm_entropy"] = norm_ent
        diag["hybrid_support"] = support
        diag["hybrid_forward_count"] = int(state.data.get("hybrid_forward_count") or 0)
        state.data["diagnostics"] = diag

        return ActionDecision(
            action=chosen.action,
            new_state=state,
            strategic_option=chosen.option,
            option_distribution={chosen.option: 1.0},
            policy_id=self.policy_id,
            model_id=self._architecture,
            confidence=float(chosen.confidence),
            legal_action_count=len(legal),
            top_candidates=[p.action for p in canon[:8]],
            proposals=canon if trace != TraceLevel.NONE else [],
            fallback_used=path != "bc_rank",
            shield_result={
                "hybrid": path,
                "margin": margin,
                "norm_entropy": norm_ent,
                "support": support,
                "confidence_ok": confidence_ok,
                "selected_module": chosen.module,
            },
        )
