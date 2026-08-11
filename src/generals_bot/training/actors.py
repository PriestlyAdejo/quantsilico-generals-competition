"""Persistent PPO actors: one env/opponent/belief/hidden across optimiser fragments.

Live cross-fragment continuity is mandatory. Exact mid-episode process-resume is
best-effort; episode-boundary checkpoint fallback is allowed without delaying PPO.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

from generals import GeneralsEnv
from generals.core import game

from generals_bot.core.belief import BeliefMemory
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.models.action_index import index_to_action
from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
from generals_bot.observation import GameContext
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.action_support import (
    MAX_EPISODE_TURN,
    SUPPORT_KIND_FULL_LEGAL_MASK,
    support_hash_from_mask,
)
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays
from generals_bot.training.conversion_reward import (
    RewardConfig,
    assert_no_privileged_keys,
    count_visible_enemy_cells,
)


@dataclass
class FragmentTransition:
    cells: np.ndarray
    glob: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    terminated: bool
    truncated: bool
    episode_id: str
    policy_version: int
    turn: int
    legal_mask: np.ndarray
    support_hash: str
    support_kind: str = "FULL_ACTION_SPACE_LEGAL_MASK"


@dataclass
class RolloutFragment:
    transitions: list[FragmentTransition]
    bootstrap_value: float
    continuation_mask: float  # 1.0 if fragment ends non-terminal
    episode_id: str
    policy_version: int
    actor_id: str
    map_seed: int
    learner_seat: int
    opponent_id: str
    state_hash_before: str
    state_hash_after: str


@dataclass
class PersistentActor:
    """Owns one competition env and continues episodes across PPO fragments."""

    actor_id: str
    seed: int
    reward_config: RewardConfig
    policy_version: int = 0
    learner_seat: int = 0
    opponent_id: str = "pass"
    checkpoint_resume_mode: str = "LIVE_ONLY"  # or PARTIAL_WITH_EPISODE_BOUNDARY_FALLBACK

    env: GeneralsEnv = field(init=False)
    transition_fn: Any = field(init=False)
    get_obs: Any = field(init=False)
    state: Any = field(init=False)
    h: int = field(init=False)
    w: int = field(init=False)
    episode_id: str = field(init=False)
    episode_index: int = 0
    turn: int = 0
    completed_games: int = 0
    map_seed: int = field(init=False)
    belief: BeliefMemory | None = field(default=None, init=False)
    opp_policy: Any = field(default=None, init=False)
    opp_state: Any = field(default=None, init=False)
    episode_shaping: float = 0.0
    prev_enemy_cells: int = 0
    discovered: bool = False
    action_history: list[int] = field(default_factory=list)
    hidden: Any = field(default=None, init=False)
    cell_mem: Any = field(default=None, init=False)
    _model_device: torch.device | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.env = GeneralsEnv(mode="competition")
        self.transition_fn = make_transition(self.env)
        self.get_obs = game.get_observation
        self.map_seed = int(self.seed)
        self.state = make_board(self.env, self.map_seed)
        self.h, self.w = (int(d) for d in self.state.armies.shape)
        self.episode_id = f"{self.actor_id}-ep{self.episode_index}-seed{self.map_seed}"
        self.belief = BeliefMemory.create(self.h, self.w)
        self.opponent_id = self.reward_config.training_opponent
        self._init_opponent()

    def _init_opponent(self) -> None:
        if self.reward_config.training_opponent == "pass":
            self.opp_policy = None
            self.opp_state = None
            self.opponent_id = "pass"
            return
        from generals_bot.policies.base import TraceLevel  # noqa: F401
        from generals_bot.selector import create_policy

        self.opp_policy = create_policy(self.reward_config.training_opponent, seed=self.seed)
        self.opp_state = self.opp_policy.initial_state(GameContext(1, self.h, self.w))
        self.opponent_id = self.reward_config.training_opponent

    def attach_model_state(self, model: nn.Module, device: torch.device) -> None:
        self._model_device = device
        self.hidden = model.initial_hidden(1, device=device)
        self.cell_mem = None
        if hasattr(model, "initial_cell_memory"):
            self.cell_mem = model.initial_cell_memory(1, device=device)

    def state_hash(self) -> str:
        armies = np.asarray(self.state.armies)
        owners = np.asarray(self.state.owners) if hasattr(self.state, "owners") else np.asarray(self.state.armies)
        payload = f"{self.episode_id}|{self.turn}|{armies.tobytes().hex()[:64]}|{owners.tobytes().hex()[:64]}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _reset_episode(self, model: nn.Module | None, *, next_seed: int) -> None:
        self.episode_index += 1
        self.completed_games += 1
        self.map_seed = int(next_seed)
        self.state = make_board(self.env, self.map_seed)
        self.h, self.w = (int(d) for d in self.state.armies.shape)
        self.episode_id = f"{self.actor_id}-ep{self.episode_index}-seed{self.map_seed}"
        self.turn = 0
        self.belief = BeliefMemory.create(self.h, self.w)
        self.episode_shaping = 0.0
        self.prev_enemy_cells = 0
        self.discovered = False
        self.action_history = []
        self._init_opponent()
        if model is not None and self._model_device is not None:
            self.hidden = model.initial_hidden(1, device=self._model_device)
            if hasattr(model, "initial_cell_memory"):
                self.cell_mem = model.initial_cell_memory(1, device=self._model_device)

    def collect_fragment(
        self,
        model: nn.Module,
        *,
        rollout_steps: int,
        device: torch.device,
        policy_version: int,
        gamma: float = 0.99,
        mixture_deterministic: bool = True,
    ) -> RolloutFragment:
        """Collect ``rollout_steps`` transitions without resetting mid-fragment env.

        On genuine terminal: reset once. On fragment end without terminal: bootstrap V(s').
        Opponent / belief / hidden persist across calls when not terminal.

        ``mixture_deterministic=True`` is Design A (provisional): mixture gate uses argmax
        so collection and update share identical option indices without RNG coupling.
        """
        del gamma  # reserved for future discounted collectors
        if self.hidden is None or self._model_device != device:
            self.attach_model_state(model, device)
        self.policy_version = policy_version
        model.eval()
        transitions: list[FragmentTransition] = []
        hash_before = self.state_hash()
        episode_at_start = self.episode_id

        for _ in range(rollout_steps):
            if self.turn > MAX_EPISODE_TURN:
                raise RuntimeError(
                    f"episode turn invariant violated: current_episode_turn={self.turn} "
                    f"> {MAX_EPISODE_TURN}"
                )
            eng = self.get_obs(self.state, self.learner_seat)
            tg, og, ag, _, meta = extract_numpy_boards(eng, self.h, self.w)
            cells = encode_grids_numpy(tg, og, ag)
            obs = _observation_from_arrays(tg, og, ag, meta)
            glob = encode_globals_numpy(obs)
            # Belief update from visible observation (no privileged hidden armies)
            if self.belief is not None:
                self.belief.update_visible(obs)

            cell_t = torch.from_numpy(cells).unsqueeze(0).to(device)
            glob_t = torch.from_numpy(glob).unsqueeze(0).to(device)
            with torch.no_grad():
                if self.cell_mem is not None:
                    raw = model.forward_tensors(
                        cell_t,
                        glob_t,
                        self.hidden,
                        self.cell_mem,
                        deterministic=mixture_deterministic,
                    )
                else:
                    flat = cell_t.reshape(1, -1) if isinstance(model, RecurrentMLPPolicy) else cell_t
                    raw = model.forward_tensors(
                        flat, glob_t, self.hidden, deterministic=mixture_deterministic
                    )
                fwd = adapt_forward_output(raw)
                if fwd.cell_memory is not None:
                    self.cell_mem = fwd.cell_memory
                self.hidden = fwd.hidden
                logits = fwd.logits
                mask = legal_mask_observation(obs, device=device).unsqueeze(0)
                masked = apply_action_mask(logits, mask)
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = fwd.value
            idx = int(action.item())
            assert bool(mask[0, idx]), "illegal action sampled"
            self.action_history.append(idx)
            mask_np = mask[0].detach().to(dtype=torch.bool, device="cpu").numpy()
            s_hash = support_hash_from_mask(mask_np)

            agent_action = index_to_action(idx)
            if self.opp_policy is None:
                opp_a = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
            else:
                from generals_bot.policies.base import TraceLevel

                eng1 = self.get_obs(self.state, 1)
                t1, o1, a1, _, m1 = extract_numpy_boards(eng1, self.h, self.w)
                obs1 = _observation_from_arrays(t1, o1, a1, m1)
                d1 = self.opp_policy.act(
                    obs1, self.opp_state, deterministic=True, trace=TraceLevel.NONE, deadline=None
                )
                self.opp_state = d1.new_state
                opp_a = _action_to_jax(d1.action)

            self.state, info = self.transition_fn(
                self.state, jnp.stack([_action_to_jax(agent_action), opp_a])
            )
            self.turn += 1

            eng_next = self.get_obs(self.state, self.learner_seat)
            _, og_next, _, _, _ = extract_numpy_boards(eng_next, self.h, self.w)
            next_enemy = count_visible_enemy_cells(og_next)
            assert_no_privileged_keys(
                {"owner_grid_visible": True, "prev_enemy_cells": self.prev_enemy_cells}
            )
            shaping, self.discovered = self.reward_config.contact_shaping.step_bonus(
                prev_enemy_cells=self.prev_enemy_cells,
                curr_enemy_cells=next_enemy,
                episode_cum=self.episode_shaping,
                discovered=self.discovered,
            )
            self.episode_shaping += shaping
            self.prev_enemy_cells = next_enemy

            reward = float(shaping)
            terminated = bool(info.is_done)
            truncated = False
            if terminated:
                winner = int(info.winner)
                term = self.reward_config.terminal.terminal_reward(
                    winner=None if winner < 0 else winner, perspective=self.learner_seat
                )
                reward += float(term)

            transitions.append(
                FragmentTransition(
                    cells=cells,
                    glob=glob,
                    action=idx,
                    logp=float(logp.item()),
                    value=float(value.item()),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    episode_id=self.episode_id,
                    policy_version=policy_version,
                    turn=self.turn,
                    legal_mask=mask_np,
                    support_hash=s_hash,
                    support_kind=SUPPORT_KIND_FULL_LEGAL_MASK,
                )
            )

            if terminated:
                # Reset exactly once on genuine terminal; do not reset at fragment boundaries.
                self._reset_episode(model, next_seed=self.map_seed + 1 + self.episode_index)

        # Bootstrap without advancing recurrent state twice: evaluate V at current (s', h')
        bootstrap_value = 0.0
        continuation = 0.0
        last = transitions[-1]
        if not last.terminated:
            continuation = 1.0
            with torch.no_grad():
                eng_b = self.get_obs(self.state, self.learner_seat)
                tb, ob, ab, _, mb = extract_numpy_boards(eng_b, self.h, self.w)
                cells_b = encode_grids_numpy(tb, ob, ab)
                obs_b = _observation_from_arrays(tb, ob, ab, mb)
                glob_b = encode_globals_numpy(obs_b)
                cell_bt = torch.from_numpy(cells_b).unsqueeze(0).to(device)
                glob_bt = torch.from_numpy(glob_b).unsqueeze(0).to(device)
                # Snapshot recurrent state so bootstrap forward cannot mutate actor memory.
                hidden_snap = self.hidden
                cell_snap = self.cell_mem
                if self.cell_mem is not None:
                    raw_b = model.forward_tensors(
                        cell_bt, glob_bt, hidden_snap, cell_snap, deterministic=True
                    )
                else:
                    flat_b = (
                        cell_bt.reshape(1, -1)
                        if isinstance(model, RecurrentMLPPolicy)
                        else cell_bt
                    )
                    raw_b = model.forward_tensors(flat_b, glob_bt, hidden_snap, deterministic=True)
                bootstrap_value = float(adapt_forward_output(raw_b).value.item())
                # Restore — do not advance recurrent state twice while constructing bootstrap.
                self.hidden = hidden_snap
                self.cell_mem = cell_snap

        return RolloutFragment(
            transitions=transitions,
            bootstrap_value=bootstrap_value,
            continuation_mask=continuation,
            episode_id=episode_at_start,
            policy_version=policy_version,
            actor_id=self.actor_id,
            map_seed=self.map_seed if last.terminated else self.map_seed,
            learner_seat=self.learner_seat,
            opponent_id=self.opponent_id,
            state_hash_before=hash_before,
            state_hash_after=self.state_hash(),
        )

    def snapshot_meta(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "turn": self.turn,
            "map_seed": self.map_seed,
            "h": self.h,
            "w": self.w,
            "learner_seat": self.learner_seat,
            "opponent_id": self.opponent_id,
            "policy_version": self.policy_version,
            "completed_games": self.completed_games,
            "action_history_len": len(self.action_history),
            "state_hash": self.state_hash(),
            "checkpoint_resume_mode": self.checkpoint_resume_mode,
            "belief_present": self.belief is not None,
            "opp_state_present": self.opp_state is not None,
        }
