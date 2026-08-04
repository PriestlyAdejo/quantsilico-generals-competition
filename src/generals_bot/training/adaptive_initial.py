"""Adaptive INITIAL campaign with durable telemetry and validation score-rate stops."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
import yaml
from generals import GeneralsEnv
from generals.core import game

from generals_bot.action import PASS_ACTION
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.evaluation.qualification import score_rate
from generals_bot.models.action_index import index_to_action
from generals_bot.models.checkpoint import apply_state_dict
from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import apply_action_mask
from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel
from generals_bot.selector import create_policy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.campaign_telemetry import (
    append_log,
    new_campaign_record,
    persist_campaign,
    sample_hardware,
)
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays
from generals_bot.training.ppo import run_bounded_ppo

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs" / "training" / "initial" / "adaptive_initial_v1.yaml"
DEFAULT_READINESS = REPO / "experiments" / "manifests" / "initial_readiness_gate.json"
DEFAULT_OUT = REPO / "experiments" / "manifests" / "adaptive_initial_campaign.json"


@dataclass
class CheckpointPolicy:
    """In-process policy backed by a Torch checkpoint (INITIAL validation only)."""

    architecture: str
    checkpoint: Path
    device: str = "cpu"
    policy_id: str = "learned_checkpoint"

    def __post_init__(self) -> None:
        self._model = build_model(self.architecture).to(self.device)
        apply_state_dict(self._model, self.checkpoint, map_location=self.device)
        self._model.eval()

    def initial_state(self, context: GameContext) -> PolicyState:
        return PolicyState(data={"hidden": None, "cell": None, "h": context.height, "w": context.width})

    def act(
        self,
        observation: Observation,
        state: PolicyState,
        *,
        deterministic: bool,
        trace: TraceLevel,
        deadline: float | None,
    ) -> ActionDecision:
        h, w = observation.height, observation.width
        type_grid = np.asarray(observation.type_grid, dtype=np.int32)
        owner_grid = np.asarray(observation.owner_grid, dtype=np.int32)
        army_grid = np.asarray(observation.army_grid, dtype=np.int32)
        cells = encode_grids_numpy(type_grid, owner_grid, army_grid)
        glob = encode_globals_numpy(observation)
        cell_t = torch.from_numpy(cells).unsqueeze(0).to(self.device)
        glob_t = torch.from_numpy(glob).unsqueeze(0).to(self.device)
        hidden = state.data.get("hidden")
        if hidden is None:
            hidden = self._model.initial_hidden(1, device=torch.device(self.device))
        cell_mem = state.data.get("cell")
        if cell_mem is None and hasattr(self._model, "initial_cell_memory"):
            cell_mem = self._model.initial_cell_memory(1, device=torch.device(self.device))
        with torch.no_grad():
            if cell_mem is not None:
                out = self._model.forward_tensors(cell_t, glob_t, hidden, cell_mem, deterministic=deterministic)
                logits, value, new_h, new_c = out[0], out[1], out[2], out[3]
                state.data["cell"] = new_c
            else:
                out = self._model.forward_tensors(cell_t, glob_t, hidden, deterministic=deterministic)
                logits, value, new_h = out[0], out[1], out[2]
            state.data["hidden"] = new_h
            mask = None
            try:
                from generals_bot.models.legal_mask import legal_mask_observation

                mask = legal_mask_observation(observation).to(self.device)
                logits = apply_action_mask(logits, mask)
            except Exception:
                pass
            idx = int(torch.argmax(logits, dim=-1).item()) if deterministic else int(
                torch.distributions.Categorical(logits=logits).sample().item()
            )
        action = index_to_action(idx)
        return ActionDecision(
            action=action,
            new_state=state,
            policy_id=self.policy_id,
            value=float(value.item()) if hasattr(value, "item") else None,
        )


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def validate_checkpoint_vs_expander(
    *,
    architecture: str,
    checkpoint: Path,
    seeds: list[int],
    max_turns: int,
    device: str,
) -> dict[str, Any]:
    learned = CheckpointPolicy(architecture=architecture, checkpoint=checkpoint, device=device)
    opp = create_policy("official_expander", seed=0)
    wins = draws = losses = 0
    faults = 0
    for seed in seeds:
        env = GeneralsEnv(mode="competition")
        transition = make_transition(env)
        get_obs = game.get_observation
        state = make_board(env, seed)
        h, w = (int(d) for d in state.armies.shape)
        st0 = learned.initial_state(GameContext(0, h, w))
        st1 = opp.initial_state(GameContext(1, h, w))
        winner = None
        for _ in range(max_turns):
            eng0 = get_obs(state, 0)
            eng1 = get_obs(state, 1)
            t0, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
            t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
            obs0 = _observation_from_arrays(t0, o0, a0, m0)
            obs1 = _observation_from_arrays(t1, o1, a1, m1)
            try:
                d0 = learned.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                st0 = d0.new_state
                act0 = d0.action
            except Exception:
                faults += 1
                act0 = PASS_ACTION
            try:
                d1 = opp.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                st1 = d1.new_state
                act1 = d1.action
            except Exception:
                faults += 1
                act1 = PASS_ACTION
            state, info = transition(state, jnp.stack([_action_to_jax(act0), _action_to_jax(act1)]))
            if bool(info.is_done):
                winner = int(info.winner)
                break
        if winner is None or winner < 0:
            draws += 1
        elif winner == 0:
            wins += 1
        else:
            losses += 1
    n = wins + draws + losses
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": n,
        "score_rate": score_rate(wins, draws, losses) if n else None,
        "protocol_faults": faults,
        "seeds": seeds,
        "opponent": "official_expander",
    }


def run_adaptive_initial_for_candidate(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    campaign_id: str,
) -> dict[str, Any]:
    budgets = config["budgets"]
    validation = config["validation"]
    plateau = config["plateau"]
    opt = config["optimisation"]
    device = opt.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    architecture = str(candidate["architecture"])
    ckpt = Path(candidate["checkpoint"])
    max_env = int(budgets["max_env_steps_per_candidate"])
    max_wall = float(budgets["max_wall_clock_min"]) * 60.0
    chunk_steps = int(budgets["chunk_env_steps"])
    chunk_updates = int(budgets["chunk_updates"])
    min_vals = int(budgets["min_validation_evals"])
    games = int(validation["games_per_eval"])
    max_turns = int(validation["max_turns"])
    seed0 = int(validation["seeds_start"])
    patience = int(plateau["patience"])
    min_imp = float(plateau["min_improvement"])

    rec = new_campaign_record(
        campaign_id=campaign_id,
        stage="INITIAL",
        config_hash=_config_hash(config_path),
        architecture=architecture,
        checkpoint=str(ckpt),
    )
    rec["plateau_patience_remaining"] = patience
    append_log(rec, f"start candidate={candidate['arm_id']} arch={architecture}")
    persist_campaign(rec)

    t0 = time.perf_counter()
    env_steps = 0
    updates_done = 0
    val_history: list[dict[str, Any]] = []
    best_score: float | None = None
    best_ckpt = ckpt
    patience_left = patience
    stop_reason = None
    chunk_idx = 0

    while env_steps < max_env:
        elapsed = time.perf_counter() - t0
        if elapsed >= max_wall:
            stop_reason = "WALL_CLOCK_CAP"
            break
        remain_steps = max_env - env_steps
        steps = min(chunk_steps, remain_steps)
        updates = max(1, min(chunk_updates, max(1, steps // max(steps // chunk_updates, 1))))
        # Map steps to PPO: rollout_steps * updates ≈ env_steps
        rollout = max(16, steps // max(updates, 1))
        out_dir = REPO / "experiments" / "checkpoints" / "initial" / campaign_id / f"chunk_{chunk_idx}"
        append_log(rec, f"chunk={chunk_idx} rollout={rollout} updates={updates}")
        persist_campaign(rec)

        result = run_bounded_ppo(
            architecture=architecture,
            rollout_steps=rollout,
            updates=updates,
            lr=float(opt.get("lr", 3e-4)),
            clip=float(opt.get("clip", 0.2)),
            seed=int(candidate.get("seed") or 0) + chunk_idx,
            device=device,
            init_checkpoint=ckpt if ckpt.is_file() else None,
            out_dir=out_dir,
        )
        ckpt = Path(result["checkpoint"])
        env_steps += int(result.get("env_steps") or rollout * updates)
        updates_done += int(updates)
        chunk_idx += 1

        hist = result.get("history") or []
        last = hist[-1] if hist else {}
        rec["env_steps"] = env_steps
        rec["ppo_updates"] = updates_done
        rec["elapsed_s"] = time.perf_counter() - t0
        rec["current_checkpoint"] = str(ckpt)
        rec["metrics"] = {
            "policy_loss": last.get("policy_loss"),
            "value_loss": last.get("value_loss"),
            "entropy": last.get("entropy"),
            "approx_kl": last.get("approx_kl"),
            "clip_fraction": last.get("clip_fraction"),
            "explained_variance": last.get("explained_variance"),
            "grad_norm": last.get("grad_norm"),
            "learning_rate": last.get("learning_rate"),
        }
        rec["hardware"] = sample_hardware()
        persist_campaign(rec)

        # Validation eval
        seeds = [seed0 + updates_done * 10 + i for i in range(games)]
        val = validate_checkpoint_vs_expander(
            architecture=architecture,
            checkpoint=ckpt,
            seeds=seeds,
            max_turns=max_turns,
            device=device,
        )
        val_history.append(val)
        score = val.get("score_rate")
        rec["latest_validation"] = val
        rec["validation_game_count"] = sum(int(v.get("games") or 0) for v in val_history)
        append_log(rec, f"validation score_rate={score} wdl={val.get('wins')}/{val.get('draws')}/{val.get('losses')}")

        improved = False
        if isinstance(score, (int, float)):
            if best_score is None or float(score) >= float(best_score) + min_imp:
                best_score = float(score)
                best_ckpt = ckpt
                patience_left = patience
                improved = True
                rec["best_checkpoint"] = str(best_ckpt)
                rec["best_validation"] = val
            else:
                patience_left -= 1
        rec["plateau_patience_used"] = patience - patience_left
        rec["plateau_patience_remaining"] = patience_left
        rec["stop_conditions"] = [
            {
                "name": "plateau",
                "triggered": patience_left <= 0 and len(val_history) >= min_vals,
                "patience_left": patience_left,
            },
            {
                "name": "wall_clock",
                "triggered": (time.perf_counter() - t0) >= max_wall,
                "elapsed_s": time.perf_counter() - t0,
            },
            {
                "name": "env_steps",
                "triggered": env_steps >= max_env,
                "env_steps": env_steps,
            },
        ]
        persist_campaign(rec)

        if patience_left <= 0 and len(val_history) >= min_vals:
            stop_reason = "PLATEAU"
            break
        if not improved and len(val_history) >= min_vals and patience_left <= 0:
            stop_reason = "PLATEAU"
            break

    if stop_reason is None:
        stop_reason = "ENV_STEP_CAP" if env_steps >= max_env else "COMPLETED"

    rec["state"] = "COMPLETED"
    rec["final_stop_reason"] = stop_reason
    rec["elapsed_s"] = time.perf_counter() - t0
    persist_campaign(rec)

    return {
        "campaign_id": campaign_id,
        "arm_id": candidate["arm_id"],
        "architecture": architecture,
        "stop_reason": stop_reason,
        "env_steps": env_steps,
        "ppo_updates": updates_done,
        "elapsed_s": rec["elapsed_s"],
        "best_checkpoint": str(best_ckpt),
        "best_validation": rec.get("best_validation"),
        "validation_history": val_history,
        "telemetry_path": str(REPO / "var" / "dashboard" / "campaigns" / f"{campaign_id}.json"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-candidates", type=int, default=2)
    args = p.parse_args(argv)

    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    if readiness.get("decision") != "READY":
        payload = {
            "schema_version": 1,
            "kind": "ADAPTIVE_INITIAL_CAMPAIGN",
            "status": "SKIPPED",
            "reason": f"INITIAL_READINESS_GATE={readiness.get('decision')}",
            "readiness": readiness,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"decision": "SKIPPED", "path": str(args.out)}))
        return 0

    config = _load_config(args.config)
    selected = list(readiness.get("selected_candidates") or [])[: args.max_candidates]
    results = []
    for i, cand in enumerate(selected):
        cid = f"initial_{cand['slot']}_{cand['arm_id']}"
        results.append(
            run_adaptive_initial_for_candidate(
                candidate=cand,
                config=config,
                config_path=args.config,
                campaign_id=cid,
            )
        )

    # Rank by best validation score rate
    def _score(r: dict[str, Any]) -> float:
        bv = r.get("best_validation") or {}
        s = bv.get("score_rate")
        return float(s) if isinstance(s, (int, float)) else -1.0

    ranked = sorted(results, key=_score, reverse=True)
    payload = {
        "schema_version": 1,
        "kind": "ADAPTIVE_INITIAL_CAMPAIGN",
        "status": "COMPLETED",
        "config": str(args.config.relative_to(REPO)).replace("\\", "/"),
        "config_hash": _config_hash(args.config),
        "candidates": results,
        "ranked": [{"arm_id": r["arm_id"], "score_rate": _score(r), "stop_reason": r["stop_reason"]} for r in ranked],
        "best": ranked[0] if ranked else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": "COMPLETED", "path": str(args.out), "best": (ranked[0] if ranked else {}).get("arm_id")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
