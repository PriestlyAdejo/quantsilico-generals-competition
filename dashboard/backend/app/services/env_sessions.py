"""Bounded official Environment Lab sessions (localhost only)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.backend.app.paths import REPO_ROOT

SESSIONS_DIR = REPO_ROOT / "var" / "dashboard" / "env_sessions"
MAX_CONCURRENT = 2
DEFAULT_TTL_S = 15 * 60
MAX_TTL_S = 60 * 60
MAX_ACTIONS = 5000

_lock = threading.RLock()


def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or _now(), tz=timezone.utc).isoformat()


@dataclass
class EnvSession:
    session_id: str
    seed: int
    map_preset: str
    created_at: float
    expires_at: float
    action_count: int = 0
    turn: int = 0
    events: list[str] = field(default_factory=list)
    width: int = 18
    height: int = 18
    # Lightweight public board: list of rows of {terrain,owner,armies,visible}
    cells: list[list[dict[str, Any]]] = field(default_factory=list)
    p1_armies: int = 0
    p2_armies: int = 0
    p1_land: int = 0
    p2_land: int = 0
    fog_pct: float = 0.0
    closed: bool = False
    # Internal jax/game handles kept off disk
    _state: Any = field(default=None, repr=False)
    _env: Any = field(default=None, repr=False)
    _transition: Any = field(default=None, repr=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "seed": self.seed,
            "map_preset": self.map_preset,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "action_count": self.action_count,
            "turn": self.turn,
            "events": list(self.events[-50:]),
            "board": {
                "width": self.width,
                "height": self.height,
                "cells": self.cells,
                "turn": self.turn,
            },
            "telemetry": {
                "p1_armies": self.p1_armies,
                "p2_armies": self.p2_armies,
                "p1_land": self.p1_land,
                "p2_land": self.p2_land,
                "fog_pct": self.fog_pct,
            },
            "closed": self.closed,
            "limits": {
                "max_concurrent": MAX_CONCURRENT,
                "ttl_s": int(self.expires_at - self.created_at),
                "max_actions": MAX_ACTIONS,
            },
        }


_sessions: dict[str, EnvSession] = {}


def _serialize_board(state: Any, player: int = 0) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Build a fogged public board from official game state."""
    from generals.core import game as gcore
    import numpy as np

    obs = gcore.get_observation(state, player)
    armies = np.asarray(obs.armies)
    ownership = np.asarray(obs.ownership) if hasattr(obs, "ownership") else None
    # ownership layout in obs may be [2,h,w] or per-cell codes — prefer state ownership
    own0 = np.asarray(state.ownership[0])
    own1 = np.asarray(state.ownership[1])
    generals = np.asarray(state.generals)
    castles = np.asarray(state.castles) if hasattr(state, "castles") else np.zeros_like(armies)
    mountains = np.asarray(state.mountains) if hasattr(state, "mountains") else np.zeros_like(armies)
    visible = np.asarray(obs.visible) if hasattr(obs, "visible") else np.ones_like(armies, dtype=bool)
    vis_armies = np.asarray(obs.armies)

    h, w = int(armies.shape[0]), int(armies.shape[1])
    cells: list[list[dict[str, Any]]] = []
    fog = 0
    p1_a = p2_a = p1_l = p2_l = 0
    for r in range(h):
        row = []
        for c in range(w):
            vis = bool(visible[r, c])
            if not vis:
                fog += 1
                row.append({"terrain": "plain", "owner": "fog", "armies": 0, "visible": False})
                continue
            if bool(own0[r, c]):
                owner = "player1"
                p1_a += int(vis_armies[r, c])
                p1_l += 1
            elif bool(own1[r, c]):
                owner = "player2"
                p2_a += int(vis_armies[r, c])
                p2_l += 1
            else:
                owner = "neutral"
            if bool(generals[r, c]):
                terrain = "general"
            elif bool(castles[r, c]):
                terrain = "city"
            elif bool(mountains[r, c]):
                terrain = "mountain"
            else:
                terrain = "plain"
            row.append(
                {
                    "terrain": terrain,
                    "owner": owner,
                    "armies": int(vis_armies[r, c]),
                    "visible": True,
                }
            )
        cells.append(row)
    telem = {
        "p1_armies": p1_a,
        "p2_armies": p2_a,
        "p1_land": p1_l,
        "p2_land": p2_l,
        "fog_pct": fog / max(h * w, 1),
    }
    return cells, telem


def step_session(
    session_id: str,
    *,
    src_row: int,
    src_col: int,
    dst_row: int,
    dst_col: int,
) -> EnvSession:
    import jax.numpy as jnp
    from generals_bot.action import Action, KIND_MOVE, PASS_ACTION
    from generals_bot.training.collect_bc import _action_to_jax

    s = get_session(session_id)
    if s is None:
        raise KeyError("session not found")
    if s.action_count >= MAX_ACTIONS:
        raise RuntimeError("Maximum actions per session exceeded")

    state = s._state
    transition = s._transition
    if state is None or transition is None:
        raise RuntimeError("Session has no live environment state")

    h, w = int(state.armies.shape[0]), int(state.armies.shape[1])
    if not (0 <= src_row < h and 0 <= src_col < w and 0 <= dst_row < h and 0 <= dst_col < w):
        raise ValueError("Action coordinates out of bounds")

    cell = s.cells[src_row][src_col]
    if cell.get("owner") != "player1" or int(cell.get("armies") or 0) <= 1:
        raise ValueError("Illegal source cell — must own a visible cell with armies > 1")

    dr, dc = dst_row - src_row, dst_col - src_col
    dir_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    direction = dir_map.get((dr, dc))
    if direction is None:
        raise ValueError("Destination must be orthogonally adjacent")

    a0 = Action(kind=KIND_MOVE, row=src_row, col=src_col, direction=direction, split=0)
    actions = jnp.stack([_action_to_jax(a0), _action_to_jax(PASS_ACTION)])
    result = transition(state, actions)
    new_state = result[0] if isinstance(result, tuple) else result

    s._state = new_state
    s.action_count += 1
    s.turn += 1
    cells, telem = _serialize_board(new_state, 0)
    s.cells = cells
    s.width = len(cells[0]) if cells else s.width
    s.height = len(cells)
    s.p1_armies = telem["p1_armies"]
    s.p2_armies = telem["p2_armies"]
    s.p1_land = telem["p1_land"]
    s.p2_land = telem["p2_land"]
    s.fog_pct = telem["fog_pct"]
    s.events.append(f"Turn {s.turn}: ({src_row},{src_col})→({dst_row},{dst_col})")
    _persist_meta(s)
    return s


def _persist_meta(session: EnvSession) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in session.public_dict().items()}
    # Do not write full board cells to keep files small — summary only
    meta["board"] = {"width": session.width, "height": session.height, "turn": session.turn}
    path = SESSIONS_DIR / f"{session.session_id}.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def cleanup_expired() -> None:
    with _lock:
        now = _now()
        dead = [sid for sid, s in _sessions.items() if s.closed or s.expires_at < now]
        for sid in dead:
            _sessions.pop(sid, None)


def create_session(*, seed: int, map_preset: str = "standard", ttl_s: int | None = None) -> EnvSession:
    cleanup_expired()
    with _lock:
        active = [s for s in _sessions.values() if not s.closed and s.expires_at >= _now()]
        if len(active) >= MAX_CONCURRENT:
            raise RuntimeError(f"Maximum concurrent sessions ({MAX_CONCURRENT}) reached")
        ttl = min(max(ttl_s or DEFAULT_TTL_S, 60), MAX_TTL_S)

        from generals import GeneralsEnv
        from generals_bot.evaluation.match import make_board, make_transition

        env = GeneralsEnv(mode="competition")
        state = make_board(env, int(seed))
        transition = make_transition(env)
        cells, telem = _serialize_board(state, 0)
        h, w = len(cells), len(cells[0]) if cells else (18, 18)
        sid = f"env-{uuid.uuid4().hex[:12]}"
        session = EnvSession(
            session_id=sid,
            seed=int(seed),
            map_preset=map_preset,
            created_at=_now(),
            expires_at=_now() + ttl,
            events=[f"Session created seed={seed} preset={map_preset}"],
            width=w,
            height=h,
            cells=cells,
            p1_armies=telem["p1_armies"],
            p2_armies=telem["p2_armies"],
            p1_land=telem["p1_land"],
            p2_land=telem["p2_land"],
            fog_pct=telem["fog_pct"],
            _state=state,
            _env=env,
            _transition=transition,
        )
        _sessions[sid] = session
        _persist_meta(session)
        return session


def get_session(session_id: str) -> EnvSession | None:
    cleanup_expired()
    with _lock:
        s = _sessions.get(session_id)
        if s is None or s.closed or s.expires_at < _now():
            return None
        return s


def close_session(session_id: str) -> bool:
    with _lock:
        s = _sessions.get(session_id)
        if s is None:
            return False
        s.closed = True
        s.events.append("Session closed")
        _persist_meta(s)
        _sessions.pop(session_id, None)
        return True


def reset_session(session_id: str, *, seed: int | None = None) -> EnvSession:
    s = get_session(session_id)
    if s is None:
        raise KeyError("session not found")
    new_seed = s.seed if seed is None else int(seed)
    close_session(session_id)
    return create_session(seed=new_seed, map_preset=s.map_preset)


def legal_actions(session_id: str) -> dict[str, Any]:
    s = get_session(session_id)
    if s is None:
        raise KeyError("session not found")
    # Return owned cells with armies>1 as legal sources (public heuristic)
    sources = []
    for r, row in enumerate(s.cells):
        for c, cell in enumerate(row):
            if cell.get("visible") and cell.get("owner") == "player1" and int(cell.get("armies") or 0) > 1:
                sources.append({"row": r, "col": c, "armies": cell["armies"]})
    return {"schema_version": 1, "session_id": session_id, "sources": sources, "note": "Destinations validated on step."}


def shutdown_all() -> None:
    with _lock:
        for sid in list(_sessions.keys()):
            close_session(sid)
