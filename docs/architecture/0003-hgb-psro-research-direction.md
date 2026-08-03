# 0003 — HGB-PSRO research direction

## Status

Planned research direction. **Not implemented** in this bootstrap.

## Intent

```text
Hierarchical Graph-Belief Policy
+
PPO best-response training
+
lightweight Policy-Space Response Oracles
```

## Planned components (all: planned, not implemented)

| Component | Role | Status |
|-----------|------|--------|
| Partial-observation memory | Track fogged / latent state over time | planned, not implemented |
| Directional graph encoder | Encode map topology and ownership flows with JAX ops already available in the official environment | planned, not implemented |
| Strategic option gate | Select high-level options before low-level actions | planned, not implemented |
| Explicit no-new-risk option | Allow abstention / risk-neutral stance | planned, not implemented |
| Option-conditioned action policy | Map chosen option to legal actions | planned, not implemented |
| General-survival risk shield | Hard constraint protecting the general / solvency | planned, not implemented |
| Opponent-style belief | Infer opponent behavioural class | planned, not implemented |
| Policy population | Maintain a set of response policies | planned, not implemented |
| Empirical payoff matrix | Estimate matchup returns across the population | planned, not implemented |
| Meta-strategy solver | Compute mixture over the population | planned, not implemented |
| Best-response evaluation | Measure exploitability / improvement | planned, not implemented |
| Robust deployment policy | Final competition deployment mixture / selection | planned, not implemented |

## Non-goals for bootstrap

- No RL framework integration (Stable Baselines3, RLlib, etc.).
- No external graph libraries (PyTorch Geometric, DGL, Jraph).
- No fabricated training curves or win rates.

## Near-term path

1. Rules-conformance suite against the official engine.
2. Deterministic heuristic foundation (`heuristic_v0` / `heuristic_v1`).
3. Only then consider learning components above.
