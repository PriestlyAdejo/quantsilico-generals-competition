# Post-submission learning continuation

Status after commit `94c0a95` (SUBMITTED record) and subsequent Phase 5/6 smokes.

## Completed

1. Manual `SUBMITTED` record — `submission/UPLOAD_RECORD_heuristic_v2_preppo.md`
2. Portal observation — `experiments/manifests/official_portal_observation_heuristic_v2_preppo_2026-08-04.json` (`MANUALLY_RECORDED`)
3. Public attribution probe — no public package hash beside matches → `MANUAL_OPERATOR_ASSIGNMENT`
4. `LEARNING_READINESS_GATE` — **PASS** (`experiments/manifests/learning_readiness_gate.json`)
5. Bridge benchmark — **PASS** (existing `jax_pytorch_bridge_benchmark.json`)
6. BC tiny overfit — MLP / CNN / graph (`experiments/manifests/bc_tiny.json`)
7. Bounded PPO smoke — MLP / CNN / graph (legal_action_rate 1.0, resume_ok)
8. Official-CPU checkpoint load script — `scripts/dev/verify_official_cpu_checkpoint_load.py`
9. Equal-budget DEVELOPMENT comparison module — `generals_bot.training.equal_budget_compare`

## Model roles

| Role | Architecture |
|---|---|
| Engineering bridge | `recurrent_mlp_v1` |
| Learned control | `recurrent_cnn_v2` |
| Principal challenger | `recurrent_graph_belief_v2` / alias `recurrent_graph_belief_v2_pure_torch` |

PPO is the training algorithm. CNN and graph-belief are alternative encoders.

## Gate board

| Gate | Status |
|---|---|
| HEURISTIC_DEVELOPMENT_GATE | FAIL (discovery) |
| PRE_PPO_SUBMISSION_GATE | PASS |
| PORTAL_SUBMISSION_GATE | PASS (`QUALIFIED` ≠ final tournament) |
| LEARNING_READINESS_GATE | PASS |
| LEARNED_PROMOTION_GATE | NONE |
| PPO at upload | NOT STARTED |

## Graph / PyG policy (unchanged)

- Deployment graph path: pure PyTorch grid message passing — no PyG by default.
- Optional research-only PyG under `.venv-training` only if measured need.
- PyG must not enter the submitted runtime unless official `.venv`, offline, Linux parity, size, latency, memory, and measurable improvement all pass.

## Next campaign (still DEVELOPMENT; stop before INITIAL/OVERNIGHT/MARATHON)

```powershell
Set-Location 'C:\Users\pries\Documents\Projects\quantsilico-generals-competition'
.\.venv\Scripts\python.exe -m generals_bot.training.equal_budget_compare --env-steps 256 --updates 4 --seed 7
# Then: population/PFSP evaluation against Expander, Hunter, heuristic_v1, heuristic_v2f_plus_planner_terminal_force
# Do NOT promote or upload learned checkpoints.
```

Do not overwrite `submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip`.
