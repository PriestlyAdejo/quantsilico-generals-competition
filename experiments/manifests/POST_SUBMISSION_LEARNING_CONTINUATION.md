# Post-submission learning continuation (do not run PPO in the packaging task)

After the operator records manual `SUBMITTED` for the second heuristic package:

1. Verify manual `SUBMITTED` record in `submission/UPLOAD_RECORD.md` (or the new package upload record).
2. Run `LEARNING_READINESS_GATE`.
3. Run recurrent MLP bridge benchmark.
4. Confirm JAX-to-PyTorch bridge PASS or PARTIAL.
5. Run CNN PPO smoke.
6. Verify checkpoint resume.
7. Verify safetensors official-CPU load/action.
8. Run bounded recurrent CNN control.
9. Run bounded pure-PyTorch recurrent graph-belief challenger (`recurrent_graph_belief_v2_pure_torch`).
10. Run ablations and population evaluation.
11. Promote only through `LEARNED_PROMOTION_GATE`.

## Graph / PyG policy (unchanged)

- Deployment graph path: pure PyTorch grid message passing — no PyG by default.
- Optional research-only `recurrent_graph_belief_pyg_v1` under `.venv-training` only if measured need.
- PyG must not enter the submitted runtime unless official `.venv`, offline, Linux parity, size, latency, memory, and measurable improvement all pass.

## Continuation command (after SUBMITTED is recorded)

```powershell
Set-Location 'C:\Users\pries\Documents\Projects\quantsilico-generals-competition'
# Begin LEARNING_READINESS_GATE + CNN PPO control track from the recorded SUBMITTED commit.
```

Do not start PPO until the operator confirms portal upload.
