# Competition-native JAX CUDA repair + daytime resume — terminal report

## Console summary

```text
NO UPLOAD-READY DAYTIME CANDIDATE

Primary blocker: BLOCKED_COMPUTE
Secondary: AWAITING_OPERATOR_ACTION (Ubuntu WSL first-user OOBE)
Deployment: TEACHER_REQUIRES_STUDENT (student shape feasible; not distilled)

Overnight parent: NO_VALID_OVERNIGHT_PARENT
Overnight: not started / not authorised
Upload: not performed
Phase 10: not authorised

Final status: AWAITING_PRE_OVERNIGHT_OPERATOR_REVIEW
```

## FINAL_RESPONSE_TEMPLATE (34 fields)

1. starting branch/commit: `research/phase9g-competition-native-jax-preovernight-v1` / `ba0e1b665491bfe6224f167c2f32a5c9a7464739`
2. ending branch/commit: same branch / `6ed651e98c0a93fb74ededf22d9eb61dfb170a86`
3. working-tree audit: PASSED (`experiments/manifests/competition_native_jax_working_tree_audit.json`); `dist/upload_ready` deletions non-canonical
4. frozen V001: `submission/packages/QS-PUBLIC-V001/e1237f77dee46993/package.zip` SHA `e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa`
5. provenance: `PARTIAL_EXTERNAL_PINS` (`competition_native_jax_provenance_gate.json`)
6. JAX architecture audit: `JAX_CORE_IMPLEMENTED_GPU_UNVERIFIED` (was `NUMPY_PROTOTYPE_ONLY`)
7. JAX files: `transformer_jax.py`, `inference_jax.py`, `rollout_selfplay_jax.py`, `gae_jax.py`, `ppo_jax.py`, `ema_jax.py`, `train_jax.py`
8. WSL distribution/version: Ubuntu WSL2 registered/Running; first-boot OOBE incomplete (non-interactive probe timeout)
9. GPU model/driver: NVIDIA GeForce RTX 3070 Laptop GPU / driver 581.42 / 8192 MiB
10. JAX/jaxlib (Windows .venv): 0.11.0 / 0.11.0 (CPU)
11. JAX backend/devices (Windows): cpu / `cpu:0`
12. device-placement proof: CPU correctness gate only (`grad_device=cpu:0`); GPU placement not verified
13. environment FPS: not measured under GPU_JAX_VERIFIED
14. full-rollout FPS: not measured under GPU_JAX_VERIFIED
15. valid learning TPS: not measured under GPU_JAX_VERIFIED (prior NumPy smoke ≈0.51 TPS, non-canonical)
16. compilation time: N/A (no GPU train)
17. peak VRAM: N/A
18. GPU correctness: CPU identity/rho PASSED; GPU_JAX_VERIFIED=false (`AWAITING_OPERATOR_ACTION`)
19. smoke (JAX GPU): NOT_RUN
20. short daytime: SKIPPED_WITH_REASON (`BLOCKED_COMPUTE`)
21. medium daytime: SKIPPED_WITH_REASON (short gate failed)
22. raw versus EMA: N/A_NO_TRAINED_CHECKPOINT
23. selected base checkpoint: none / `BASE_DAYTIME_BLOCKED_PACKAGE`
24. deployment backend: NumPy reference; teacher shape undeployable; student shapes feasible (random weights)
25. cold start and p50/p95/p99: teacher ~0.93 / 1.45 / 3.42 / 3.42 s; selected student emb96_d2 ~0.021 / 0.021 / 0.031 / 0.031 s
26. memory and package size: no package built
27. base package classification: `BASE_DAYTIME_BLOCKED_PACKAGE`
28. control dispositions: `NOT_JUSTIFIED_BY_EVIDENCE` / skipped
29. final candidate: `NO_CANDIDATE_CURRENTLY_RECOMMENDED`
30. exact package path and SHA: null
31. UPLOAD_THIS status: `NO_CANDIDATE_CURRENTLY_RECOMMENDED` (`submission/UPLOAD_THIS.md`)
32. overnight-parent classification: `NO_VALID_OVERNIGHT_PARENT` (write-only)
33. confirmation no programme process remains: programme-owned list cleared; hung WSL OOBE install may still exist OS-side until operator completes Ubuntu setup
34. exact next human decision: Complete Ubuntu first-user setup, run `scripts/windows/bootstrap_quantsilico_wsl_jax.ps1` to reach `GPU_JAX_VERIFIED`, then authorise a new daytime train/package programme. Do not overnight/upload/Phase 10 without new explicit authorisation.

## Exact next human action

1. Open **Ubuntu** from Start (or `wsl -d Ubuntu`) and finish UNIX username/password OOBE.
2. `powershell -ExecutionPolicy Bypass -File scripts/windows/bootstrap_quantsilico_wsl_jax.ps1`
3. Review this report + `experiments/manifests/competition_native_jax_final_recommendation.json`.
4. Do not upload; do not start overnight; do not claim Phase 10 readiness.
