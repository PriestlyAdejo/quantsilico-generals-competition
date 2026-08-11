# Candidate B hybrid BC builder

Created as part of Phase 9FU Stage 1A (packaging path only).

## What was built

- Shared `HeuristicV2AblationPolicy.generate_proposals` (used by `act` and hybrid).
- Hunt filter also strips `option=="BUILD"` (legacy `"CASTLE"` kept).
- Competition-safe `HybridBcRankerPolicy` (`src/generals_bot/policies/hybrid_bc_ranker.py`).
- `build_hybrid_bc_package` + `promote_package_to_submission` under `submission/` (not `dist/`).
- Packaging script: `scripts/phase9fu_package_hybrid_bc.py` → `QS-P9FU-HYBRID-BC-V1`.

## Pipeline

```text
generate_proposals → canonicalize → legal_mask → one BC forward →
rank candidates → confidence (margin/entropy/support) → SurvivalShield on chosen
→ else SurvivalShield on full set → verify legal
```

Load failure → heuristic-only forever (zero neural forwards; no fabricated hidden).

## Confidence

`HybridConfidenceConfig` defaults are **provisional** (not yet calibrated on frozen BC validation). Calibrate before challenger behavioural seeds (`HYBRID_CONFIDENCE_CALIBRATION_GATE`).

## Package (this run)

```text
.\.venv-training\Scripts\python.exe scripts/phase9fu_package_hybrid_bc.py
```

| Field | Value |
|-------|-------|
| Status | `PACKAGED` |
| Path | `submission/packages/QS-P9FU-HYBRID-BC-V1/5152a08eb774cf0e/package.zip` |
| SHA-256 | `5152a08eb774cf0e29167e9469422834b0a6e40392a6035ccc0f830d50674b9f` |
| build_hash | `5152a08eb774cf0e` |
| Size | 1222953 bytes |

Registry updated with `hybrid_packages=PRESENT`. No behavioural or paired eval results claimed here.
