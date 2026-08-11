# Phase 9F-R matched integrity

Created: 2026-08-05T09:48:41.402285+00:00

Expected init checkpoint: C:\Users\pries\Documents\Projects\quantsilico-generals-competition\experiments\phase9f_cnn_ranker_v1\checkpoints\bc\model.json
Reported init checkpoint in overnight manifest: C:\Users\pries\Documents\Projects\quantsilico-generals-competition\experiments\phase9f_cnn_ranker_v1\checkpoints\bc\model.json

SHA-256:
- model.json: ee70e00bf3883568865615ee6a73e638549f1b71d79439df7030704b103b000e
- model.safetensors: 4ba41f77a041f6f990f8f11ac5d02dab09e5cc79710c7edf52fd41c80d10f4bf

Gate results:
- MATCHED_SOURCE_GATE: PASS
- MATCHED_CONFIG_GATE: PASS
- TREATMENT_ISOLATION_GATE: PASS

Reward configs differ by design; the init checkpoint is the shared immutable source.
