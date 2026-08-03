# Heuristic v2 pre-PPO upload record — fill after portal confirmation

| Field | Value |
|---|---|
| candidate | heuristic_v2f_plus_planner_terminal_fix |
| package path | submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip |
| SHA-256 | e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa |
| config hash | 8f7405fe9834161c |
| bot commit | 027ff5d (packaging tree; confirm after final push) |
| engine commit | 9e3b9d13cca51caa1bb07db48bb85c9e90ce0462 |
| Windows validation | PASS |
| Docker Linux parity | PASS (approximation) |
| PRE_PPO_SUBMISSION_GATE | PASS |
| UPLOAD_READY | true |
| lifecycle | UPLOAD_READY — awaiting manual portal upload |
| upload time (UTC) | _operator fills_ |
| portal accept/build | _operator fills_ |
| portal status URL/id | _operator fills_ |
| SUBMITTED | false until operator confirms |
| notes | Keep local ZIP. Do not upload learned checkpoints. Do not start PPO until SUBMITTED. |

Manual steps: see `submission/MANUAL_UPLOAD_heuristic_v2_preppo.md`.
