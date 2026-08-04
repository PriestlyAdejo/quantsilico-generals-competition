# Heuristic v2 pre-PPO upload record

| Field | Value |
|---|---|
| candidate | heuristic_v2f_plus_planner_terminal_fix |
| package path | submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip |
| SHA-256 | e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa |
| config hash | 8f7405fe9834161c |
| package content source commit | `027ff5d` — verified by hashing embedded `terminal_win_oracle.py` (and other policy modules) against git; ZIP bytes match this commit |
| embedded package_manifest.bot_commit | `ee06778` — stale relative to content: package was rebuilt from a dirty tree on top of `ee06778` before `027ff5d` was committed; **do not treat this field as content identity** |
| repository completion commit | `26954e6` — UPLOAD_READY docs/parity marking only; not required to reproduce ZIP bytes |
| engine commit | 9e3b9d13cca51caa1bb07db48bb85c9e90ce0462 |
| Windows validation | PASS |
| Docker Linux parity | PASS — local approximation of the competition environment |
| PRE_PPO_SUBMISSION_GATE | PASS |
| HEURISTIC_DEVELOPMENT_GATE | FAIL — discovery remained below 0.60 (research status only) |
| UPLOAD_READY | true |
| lifecycle | SUBMITTED |
| portal display upload time | 4 Aug 2026, 00:37 |
| upload time (UTC) | 2026-08-03T23:37:00Z — inferred from BST browser-local display; preserve portal display time above |
| portal accept/build | PASS — QUALIFIED at **PORTAL_SUBMISSION_GATE** (not final-tournament qualification) |
| portal Expander gate | 2W / 1D / 0L — displayed as W / D / W |
| portal Hunter informational | 1W / 0D / 2L — displayed as L / W / L |
| portal faults | 0 |
| portal submission display name | heuristic_v2_preppo_8f7405fe983… |
| portal package size | 209 KB displayed (ZIP on disk 214205 bytes) |
| portal status URL | https://www.generals.bot/submissions |
| public player profile | https://www.generals.bot/player?name=QuantSilico&id=88151 |
| public player ID | 88151 |
| SUBMITTED | true |
| learned model included | false — heuristic-only package; no checkpoints |
| notes | Passed the portal submission gate and entered the leaderboard. This is not qualification for the final tournament. Operator deleted inferior older portal submissions, leaving this version active. PPO had not started when uploaded. Authoritative identity is package SHA-256, not repository HEAD. |

Manual steps archive: see `submission/MANUAL_UPLOAD_heuristic_v2_preppo.md`.
