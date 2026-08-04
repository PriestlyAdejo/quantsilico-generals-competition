# Dashboard data / board / UX defect matrix

Milestone base: `a59c433`. Branch: `fix/dashboard-data-and-board-integrity`.
Screenshot ZIP: not present at start; seeded from operator review 2026-08-04.

Statuses: `DATA_MAPPING_BROKEN` | `DEMO_LEAK` | `BOARD_OVERFLOW` | `BROWSER_VARIANCE` | `MISSING_BACKEND_CAPABILITY` | `HONEST_EMPTY` | `COMPLETE`

| Route | Current behaviour | Expected | Endpoint / adapter | Missing / misleading | Board / browser | Required test | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overview | Qual-backed WDL/rates; NaN job progress | Honest metrics + provenance | `/api/overview` + qualification | — | — | formatter + adapter | COMPLETE |
| Arena | Allowlist + match job lifecycle | Allowlist match job E2E | `/api/jobs/*` | browser smoke pending evidence | waiting panel | Arena smoke script | DEMO_LEAK → in progress |
| Environment Lab | OFFICIAL sessions default; DEMO secondary | OFFICIAL sessions default | `/api/environment/sessions*` | — | Opera Force Dark docs | session DTO test | COMPLETE |
| Replay Lab | Latest / recovery / frames-missing | Latest replay / recovery | `/api/replays` | — | responsive board | recovery UI | COMPLETE |
| Qualification | 21W/27D/0L, 43.8%, 100% | 21W/27D/0L, 43.8%, 100% | phase_9q ABC + gates | — | — | integrity DTO | COMPLETE |
| Training | No forced BLOCKED in API; DEMO smoke gated | Campaign evidence browsable | `/api/training` | arm telemetry thin | — | blocked=null | COMPLETE |
| Experiments | Manifest kind labels | kind-aware parsers | `/api/experiments` | infra WDL still empty struct | — | registry | HONEST_EMPTY |
| Models | Registry rows; no competitive claim | latency/checkpoint/lineage | `/api/models` | WDL placeholder zeros in type | — | registry | HONEST_EMPTY |
| Population | NaN weights; hatched nulls | empirical cells | payoff + pfsp | — | — | antifallback | COMPLETE |
| Explainability | Mapped list; empty CF honest | mapped fields + fit board | explainability_report | thin fields | responsive board | layout | HONEST_EMPTY |
| Champion | Qual-backed WDL/discovery | from API/checklist | `/api/champion` + qual | — | — | no invent | COMPLETE |
| Submission | Capability-disabled demo actions | capability-disabled + real package | `/api/submission` | — | — | badge only if DEMO | COMPLETE |
| Competition | Portal snapshot mapping | portal observations | `/api/competition` | — | — | provenance | COMPLETE |
| Repository | PASS/FAIL/NOT_RUN… + Git commits | Git + records explicit statuses | `/api/repository` | — | — | DTO honesty | COMPLETE |
| Documentation | 25 Markdown sections + glossary + search | 24+ Markdown sections + glossary | `/api/documentation` | — | — | no placeholder | COMPLETE |
| Shell / browsers | color-scheme dark; Opera docs | color-scheme + Force Dark docs | fonts/theme | Opera visual pending operator | BROWSER_VARIANCE | Chrome vs Opera | BROWSER_VARIANCE |

Integrity gate cannot PASS while Arena browser smoke evidence is missing or operator visual approval is pending for Chrome/Opera.
