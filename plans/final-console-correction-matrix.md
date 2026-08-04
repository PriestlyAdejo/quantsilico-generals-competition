# Final console correction matrix

Working branch: `fix/console-and-phase9-phase10` from `3c80b5c`.

| Area | Defect | Touch points | Test | Status |
| --- | --- | --- | --- | --- |
| Env Lab board size | Board dominates central column | GeneralsBoard size variants; EnvironmentLabPage caps | overflow matrix | DONE |
| Session concurrent UX | Internal limit jargon; stale sessions | env_sessions lease/list/close; Env Lab UI | session lifecycle | DONE |
| Provenance labels | IMPORTED PROJECT EVIDENCE opaque | DataSourceBadge display map | badge copy | DONE |
| Qualification jargon | PHASE_9Q primary; raw predicates | QualificationPage | qual metrics | DONE |
| Documentation | Plain pre Markdown | MDX pipeline + DocMarkdown + prose CSS | render + search | DONE |
| Date/time | Date without time; Invalid Date | DateTimeCell + formatting | datetime tests | DONE |
| Population | Invalid dates; antifallback | PopulationPage + apiDataSource | antifallback | DONE |
| Explainability | Random DEMO concepts; synthetic boards | ExplainabilityPage | tab empty states | DONE |
| Display names | Truncated IDs | displayNames helper | selectors | DONE |
| Arena smoke | Re-verify after changes | arena_browser_smoke | DONE |

Statuses update as work lands.
