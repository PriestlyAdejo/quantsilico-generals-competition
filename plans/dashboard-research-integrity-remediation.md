# Dashboard research-integrity remediation matrix

Working branch: `fix/protocol-and-dashboard-research-integrity` from `c671bab`.

Blocking scopes: `RESEARCH_EVIDENCE` | `DASHBOARD_OPERABILITY` | `DELIVERY` | `UX_NONBLOCKING` | `OPTIONAL_POLISH`

| Route | Visible defect | Touch points | Gate scope | Status |
| --- | --- | --- | --- | --- |
| Arena / Replay | Metadata-only replays; no board frames | cli/main.py, jobs.py, apiDataSource, ReplayLabPage | RESEARCH_EVIDENCE (diagnostics); OPERABILITY (nav) | PARTIAL — jobs METADATA_ONLY; Stage1 diagnostic frames+actions path added |
| Env Lab | Unclear workflow; session limit stuck | EnvironmentLabPage, env_sessions.py | DASHBOARD_OPERABILITY | PENDING |
| Qualification | Suite selector does not swap detail/charts | QualificationPage.tsx | RESEARCH_EVIDENCE | FIXED — suite maps to stage + filters WDL chart |
| Experiments | 0W/0D/0L for non-eval manifests | apiDataSource mapExperiment | RESEARCH_EVIDENCE | FIXED — null + MISSING when absent |
| Models | Zero params / false WDL | apiDataSource mapModel | RESEARCH_EVIDENCE | FIXED — null + MISSING when absent |
| Population | Fabricated 0.5 cells / PFSP fill | population.py, PopulationPage | RESEARCH_EVIDENCE | FIXED — empirical-only; no 0.5 fill |
| Explainability | Empty without prerequisites | ExplainabilityPage, api | DASHBOARD_OPERABILITY | PENDING |
| Submission | DEMO controls in API mode; weak package registry | SubmissionPage, evidence readers | DELIVERY | PENDING |
| Competition | Portal attribution clarity | CompetitionPage, portal manifests | DELIVERY (+ attribution already RESOLVED) | PENDING |
| Provenance | Internal kind IMPORTED_PROJECT_EVIDENCE | DataSourceBadge | DASHBOARD_OPERABILITY | PENDING |
| Overview/Champion | Conflated statuses | OverviewPage, ChampionPage | DASHBOARD_OPERABILITY | PENDING |
| Training | Observer present; polish charts | TrainingPage | DASHBOARD_OPERABILITY | PENDING |
| Learned protocol | CheckpointPolicy dict-as-tuple hypothesis | adaptive_initial.py, models | RESEARCH_EVIDENCE | FIXED — Class A; typed contract; ladder PASS |

Statuses update as work lands. Protocol forensics is not blocked by OPERABILITY/DELIVERY.
