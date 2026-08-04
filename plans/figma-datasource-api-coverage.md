# Figma DataSource → production API coverage

Source of truth: golden `DataSource` in `dashboard/frontend/src/services/dataSource.ts`.
Production adapter: `dashboard/frontend/src/services/apiDataSource.ts` (`export class ApiDataSource implements DataSource`).

Allowed support states:

- `IMPLEMENTED` — mapped from a real `/api/*` response (no invented competitive metrics)
- `INTENTIONALLY_EMPTY` — method returns empty collection / null / zero WDL because evidence is absent
- `CAPABILITY_DISABLED` — throws `CapabilityDisabledError` (demo mutations / unsafe actions)
- `NOT_APPLICABLE` — no meaningful production mapping; empty or stub with provenance note

Missing-data rule: production must never silently return plausible **mock** fixture values.

| method | page consumers | production API | adapter | provenance | support state | demo implementation | missing-data behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `getOverviewStats` | shell / legacy | `GET /api/overview` | branch, engine sha, phase, baseline | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | Mock fixtures | hardware `NOT RECORDED` |
| `listMatches` | Arena | none (no frame match list) | `[]` | — | INTENTIONALLY_EMPTY | in-memory map | empty list |
| `getMatchById` | Arena | none | `null` | — | INTENTIONALLY_EMPTY | in-memory map | null |
| `createDemoMatch` | Arena | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast via page catch |
| `appendMatchFrame` | Arena | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast via page catch |
| `completeDemoMatch` | Arena | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast via page catch |
| `listReplays` | Replay Lab | `GET /api/replays` | id/name stubs, empty frames | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixture replay | empty frames labelled |
| `getReplayById` | Replay Lab | `GET /api/replays/{id}` | metadata stub, empty frames | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixture frames | 404 → null; no board → empty frames |
| `createReplayFromMatch` | Arena | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast via page catch |
| `listCandidates` | Qualification | `GET /api/qualification` | champion_until_promoted + gate board → Phase9Q | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixture candidates | WDL zeros; rates 0 |
| `getCandidateById` | Qualification | via list | filter | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | null if absent |
| `getQualSummary` | Qualification | `GET /api/qualification` | counts from candidates + gates | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | WDL/rates empty |
| `getTrainingBlockedState` | Training | `GET /api/training` | blocked when `launch_enabled=false` | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixture blocked | null only if launch enabled |
| `listTrainingRuns` | Training | `GET /api/training` | campaigns + chart telemetry points | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty if none |
| `getTrainingRunById` | Training | via list | filter | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | null if absent |
| `startDemoTrainingRun` | Training | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast via page catch |
| `appendTrainingMetric` | Training | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast via page catch |
| `listModels` | Models | `GET /api/models` | architecture/lifecycle/role/delivery | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | WDL zeros (API has no WDL) |
| `getModelById` | Models | via list | filter | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | null if absent |
| `getOverview` | Overview | `GET /api/overview` | candidate, jobs, funnel from readiness/package | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | `wdlHistory=[]`, zero WDL, rates 0 |
| `getApplicationStatus` | TopStatusBar | `GET /api/overview` | branch/engine/candidate/phase | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | hardware NOT RECORDED |
| `listExperiments` | Experiments | `GET /api/experiments` | manifest metadata rows | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty WDL; filter client-side |
| `getExperimentById` | Experiments | `GET /api/experiments/{id}` | same mapper | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | 404 → null |
| `compareExperiments` | Experiments | via get-by-id | gather records | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | missing ids omitted |
| `compareModels` | Models | via list | filter ids | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty if none match |
| `getModelLineage` | Models | via get-by-id | single node, no edges | IMPORTED_PROJECT_EVIDENCE | INTENTIONALLY_EMPTY | fixtures | empty edges |
| `getPopulationSummary` | Population | `GET /api/population` | labels + matrix | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty agents if no payoff |
| `getPayoffMatrix` | Population | `GET /api/population` | labels/matrix | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty matrix |
| `listExplanations` | Explainability | `GET /api/explainability` | recorded explanation blobs | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty list + note |
| `getExplanationById` | Explainability | via list | filter | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | null if absent |
| `getCounterfactuals` | Explainability | none | `[]` | — | INTENTIONALLY_EMPTY | fixtures | empty list |
| `getChampionWorkspace` | Champion | `GET /api/champion` | package + checklist | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | learned champion null |
| `getPromotionChecklist` | Champion | `GET /api/champion` | promotion_checklist rows | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | blocked reasons from API |
| `getSubmissionPipeline` | Submission | `GET /api/submission` | package + portal + upload flags | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | upload stage blocked |
| `listPackages` | Submission | via pipeline | active package or `[]` | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty list |
| `runDemoPackageAction` | Submission | — | throws | — | CAPABILITY_DISABLED | Mock demo | toast error |
| `listPortalObservations` | Competition | `GET /api/competition` | active_submission | OFFICIAL_PORTAL_OBSERVATION | IMPLEMENTED | fixtures | empty if none |
| `listManualSubmissionRecords` | Competition | `GET /api/competition` | profile_snapshot | MANUALLY_RECORDED | IMPLEMENTED | fixtures | empty if none |
| `getRepositoryStatus` | Repository | `GET /api/repository` | branch/engine | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | CI statuses skipped |
| `listRecentCommits` | Repository | `GET /api/repository` | oneline commits | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | author NOT RECORDED |
| `listCiRuns` | Repository | none | `[]` | — | INTENTIONALLY_EMPTY | fixtures | empty list |
| `getEnvironmentLocks` | Repository | none | `[]` | — | INTENTIONALLY_EMPTY | fixtures | empty list |
| `getDocumentationIndex` | Documentation | `GET /api/documentation` | section titles | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | empty if none |
| `getDocumentationSection` | Documentation | via index | stub body noting API limit | IMPORTED_PROJECT_EVIDENCE | IMPLEMENTED | fixtures | null if id missing; body explains gap |

## Demo bundling

- Default mode: **API** (`ApiDataSource`).
- Demo mode only when `VITE_DASHBOARD_DATA_MODE=demo`.
- `MockDataSource` is loaded via dynamic `import()` so the fixture graph is not required for the normal API bundle entry.

## Gates

Current vs historical upload observations remain backend-owned (`gate_status.current` / `historical_observations`). Frontend adapters must not re-merge upload-time FAIL into current PASS chips.
