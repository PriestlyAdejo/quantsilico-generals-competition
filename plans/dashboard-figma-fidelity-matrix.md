# Dashboard Figma fidelity matrix

Corrective branch: `fix/dashboard-figma-fidelity`  
Base commit: `b030abc`  
Primary target: `feature/full-research-platform-v0`

## Golden source integrity

| Field | Value |
|-------|--------|
| Immutable source | `C:\Users\pries\Downloads\generals dashboard.zip` |
| ZIP SHA-256 | `C7898A5EC420D9C43E5398ED3014D6A09490D73D32568536D21312CFB39413B3` |
| ZIP bytes | `243884` |
| Extraction path | `var/imports/figma-console-golden/` (gitignored; re-extracted from ZIP) |
| Extracted file count | `158` |
| Tree content hash (path\|size) | `34967ab385522277a960a81bd144b68bb1b943f37fffc16abbf2429d27939f0f` |

Production code must never import from the golden tree. Golden is visual/structural reference only.

## Dependency discipline

- Keep: React 19, react-router-dom, Vite, Vitest, Testing Library.
- Do **not** add Tailwind, MUI, Emotion, or the full shadcn dump.
- Prefer SVG/`MetricChart` and CSS heatmaps over Recharts unless a documented exception is required after build is green.
- Add packages only when imported by final production files; document why, imports, bundle impact, React 19 compatibility.

## Status legend

- `MISSING` — major Figma regions absent
- `PARTIAL` — some structure, incomplete binding or interaction
- `FUNCTIONALLY PRESENT / VISUALLY WRONG` — API wired, layout not Figma-faithful
- `COMPLETE` — all concrete acceptance conditions below satisfied (not merely “renders”)

---

## Overview

| Field | Value |
|-------|--------|
| Golden | `var/imports/figma-console-golden/src/pages/OverviewPage.tsx` |
| Production | `dashboard/frontend/src/pages/OverviewPage.tsx` |
| Backend | `GET /api/overview` (`gate_status.current` only for live chips) |
| Status | PARTIAL — structured rebuild landed; visual screenshot gate pending operator inspect |

**COMPLETE only when:** page header + provenance; blocker/status strip; ≥4 metric cards (candidate, readiness, CNN latency, graph latency, promotion); gate funnel using **current** gates; jobs strip; package summary cards; W/D/L or empty composed state; raw only via RawRecordDrawer; no primary `<pre>` JSON; desktop screenshot density matches golden hierarchy; mobile stacks intentionally.

---

## Arena

| Field | Value |
|-------|--------|
| Golden | `.../ArenaPage.tsx` |
| Production | `dashboard/frontend/src/pages/ArenaPage.tsx` |
| Backend | `/api/capabilities`, `/api/jobs/match`, `/api/jobs/{id}` |
| Status | FUNCTIONALLY PRESENT / VISUALLY WRONG → COMPLETE |

**COMPLETE only when:** left config inspector; **dominant** central GeneralsBoard; right telemetry inspector; launch → job ID → queued/running/completed visible; production mode never animates synthetic armies; if only final results: waiting board + “live board telemetry not emitted”; DEMO opt-in only; completed match links to replay when available; mobile stacks regions.

---

## Environment Lab

| Field | Value |
|-------|--------|
| Golden | `.../EnvironmentLabPage.tsx` |
| Production | `MorePages.tsx` EnvironmentLabPage |
| Backend | `/api/environment`, `/api/capabilities`, `/api/replays` |
| Status | MISSING → COMPLETE |

**COMPLETE only when:** OFFICIAL / DEMO mode selector; official: real board/replay inspect + disabled Step/Reset with capability reasons; DEMO: interactive step/reset with persistent DEMO banner, `DEMO` provenance, **no** API persistence into experiments/replays/models/qualification; three-column density; mobile stacks.

---

## Replay Lab

| Field | Value |
|-------|--------|
| Golden | `.../ReplayLabPage.tsx` |
| Production | ReplayLabPage in MorePages / dedicated |
| Backend | `/api/replays`, `/api/replays/{id}` |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** prominent board; play/pause/speed/scrubber; real replay selector; empty composed state; inspector tabs; raw drawer; NOT RECORDED for missing beliefs; no fabricated decisions.

---

## Qualification

| Field | Value |
|-------|--------|
| Golden | `.../QualificationPage.tsx` |
| Production | QualificationPage |
| Backend | `/api/qualification` |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** PipelineStepper; named gate cards (never bare QUALIFIED); comparison/provenance; charts or composed empty; historical vs current separated; raw drawer only.

---

## Training

| Field | Value |
|-------|--------|
| Golden | `.../TrainingPage.tsx` |
| Production | TrainingPage |
| Backend | `/api/training` |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** 7 tabs; charts with series-level provenance (manifest id, kind, missing fields); DEVELOPMENT telemetry bound; empty states for NOT RECORDED; no JSON primary; run/architecture selection.

---

## Experiments

| Field | Value |
|-------|--------|
| Golden | `.../ExperimentsPage.tsx` |
| Production | ExperimentsPage |
| Backend | `/api/experiments` (+ DEVELOPMENT arms) |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** FilterBar/search; selectable table; 2–3 way compare; metric cards; deep-link `?compare=`; raw drawer; DEVELOPMENT arms visible.

---

## Models

| Field | Value |
|-------|--------|
| Golden | `.../ModelsPage.tsx` |
| Production | ModelsPage |
| Backend | `/api/models` + latency gate |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** registry table; selection detail; latency board-size comparison chart with source id; no learned champion; promotion blocker visible; raw drawer.

---

## Population

| Field | Value |
|-------|--------|
| Golden | `.../PopulationPage.tsx` |
| Production | PopulationPage |
| Backend | `/api/population` |
| Status | FUNCTIONALLY PRESENT / VISUALLY WRONG → COMPLETE |

**COMPLETE only when:** payoff SVG heatmap with numeric cells + hatching for MISSING; PFSP bars; meta-strategy; provenance per chart; accessible text fallback; no plain manifest dump.

---

## Explainability

| Field | Value |
|-------|--------|
| Golden | `.../ExplainabilityPage.tsx` |
| Production | ExplainabilityPage |
| Backend | `/api/explainability` |
| Status | FUNCTIONALLY PRESENT / VISUALLY WRONG → COMPLETE |

**COMPLETE only when:** 7 tabs (Decision, Why cell, Why not, Next, Concepts, Counterfactuals, Faithfulness); GeneralsBoard where data allows; missing components honest; partial/failed fidelity visible; raw secondary.

---

## Champion

| Field | Value |
|-------|--------|
| Golden | `.../ChampionPage.tsx` |
| Production | ChampionPage |
| Backend | `/api/champion` |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** comparison workspace; EvidenceChecklist; NO LEARNED CHAMPION; promotion disabled with reasons; current gates; raw drawer.

---

## Submission

| Field | Value |
|-------|--------|
| Golden | `.../SubmissionPage.tsx` |
| Production | SubmissionPage |
| Backend | `/api/submission` |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** PipelineStepper lifecycle; structured package metadata cards; authoritative `027ff5d` vs stale `ee06778`; no upload control; historical upload gates labelled; raw drawer.

---

## Competition

| Field | Value |
|-------|--------|
| Golden | `.../CompetitionPage.tsx` |
| Production | CompetitionPage |
| Backend | `/api/competition` |
| Status | FUNCTIONALLY PRESENT / VISUALLY WRONG → COMPLETE |

**COMPLETE only when:** summary cards; SourceSelector; non-live profile snapshot; attribution warning; version table; raw in drawer — **not** primary JSON.

---

## Repository

| Field | Value |
|-------|--------|
| Golden | `.../RepositoryPage.tsx` |
| Production | RepositoryPage |
| Backend | `/api/repository`, `/api/build-info` |
| Status | FUNCTIONALLY PRESENT / VISUALLY WRONG → COMPLETE |

**COMPLETE only when:** branch/commit/dirty/engine/env cards; recent commits; frontend vs backend build identity; read-only; raw drawer only.

---

## Documentation

| Field | Value |
|-------|--------|
| Golden | `.../DocumentationPage.tsx` |
| Production | DocumentationPage |
| Backend | `/api/documentation` + static sections |
| Status | PARTIAL → COMPLETE |

**COMPLETE only when:** section nav + search; full operator guide sections (startup→troubleshooting); copyable commands; not four sparse panels.

---

## Empty-space rule

Fail unnecessary empty space only when real content or designed controls were omitted. Legitimate no-data pages must still provide composed empty state, provenance, context, and next action.

## Screenshot review

Store under `artifacts/ui-review/dashboard-fidelity/` (ignored). Prefer existing tooling; add Playwright only after build green if needed.
