import { DocSection } from "../../types/documentation";
import { SCHEMA_VERSION } from "../../types/common";

export const docSections: DocSection[] = [
  {
    id: "doc-overview",
    title: "Overview",
    order: 1,
    content: `# QuantSilico Generals Research Console

A research-grade console for analysing AI agents playing Generals.io. The console provides a typed data boundary between your training infrastructure and the frontend.

## Current Status
- **Candidate:** \`heuristic_v2f_plus_planner_terminal_fix\`
- **Development result:** 21W / 27D / 0L
- **Discovery rate:** 0.438 (gate FAILED — below threshold)
- **PPO:** NOT STARTED
- **Branch:** \`feature/full-research-platform-v0\`
- **Engine SHA:** \`9e3b9d13cca51caa1bb07db48bb85c9e90ce0462\``,
    tags: ["intro", "status"],
  },
  {
    id: "doc-data-source",
    title: "Data Source Adapter",
    order: 2,
    content: `# Data Source Adapter

All data flows through the \`DataSource\` interface. The Mock adapter serves fixture data for development and demonstration. A real adapter can be wired to a FastAPI backend.

\`\`\`ts
const ds = useDataSource();
const overview = await ds.getOverview();
const status = await ds.getApplicationStatus();
\`\`\`

> **Note:** These commands describe the eventual integrated QuantSilico project. They may not exist in the standalone Figma Make frontend export.

## Provenance
Every record carries a \`kind\` field indicating its data provenance. See the Provenance System section.`,
    tags: ["architecture", "data"],
  },
  {
    id: "doc-provenance",
    title: "Provenance System",
    order: 3,
    content: `# Provenance System

Every record carries a \`kind\` field:

| Kind | Badge | Meaning |
|---|---|---|
| \`IMPORTED_PROJECT_EVIDENCE\` | Green | Real research data — immutable |
| \`OFFICIAL_PORTAL_OBSERVATION\` | Teal | Copied from the competition portal |
| \`MANUALLY_RECORDED\` | Grey | Operator-entered record |
| \`DEMO\` | Amber | Synthetic session data — mutable |

## Immutability
Imported evidence (\`IMPORTED_PROJECT_EVIDENCE\` and \`OFFICIAL_PORTAL_OBSERVATION\`) cannot be mutated through the MockDataSource boundary. Mutating methods reject non-DEMO IDs with an error.`,
    tags: ["architecture", "provenance"],
  },
  {
    id: "doc-arena",
    title: "Arena",
    order: 4,
    content: `# Arena

The Arena page simulates live matches between configured agents. Demo matches produce \`DEMO\`-kind records that flow through to the Replay Lab.

## Usage
1. Configure player slots (player1 and player2 agent types)
2. Set map size (16–22)
3. Click **Start Match**
4. Watch the board animate in real time
5. On completion, navigate automatically to Replay Lab

## Agent slots
- **Heuristic** — the current research candidate
- **Legal Random** — random-action baseline
- **Manual** — click-to-move board control`,
    tags: ["usage", "arena"],
  },
  {
    id: "doc-environment",
    title: "Environment Lab",
    order: 5,
    content: `# Environment Lab

Manually step through a game board to inspect action legality, army propagation, and fog-of-war rules.

## Controls
- Click a source cell, then a destination cell to move armies
- Use the step counter to advance turns
- Observe fog-of-war mask update per turn
- Player labels: Heuristic / Legal Random / Manual`,
    tags: ["usage", "environment"],
  },
  {
    id: "doc-replay",
    title: "Replay Lab",
    order: 6,
    content: `# Replay Lab

Scrub through recorded matches to inspect board state at each turn.

## Overlays
- **Attribution** — saliency intensity per cell (gradient method)
- **Belief** — agent posterior probability per cell
- **Risk region** — highlights high-risk zones
- **Path** — shows planned movement path

## Controls
- Play / Pause / Step (← →)
- Speed selector (0.5×, 1×, 2×, 4×)
- Turn scrubber
- Event markers on the timeline`,
    tags: ["usage", "replay"],
  },
  {
    id: "doc-qualification",
    title: "Qualification",
    order: 7,
    content: `# Qualification (Phase 9Q)

Phase 9Q is the evaluation pipeline for promotion candidates.

## Steps
1. Screening
2. Development evaluation
3. Holdout evaluation
4. Package build
5. Linux parity
6. Upload ready
7. Portal submission

## Current candidate
\`heuristic_v2f_plus_planner_terminal_fix\`
- Development result: 21W / 27D / 0L
- Discovery rate: 0.438 — **gate FAILED**
- Conversion: 1.0 (post-discovery)
- PPO: NOT STARTED

## Historical
CNN-v3-Expander: 11W / 37D / 0L — historical, exact timestamp not recorded.`,
    tags: ["usage", "qualification"],
  },
  {
    id: "doc-training",
    title: "Training",
    order: 8,
    content: `# Training

The Training cockpit displays the PPO training loop. Two distinct panels:

**PROJECT STATUS (IMPORTED)**
PPO NOT STARTED. Discovery gate is the current blocker.

**DEMO TRAINING SESSION (DEMO)**
Synthetic frontend telemetry — animated policy loss, GPU utilisation, and checkpoint progress. All charts in the Optimisation, Hardware, Performance and Auxiliary Heads tabs carry the DEMO badge.

## Presets
- **SMOKE** — 2K-step demo run, runnable
- **DEVELOPMENT / INITIAL / OVERNIGHT / MARATHON** — disabled until discovery gate passes`,
    tags: ["usage", "training"],
  },
  {
    id: "doc-experiments",
    title: "Experiments",
    order: 9,
    content: `# Experiments

Filter, sort, and compare evaluation experiments.

## Comparison deep links
\`\`\`
/experiments/exp-current?compare=exp-current,exp-historical
\`\`\`

## Filtering
Query parameters support: \`compare\`, \`candidate\`, \`opponent\`, \`suite\`, \`lifecycle\`

## Missing values
Values not recorded in imported evidence render as \`NOT MEASURED\` or \`PARTIAL RECORD\` — never as zero.

## Evidence
- **Current:** heuristic_v2f — 21W/27D/0L — IMPORTED
- **Historical:** CNN-v3-Expander — 11W/37D/0L — IMPORTED (timestamp not recorded)`,
    tags: ["usage", "experiments"],
  },
  {
    id: "doc-models",
    title: "Models",
    order: 10,
    content: `# Models

The model registry tracks all known architectures and their lifecycle status.

## Architectures
- \`heuristic\` — rule-based, no learned weights (amber)
- \`mlp_control\` — MLP control baseline (cyan)
- \`recurrent_cnn\` — recurrent convolutional network (neutral)
- \`recurrent_graph_belief\` — GNN with belief propagation (blue)
- \`graph_belief_pyg_research\` — PyG research graph network (blue)

## Lifecycle
SCAFFOLDED → SMOKE_TESTED → TRAINED → EVALUATED → REJECTED / REJECTED_INCOMPATIBLE

## Comparison deep links
\`\`\`
/models/model-graph?compare=model-graph,model-cnn
\`\`\``,
    tags: ["usage", "models"],
  },
  {
    id: "doc-population",
    title: "Population",
    order: 11,
    content: `# Population

PFSP (Prioritised Fictitious Self-Play) population management.

## Payoff matrix
Win rates shown as a heatmap. Colour scale:
- Red (< 0.4) — losing matchup
- Grey (≈ 0.5) — neutral
- Green (> 0.6) — winning matchup

Null cells render as **MISSING** with hatched pattern — never as zero.

## Keyboard navigation
Arrow keys navigate the matrix. Press Enter to inspect a cell.`,
    tags: ["usage", "population"],
  },
  {
    id: "doc-submission",
    title: "Submission",
    order: 12,
    content: `# Submission

**Uploads are manual by design. Credentials never enter this application.**

## Pipeline stages
1. Candidate Selected
2. Package Built
3. Windows Validated
4. Linux Validated
5. Upload Ready — manual upload via portal web UI
6. Manually Submitted
7. Portal Accepted
8. Qualified

## Demo actions
Simulate build, simulate validation, and reveal the manual upload path — all DEMO operations with no real file system access.`,
    tags: ["usage", "submission"],
  },
  {
    id: "doc-integration",
    title: "Integration Notes",
    order: 13,
    content: `# Integration Notes

To connect a real backend, implement the \`DataSource\` interface in a new adapter file.

\`\`\`ts
// src/services/apiDataSource.ts
class ApiDataSource implements DataSource {
  async getOverview(): Promise<OverviewRecord> {
    const res = await fetch("/api/overview");
    const data = await res.json();
    if (data.schemaVersion !== SCHEMA_VERSION) {
      throw new Error("Schema version mismatch");
    }
    return data;
  }
  // ... implement remaining methods
}
\`\`\`

\`\`\`python
# FastAPI route hint
@app.get("/api/overview")
def get_overview() -> OverviewRecord:
    ...
\`\`\`

> **Note:** These commands describe the eventual integrated QuantSilico project. They may not exist in the standalone Figma Make frontend export.

## Schema versioning
All records include a \`schemaVersion\` field. Validate against \`SCHEMA_VERSION = "2.0.0"\` on import.`,
    tags: ["integration", "api"],
  },
];

export const docIndex = {
  sections: docSections.map(({ id, title, order, tags }) => ({ id, title, order, tags })),
  schemaVersion: SCHEMA_VERSION,
};
