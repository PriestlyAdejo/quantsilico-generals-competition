import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { GeneralsBoard, type BoardFrame } from "../components/board/GeneralsBoard";
import {
  ChartCard,
  DataTable,
  EvidenceChecklist,
  FilterBar,
  MetricCard,
  MetricStrip,
  PageHeader,
  Panel,
  PipelineStepper,
} from "../components/data-display/Panel";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, EmptyState, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

function demoBoard(turn: number): BoardFrame {
  const n = 12;
  const typeGrid = Array.from({ length: n }, (_, r) =>
    Array.from({ length: n }, (_, c) => ((r * c + turn) % 9 === 0 ? 0 : 1)),
  );
  const ownerGrid = Array.from({ length: n }, (_, r) =>
    Array.from({ length: n }, (_, c) => (r < 3 && c < 3 ? 1 : r > 8 && c > 8 ? 2 : 0)),
  );
  const armyGrid = ownerGrid.map((row, r) =>
    row.map((o, c) => (o ? 3 + ((r + c + turn) % 7) : 0)),
  );
  return { mapKey: `demo-${turn}`, height: n, width: n, typeGrid, ownerGrid, armyGrid };
}

export default function QualificationPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [overview, setOverview] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/qualification", ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setDown(true);
      });
    ds.getOverview(ac.signal)
      .then((o) => setOverview(o as unknown as Record<string, unknown>))
      .catch(() => undefined);
    return () => ac.abort();
  }, [ds]);

  if (down) return <BackendUnavailable />;
  if (!data) return <LoadingState />;

  const gates = (data.gates || {}) as Record<string, string>;
  const board = (overview?.gate_board || gates) as Record<string, string>;
  const steps = [
    { id: "dev", label: "Heuristic development", state: board.HEURISTIC_DEVELOPMENT_GATE === "PASS" ? "done" : "failed" },
    { id: "pre", label: "Pre-PPO submission", state: board.PRE_PPO_SUBMISSION_GATE === "PASS" ? "done" : "pending" },
    { id: "portal", label: "Portal submission gate", state: board.PORTAL_SUBMISSION_GATE === "PASS" ? "done" : "pending" },
    { id: "learn", label: "Learning readiness", state: board.LEARNING_READINESS_GATE === "PASS" ? "done" : "pending" },
    { id: "promo", label: "Learned promotion", state: board.LEARNED_PROMOTION_GATE === "PASS" ? "done" : "blocked" },
  ] as const;

  return (
    <div>
      <PageHeader
        eyebrow="$ QUALIFICATION /"
        title="Qualification"
        subtitle="Named gates only — never unqualified QUALIFIED."
      />
      <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>
        View raw record
      </button>
      <div className="grid-2">
        <Panel title="Pipeline">
          <PipelineStepper steps={steps.map((s) => ({ ...s, state: s.state as "done" | "failed" | "pending" | "blocked" }))} />
        </Panel>
        <Panel title="Gate cards">
          <div className="stack">
            {Object.entries(board).map(([name, value]) => (
              <div key={name} className="row between">
                <code>{name}</code>
                <StatusBadge value={String(value)} />
              </div>
            ))}
          </div>
          <p className="muted">Champion until promoted: {String(data.champion_until_promoted)}</p>
        </Panel>
      </div>
      <RawRecordDrawer
        open={rawOpen}
        onClose={() => setRawOpen(false)}
        title="Qualification raw"
        recordId="qualification"
        schema="QUALIFICATION_DASHBOARD"
        provenance="LIVE_REPOSITORY"
        raw={data}
      />
    </div>
  );
}

export function PopulationPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [rawOpen, setRawOpen] = useState(false);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/population", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  const recorded = String(data.state).includes("RECORDED") && !String(data.state).includes("NOT YET");
  const payoff = (data.payoff_matrix || {}) as {
    labels?: string[];
    matrix?: (number | null)[][];
    cells?: { status?: string; score_rate?: number | null; policy_a?: string; policy_b?: string }[];
  };
  const labels = payoff.labels || (data.population as string[]) || [];
  const matrix = payoff.matrix || [];
  const pfsp = (data.pfsp || {}) as { opponents?: string[]; probabilities?: number[]; focal?: string };
  const cell = 36;
  return (
    <div>
      <PageHeader eyebrow="$ POPULATION /" title="Population" subtitle="Empirical payoff / PFSP — not a promotion claim." />
      {!recorded ? (
        <EmptyState title={String(data.state)} detail={String(data.note || "")} />
      ) : (
        <>
          <div className="row gap">
            <MetricCard label="Games" value={String(data.games_total ?? "—")} />
            <MetricCard label="Synthetic" value={String(data.synthetic)} />
            <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>
              View raw record
            </button>
          </div>
          <div className="grid-2">
            <ChartCard
              title="Payoff heatmap"
              provenance={{
                manifestId: "payoff_pfsp_development.json",
                kind: "EMPIRICAL",
                missingFields: [],
              }}
            >
              {labels.length && matrix.length ? (
                <svg className="heatmap" viewBox={`0 0 ${(labels.length + 1) * cell} ${(labels.length + 1) * cell}`}>
                  {labels.map((lab, i) => (
                    <text key={`r-${lab}`} className="axis" x={4} y={(i + 1.65) * cell}>
                      {lab.slice(0, 10)}
                    </text>
                  ))}
                  {labels.map((lab, j) => (
                    <text key={`c-${lab}`} className="axis" x={(j + 1.2) * cell} y={14}>
                      {lab.slice(0, 6)}
                    </text>
                  ))}
                  {matrix.map((row, i) =>
                    row.map((v, j) => {
                      if (i === j) return null;
                      const missing = v == null;
                      const fill = missing
                        ? "url(#hatch)"
                        : `rgb(${Math.round(80 + (1 - Number(v)) * 140)},${Math.round(40 + Number(v) * 140)},70)`;
                      return (
                        <g key={`${i}-${j}`}>
                          <rect x={(j + 1) * cell} y={(i + 1) * cell} width={cell - 2} height={cell - 2} fill={fill} stroke="#1e2630" />
                          <text x={(j + 1.5) * cell} y={(i + 1.6) * cell} textAnchor="middle">
                            {missing ? "—" : Number(v).toFixed(2)}
                          </text>
                        </g>
                      );
                    }),
                  )}
                  <defs>
                    <pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6">
                      <path d="M0,6 L6,0" stroke="#6f7c89" strokeWidth="1" />
                    </pattern>
                  </defs>
                </svg>
              ) : (
                <EmptyState title="Matrix unavailable" detail="Payoff labels/matrix missing." />
              )}
              <p className="muted">Hatched / — cells are MISSING pairings, not zero wins.</p>
            </ChartCard>
            <ChartCard
              title="PFSP weights"
              provenance={{
                manifestId: "pfsp_empirical.json",
                kind: "EMPIRICAL_PFSP",
                candidate: pfsp.focal || undefined,
              }}
            >
              {(pfsp.opponents || []).map((opp, i) => {
                const p = pfsp.probabilities?.[i] ?? 0;
                return (
                  <div key={opp} className="row between" style={{ marginBottom: "0.35rem" }}>
                    <span className="mono">{opp}</span>
                    <div style={{ flex: 1, margin: "0 0.65rem", height: 8, background: "#1e2630" }}>
                      <div style={{ width: `${p * 100}%`, height: "100%", background: "#ffb000" }} />
                    </div>
                    <span className="mono">{(p * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </ChartCard>
          </div>
        </>
      )}
      <RawRecordDrawer
        open={rawOpen}
        onClose={() => setRawOpen(false)}
        title="Population raw"
        recordId="population"
        schema="POPULATION"
        provenance="EMPIRICAL"
        raw={data}
      />
    </div>
  );
}

const EXPLAIN_TABS = [
  "Decision",
  "Why this cell?",
  "Why not?",
  "What happens next?",
  "Concepts",
  "Counterfactuals",
  "Faithfulness",
] as const;

export function ExplainabilityPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [tab, setTab] = useState<(typeof EXPLAIN_TABS)[number]>("Decision");
  const [rawOpen, setRawOpen] = useState(false);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/explainability", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  const present = String(data.state).includes("PRESENT");
  const explanations = (data.explanations as Record<string, unknown>[]) || [];
  const learned = explanations.find((e) => e.fidelity) || explanations[0] || {};
  const fidelity = (learned.fidelity || {}) as Record<string, string>;
  return (
    <div>
      <PageHeader
        eyebrow="$ EXPLAINABILITY /"
        title="Explainability"
        subtitle="Frozen-checkpoint smoke — LEARNED_PROMOTION stays NONE."
      />
      <div className="tabs" role="tablist">
        {EXPLAIN_TABS.map((t) => (
          <button key={t} type="button" role="tab" className={tab === t ? "active" : undefined} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>
      {!present ? (
        <EmptyState title={String(data.state)} detail={String(data.note || "")} />
      ) : (
        <Panel
          title={tab}
          actions={
            <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>
              Raw
            </button>
          }
        >
          {tab === "Decision" ? (
            <dl className="kv">
              <dt>Chosen action</dt>
              <dd>{String(learned.chosen_action_index ?? "NOT RECORDED")}</dd>
              <dt>Alternative</dt>
              <dd>{String(learned.alternative_action_index ?? "NOT RECORDED")}</dd>
              <dt>Margin</dt>
              <dd>{String(learned.chosen_vs_alternative_margin ?? "NOT RECORDED")}</dd>
            </dl>
          ) : null}
          {tab === "Faithfulness" ? (
            <div className="stack">
              {Object.entries(fidelity).map(([k, v]) => (
                <div key={k} className="row between">
                  <span>{k}</span>
                  <StatusBadge value={String(v)} />
                </div>
              ))}
            </div>
          ) : null}
          {tab === "Counterfactuals" ? (
            <p className="muted">{JSON.stringify(learned.counterfactual || { status: "NOT RECORDED" })}</p>
          ) : null}
          {tab !== "Decision" && tab !== "Faithfulness" && tab !== "Counterfactuals" ? (
            <EmptyState
              title="NOT RECORDED for this tab"
              detail="Frozen explainability smoke did not emit this view. Layout retained for Figma parity."
            />
          ) : null}
          <p className="muted">LEARNED_PROMOTION_GATE: {String(data.learned_promotion_gate || "NONE")}</p>
        </Panel>
      )}
      <RawRecordDrawer
        open={rawOpen}
        onClose={() => setRawOpen(false)}
        title="Explainability raw"
        recordId="explainability"
        schema="EXPLANATION"
        provenance="FROZEN_CHECKPOINT"
        raw={data}
      />
    </div>
  );
}

export function ChampionPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [rawOpen, setRawOpen] = useState(false);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/champion", ac.signal).then(setData).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  const checklist = (data.promotion_checklist || {}) as Record<string, unknown>;
  const items = Object.entries(checklist).map(([k, v]) => ({
    id: k,
    label: k,
    status: typeof v === "boolean" ? (v ? "PASS" : "FAIL") : String(v),
    detail: typeof v === "object" ? JSON.stringify(v) : undefined,
  }));
  return (
    <div>
      <PageHeader eyebrow="$ CHAMPION /" title="Champion Workspace" subtitle="Submitted heuristic remains active." />
      <MetricStrip
        items={[
          { label: "Heuristic baseline", value: String(data.heuristic_baseline || "—"), tone: "accent" },
          { label: "Learned champion", value: String(data.learned_champion_note || "NO LEARNED CHAMPION") },
          { label: "Promotion", value: String(checklist.LEARNED_PROMOTION_GATE || "NONE"), tone: "muted" },
        ]}
      />
      <Panel title="Evidence checklist" actions={<button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>Raw</button>}>
        <EvidenceChecklist items={items} />
      </Panel>
      <button type="button" className="btn" disabled>
        Promote to champion — DISABLED
      </button>
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Champion raw" recordId="champion" schema="CHAMPION" provenance="LIVE_REPOSITORY" raw={data} />
    </div>
  );
}

export function EnvironmentLabPage() {
  const ds = useDataSource();
  const [caps, setCaps] = useState<Record<string, unknown> | null>(null);
  const [env, setEnv] = useState<Record<string, unknown> | null>(null);
  const [mode, setMode] = useState<"official" | "demo">("official");
  const [turn, setTurn] = useState(0);
  const [selected, setSelected] = useState<{ r: number; c: number } | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    Promise.all([ds.getCapabilities(ac.signal), ds.getJson("/api/environment", ac.signal)]).then(([c, e]) => {
      setCaps(c as unknown as Record<string, unknown>);
      setEnv(e);
    });
    return () => ac.abort();
  }, [ds]);
  if (!caps || !env) return <LoadingState />;
  const capabilities = (caps.capabilities || {}) as Record<string, { enabled: boolean; reason: string }>;
  const board = demoBoard(turn);
  return (
    <div>
      <PageHeader eyebrow="$ ENVIRONMENT /" title="Environment Lab" subtitle="Official inspection vs isolated DEMO adapter." />
      <div className="mode-toggle row gap">
        <button type="button" className={mode === "official" ? "btn primary" : "btn ghost"} onClick={() => setMode("official")}>
          OFFICIAL / REPLAY INSPECTION
        </button>
        <button type="button" className={mode === "demo" ? "btn primary" : "btn ghost"} onClick={() => setMode("demo")}>
          DEMO ADAPTER
        </button>
      </div>
      {mode === "demo" ? (
        <div className="banner warning">DEMO ADAPTER — Synthetic only. Never writes experiments, replays, models, or qualification evidence.</div>
      ) : null}
      <div className="workspace-3col">
        <aside className="inspector">
          <Panel title="Controls">
            {mode === "official" ? (
              <>
                <p className="muted">{capabilities.environment_reset?.reason}</p>
                <p className="muted">{capabilities.environment_step?.reason}</p>
                <button className="btn" type="button" disabled title={capabilities.environment_reset?.reason}>
                  Reset
                </button>{" "}
                <button className="btn" type="button" disabled title={capabilities.environment_step?.reason}>
                  Step
                </button>
                <EmptyState title={String(env.state)} detail={String(env.reason || "Inspect replays for official boards.")} />
              </>
            ) : (
              <>
                <button className="btn primary" type="button" onClick={() => setTurn((t) => t + 1)}>
                  Step
                </button>{" "}
                <button
                  className="btn ghost"
                  type="button"
                  onClick={() => {
                    setTurn(0);
                    setSelected(null);
                  }}
                >
                  Reset
                </button>
                <p className="muted">Local demo turn {turn}. Refresh clears state.</p>
              </>
            )}
          </Panel>
        </aside>
        <section className="board-stage">
          <GeneralsBoard frame={mode === "demo" ? board : demoBoard(0)} />
          {mode === "official" ? (
            <p className="board-caption muted">Official session board unavailable — showing placeholder until replay selection lands frames.</p>
          ) : (
            <p className="board-caption muted">Demo board · provenance DEMO · excluded from project metrics.</p>
          )}
        </section>
        <aside className="inspector">
          <Panel title="Telemetry">
            <dl className="kv">
              <dt>Mode</dt>
              <dd>{mode.toUpperCase()}</dd>
              <dt>Turn</dt>
              <dd>{mode === "demo" ? turn : "NOT RECORDED"}</dd>
              <dt>Selection</dt>
              <dd>{selected ? `${selected.r},${selected.c}` : "none"}</dd>
            </dl>
          </Panel>
        </aside>
      </div>
    </div>
  );
}

export function RepositoryPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [build, setBuild] = useState<Record<string, unknown> | null>(null);
  const [rawOpen, setRawOpen] = useState(false);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/repository", ac.signal).then(setData).catch(() => undefined);
    ds.getJson("/api/build-info", ac.signal).then(setBuild).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;
  return (
    <div>
      <PageHeader eyebrow="$ REPOSITORY /" title="Repository" subtitle="Read-only — no mutation controls." />
      <div className="metric-grid">
        <MetricCard label="Branch" value={String(data.branch)} tone="accent" />
        <MetricCard label="Commit" value={String(data.commit).slice(0, 7)} />
        <MetricCard label="Dirty" value={String(data.dirty)} />
        <MetricCard label="Engine" value={String(data.engine_commit || "—").slice(0, 7)} />
        <MetricCard label="Remote" value={String(data.remote || "origin")} />
        <MetricCard
          label="UI build"
          value={build?.frontend ? String((build.frontend as { commit?: string }).commit || "—").slice(0, 7) : "NOT BUILT"}
          sublabel={build?.mismatch ? "MISMATCH vs HEAD" : undefined}
          tone={build?.mismatch ? "failure" : "muted"}
        />
      </div>
      <Panel title="Status" actions={<button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>View raw record</button>}>
        <p className="muted">Git mutation disabled. Schema versions and environment locks appear here when recorded.</p>
        {build?.warning ? <div className="banner warning">{String(build.warning)}</div> : null}
      </Panel>
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Repository raw" recordId="repository" schema="REPOSITORY" provenance="LIVE_REPOSITORY" raw={{ repository: data, build }} />
    </div>
  );
}

const DOC_SECTIONS = [
  { id: "startup", title: "Startup", body: "scripts\\dashboard\\start.cmd / open.cmd / status.cmd / stop.cmd — API on 127.0.0.1:8765." },
  { id: "arena", title: "Arena", body: "Official evaluator jobs only. No synthetic live board animation when telemetry is absent." },
  { id: "environment", title: "Environment Lab", body: "OFFICIAL inspection vs DEMO adapter. Demo never writes evidence." },
  { id: "replay", title: "Replay Lab", body: "Real replay IDs. Missing fields render NOT RECORDED." },
  { id: "qualification", title: "Qualification", body: "Named gates: DEVELOPMENT, PRE_PPO, PORTAL, LEARNING_READINESS, LEARNED_PROMOTION." },
  { id: "training", title: "Training", body: "Smoke + DEVELOPMENT telemetry. Charts require producers; missing stays NOT RECORDED." },
  { id: "experiments", title: "Experiments", body: "Compare manifests and DEVELOPMENT arms. Do not invent metrics." },
  { id: "models", title: "Models", body: "Heuristic + MLP/CNN/graph. No learned champion." },
  { id: "population", title: "Population", body: "Empirical PFSP heatmap. Missing cells are hatched, not zero." },
  { id: "explain", title: "Explainability", body: "Seven tabs; partial faithfulness remains visible." },
  { id: "package", title: "Package validation", body: "CLI/operator builds. Dashboard does not upload." },
  { id: "upload", title: "Manual upload", body: "Credentials never enter this application." },
  { id: "portal", title: "Portal observations", body: "Non-live snapshots with provenance labels." },
  { id: "troubleshoot", title: "Troubleshooting", body: "If UI commit ≠ repo HEAD, rebuild frontend dist. Stop dashboard during latency benches." },
];

export function DocumentationPage() {
  const [q, setQ] = useState("");
  const [active, setActive] = useState("startup");
  const filtered = DOC_SECTIONS.filter(
    (s) => !q || s.title.toLowerCase().includes(q.toLowerCase()) || s.body.toLowerCase().includes(q.toLowerCase()),
  );
  const current = DOC_SECTIONS.find((s) => s.id === active) || filtered[0];
  return (
    <div>
      <PageHeader eyebrow="$ DOCUMENTATION /" title="Documentation" subtitle="Searchable operator guide." />
      <div className="grid-2">
        <Panel title="Sections">
          <FilterBar value={q} onChange={setQ} placeholder="Search sections…" />
          <ul className="stack">
            {filtered.map((s) => (
              <li key={s.id}>
                <button type="button" className={s.id === current?.id ? "btn primary" : "btn ghost"} onClick={() => setActive(s.id)}>
                  {s.title}
                </button>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel title={current?.title || "Section"}>
          <p>{current?.body}</p>
          {current?.id === "startup" ? (
            <div className="pre">{`scripts\\dashboard\\start.cmd
scripts\\dashboard\\open.cmd
scripts\\dashboard\\status.cmd
scripts\\dashboard\\stop.cmd`}</div>
          ) : null}
        </Panel>
      </div>
    </div>
  );
}

export function ExperimentsPage() {
  const ds = useDataSource();
  const [params, setParams] = useSearchParams();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [training, setTraining] = useState<Record<string, unknown> | null>(null);
  const [q, setQ] = useState("");
  const [rawOpen, setRawOpen] = useState(false);
  const compare = (params.get("compare") || "").split(",").filter(Boolean);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/experiments", ac.signal).then(setData).catch(() => undefined);
    ds.getJson("/api/training", ac.signal).then(setTraining).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  if (!data) return <LoadingState />;

  const experiments = (data.experiments as Record<string, unknown>[]) || [];
  const arms = ((training?.smoke as { bounded_development_ppo?: { arms?: Record<string, unknown> } } | undefined)?.bounded_development_ppo?.arms) || {};
  const armRows = Object.entries(arms).map(([id, arm]) => ({
    id,
    kind: "BOUNDED_DEVELOPMENT_PPO",
    path: `dev_ppo/${id}`,
    legal: String((arm as { legal_action_rate?: number }).legal_action_rate ?? "—"),
  }));
  const rows = [
    ...experiments.map((e) => ({ id: String(e.id), kind: String(e.kind), path: String(e.path), legal: "—" })),
    ...armRows,
  ].filter((r) => !q || r.id.toLowerCase().includes(q.toLowerCase()) || r.kind.toLowerCase().includes(q.toLowerCase()));

  function toggleCompare(id: string) {
    const set = new Set(compare);
    if (set.has(id)) set.delete(id);
    else if (set.size < 3) set.add(id);
    setParams(set.size ? { compare: [...set].join(",") } : {});
  }

  return (
    <div>
      <PageHeader eyebrow="$ EXPERIMENTS /" title="Experiments" subtitle="Select up to 3 rows to compare." />
      <FilterBar value={q} onChange={setQ} placeholder="Search candidates, kinds…" />
      <Panel title="Runs" actions={<button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>Raw</button>}>
        <DataTable
          columns={[
            { key: "sel", label: "" },
            { key: "id", label: "ID" },
            { key: "kind", label: "Kind" },
            { key: "legal", label: "Legal rate" },
            { key: "path", label: "Path" },
          ]}
          rows={rows.slice(0, 80).map((r) => ({
            sel: (
              <input type="checkbox" checked={compare.includes(r.id)} onChange={() => toggleCompare(r.id)} aria-label={`Select ${r.id}`} />
            ),
            id: <span className="mono">{r.id}</span>,
            kind: r.kind,
            legal: r.legal,
            path: <span className="mono">{r.path}</span>,
          }))}
        />
      </Panel>
      {compare.length ? (
        <Panel title="Comparison">
          <p className="muted">Comparing: {compare.join(" · ")}</p>
          <p className="muted">Missing metrics stay unavailable — not zero-filled.</p>
        </Panel>
      ) : null}
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Experiments raw" recordId="experiments" schema="EXPERIMENTS" provenance="LIVE_REPOSITORY" raw={{ experiments: data, trainingArms: arms }} />
    </div>
  );
}

export function ReplayLabPage() {
  const ds = useDataSource();
  const [list, setList] = useState<Record<string, unknown> | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [rawOpen, setRawOpen] = useState(false);
  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/replays", ac.signal).then(setList).catch(() => undefined);
    return () => ac.abort();
  }, [ds]);
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    const ac = new AbortController();
    ds.getJson(`/api/replays/${selected}`, ac.signal).then(setDetail).catch(() => setDetail(null));
    return () => ac.abort();
  }, [ds, selected]);

  const frame = useMemo(() => {
    const h = Number(detail?.height || 0);
    const w = Number(detail?.width || 0);
    if (!h || !w || !detail) return null;
    const typeGrid = (detail.type_grid || detail.typeGrid) as number[][] | undefined;
    if (!typeGrid) return null;
    return {
      mapKey: String(detail.id || selected),
      height: h,
      width: w,
      typeGrid,
      ownerGrid: (detail.owner_grid || detail.ownerGrid) as number[][] | undefined,
      armyGrid: (detail.army_grid || detail.armyGrid) as number[][] | undefined,
    } satisfies BoardFrame;
  }, [detail, selected]);

  if (!list) return <LoadingState />;
  const replays = (list.replays as Record<string, unknown>[]) || [];

  return (
    <div>
      <PageHeader eyebrow="$ REPLAY /" title="Replay Lab" subtitle="Real replay IDs only. Missing fields render NOT RECORDED." />
      <div className="workspace-3col">
        <aside className="inspector">
          <Panel title="Replays">
            {!replays.length ? (
              <EmptyState title="No private replays on disk" detail="Arena may complete with REPLAY NOT RECORDED." />
            ) : (
              <ul className="stack">
                {replays.map((r) => (
                  <li key={String(r.id)}>
                    <button type="button" className={selected === r.id ? "btn primary" : "btn ghost"} onClick={() => setSelected(String(r.id))}>
                      <span className="mono">{String(r.id)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </aside>
        <section className="board-stage">
          {frame ? <GeneralsBoard frame={frame} /> : <EmptyState title="No board frame" detail="Selected replay has no renderable grids (NOT RECORDED)." />}
        </section>
        <aside className="inspector">
          <Panel title="Inspector" actions={detail ? <button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>Raw</button> : null}>
            <p className="muted">Playback scrubber appears when frame timelines are recorded.</p>
            <dl className="kv">
              <dt>Selected</dt>
              <dd className="mono">{selected || "—"}</dd>
              <dt>Beliefs</dt>
              <dd>NOT RECORDED</dd>
            </dl>
          </Panel>
        </aside>
      </div>
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Replay raw" recordId={selected || undefined} schema="REPLAY" provenance="LOCAL_REPLAY" raw={detail} />
    </div>
  );
}

export function NotFoundPage() {
  return <EmptyState title="Not found" detail="No route matched this path." />;
}
