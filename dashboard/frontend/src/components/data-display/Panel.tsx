import type { ReactNode } from "react";
import type { ChartProvenance } from "../../data/types";

export function Panel({
  title,
  children,
  actions,
}: {
  title?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="panel">
      {(title || actions) && (
        <div className="panel-head">
          {title ? <h2>{title}</h2> : <span />}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <header className="page-header">
      {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
      <h1>{title}</h1>
      {subtitle ? <p className="subtitle">{subtitle}</p> : null}
    </header>
  );
}

export function MetricCard({
  label,
  value,
  sublabel,
  tone,
}: {
  label: string;
  value: string;
  sublabel?: string;
  tone?: "accent" | "success" | "failure" | "muted";
}) {
  return (
    <div className={`metric-card${tone ? ` tone-${tone}` : ""}`}>
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sublabel ? <div className="sublabel">{sublabel}</div> : null}
    </div>
  );
}

export function MetricStrip({
  items,
}: {
  items: { label: string; value: string; sublabel?: string; tone?: "accent" | "success" | "failure" | "muted" }[];
}) {
  return (
    <div className="metric-strip">
      {items.map((item) => (
        <MetricCard key={item.label} {...item} />
      ))}
    </div>
  );
}

export function ChartCard({
  title,
  provenance,
  children,
}: {
  title: string;
  provenance?: ChartProvenance | ChartProvenance[];
  children: ReactNode;
}) {
  const list = provenance ? (Array.isArray(provenance) ? provenance : [provenance]) : [];
  return (
    <section className="chart-card panel">
      <div className="panel-head">
        <h2>{title}</h2>
      </div>
      {children}
      {list.length ? (
        <ul className="chart-sources">
          {list.map((p) => (
            <li key={`${p.manifestId}-${p.kind}`}>
              <span className="mono">{p.manifestId}</span>
              <span className="muted"> · {p.kind}</span>
              {p.architecture ? <span className="muted"> · {p.architecture}</span> : null}
              {p.missingFields?.length ? (
                <span className="muted"> · missing: {p.missingFields.join(", ")}</span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

export function PipelineStepper({
  steps,
}: {
  steps: { id: string; label: string; state: "done" | "active" | "pending" | "blocked" | "failed" }[];
}) {
  return (
    <ol className="pipeline-stepper">
      {steps.map((s) => (
        <li key={s.id} className={`step ${s.state}`}>
          <span className="dot" aria-hidden />
          <span className="step-label">{s.label}</span>
          <span className="step-state">{s.state}</span>
        </li>
      ))}
    </ol>
  );
}

export function EvidenceChecklist({
  items,
}: {
  items: { id: string; label: string; status: string; detail?: string }[];
}) {
  return (
    <ul className="evidence-checklist">
      {items.map((i) => (
        <li key={i.id}>
          <strong>{i.label}</strong>
          <span className={`pill ${String(i.status).toLowerCase()}`}>{i.status}</span>
          {i.detail ? <p className="muted">{i.detail}</p> : null}
        </li>
      ))}
    </ul>
  );
}

export function FilterBar({
  value,
  onChange,
  placeholder = "Search…",
  chips,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  chips?: { id: string; label: string; active?: boolean; onClick: () => void }[];
}) {
  return (
    <div className="filter-bar">
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label={placeholder}
      />
      {chips?.length ? (
        <div className="filter-chips">
          {chips.map((c) => (
            <button
              key={c.id}
              type="button"
              className={c.active ? "chip active" : "chip"}
              onClick={c.onClick}
            >
              {c.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function DataTable({
  columns,
  rows,
}: {
  columns: { key: string; label: string }[];
  rows: Record<string, ReactNode>[];
}) {
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key}>{row[c.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
