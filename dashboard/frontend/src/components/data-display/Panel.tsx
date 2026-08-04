import type { ReactNode } from "react";

export function Panel({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="panel">
      {title ? <h2 style={{ marginTop: 0, fontSize: "0.95rem" }}>{title}</h2> : null}
      {children}
    </section>
  );
}

export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <header className="page-header" style={{ marginBottom: "0.85rem" }}>
      <h1>{title}</h1>
      {subtitle ? <p>{subtitle}</p> : null}
    </header>
  );
}

export function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

export function MetricStrip({ items }: { items: { label: string; value: string }[] }) {
  return (
    <div className="metric-strip">
      {items.map((item) => (
        <MetricCard key={item.label} label={item.label} value={item.value} />
      ))}
    </div>
  );
}
