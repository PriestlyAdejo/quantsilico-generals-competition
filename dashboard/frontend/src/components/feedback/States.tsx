export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state-box" role="status">
      <strong>{label}</strong>
    </div>
  );
}

export function ErrorState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="state-box" role="alert">
      <strong>{title}</strong>
      {detail ? <div className="muted">{detail}</div> : null}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="state-box">
      <strong>{title}</strong>
      {detail ? <div className="muted">{detail}</div> : null}
    </div>
  );
}

export function BackendUnavailable({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="state-box" role="alert">
      <strong>BACKEND UNAVAILABLE</strong>
      <div className="muted">Navigation remains available. Demo data was not activated.</div>
      {onRetry ? (
        <button className="btn primary" type="button" onClick={onRetry} style={{ marginTop: "0.75rem" }}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function SchemaMismatch({ detail }: { detail?: string }) {
  return (
    <div className="state-box" role="alert">
      <strong>SCHEMA MISMATCH</strong>
      <div className="muted">{detail || "Unsupported schema_version — refusing to coerce."}</div>
    </div>
  );
}
