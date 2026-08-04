import { useEffect, useId, useRef, useState, type ReactNode } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  title: string;
  recordId?: string;
  schema?: string;
  provenance?: string;
  timestamp?: string;
  formatted?: ReactNode;
  raw?: unknown;
};

/** Accessible secondary raw-record drawer — never the primary page content. */
export function RawRecordDrawer({
  open,
  onClose,
  title,
  recordId,
  schema,
  provenance,
  timestamp,
  formatted,
  raw,
}: Props) {
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);
  const [tab, setTab] = useState<"formatted" | "raw">("formatted");

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const rawText = typeof raw === "string" ? raw : JSON.stringify(raw ?? {}, null, 2);

  return (
    <div className="drawer-backdrop" role="presentation" onClick={onClose}>
      <aside
        className="raw-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="raw-drawer-header">
          <h2 id={titleId}>{title}</h2>
          <button ref={closeRef} type="button" className="btn ghost" onClick={onClose}>
            Close
          </button>
        </header>
        <dl className="kv">
          {recordId ? (
            <>
              <dt>Record ID</dt>
              <dd className="mono">{recordId}</dd>
            </>
          ) : null}
          {schema ? (
            <>
              <dt>Schema</dt>
              <dd className="mono">{schema}</dd>
            </>
          ) : null}
          {provenance ? (
            <>
              <dt>Provenance</dt>
              <dd>{provenance}</dd>
            </>
          ) : null}
          {timestamp ? (
            <>
              <dt>Timestamp</dt>
              <dd className="mono">{timestamp}</dd>
            </>
          ) : null}
        </dl>
        <div className="tabs" role="tablist">
          <button
            type="button"
            role="tab"
            className={tab === "formatted" ? "active" : undefined}
            aria-selected={tab === "formatted"}
            onClick={() => setTab("formatted")}
          >
            Formatted
          </button>
          <button
            type="button"
            role="tab"
            className={tab === "raw" ? "active" : undefined}
            aria-selected={tab === "raw"}
            onClick={() => setTab("raw")}
          >
            Raw JSON
          </button>
        </div>
        <div className="raw-drawer-body">
          {tab === "formatted" ? (
            formatted || <p className="muted">No formatted view.</p>
          ) : (
            <>
              <button
                type="button"
                className="btn ghost"
                onClick={() => void navigator.clipboard.writeText(rawText)}
              >
                Copy JSON
              </button>
              <pre className="pre">{rawText}</pre>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
