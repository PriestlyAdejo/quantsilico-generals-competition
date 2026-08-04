import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { PageHeader, Panel, PipelineStepper } from "../components/data-display/Panel";
import { RawRecordDrawer } from "../components/feedback/RawRecordDrawer";
import { BackendUnavailable, LoadingState } from "../components/feedback/States";
import { StatusBadge } from "../components/status/StatusBadge";

export default function SubmissionPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);
  const [rawOpen, setRawOpen] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    ds.getJson("/api/submission", ac.signal)
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.kind === "backend_unavailable") setDown(true);
      });
    return () => ac.abort();
  }, [ds]);

  if (down) return <BackendUnavailable />;
  if (!data) return <LoadingState />;

  const pkg = (data.package || {}) as Record<string, string>;
  const steps = [
    { id: "cand", label: "Candidate selected", state: "done" as const },
    { id: "pkg", label: "Package built", state: pkg.package_path ? ("done" as const) : ("pending" as const) },
    { id: "win", label: "Windows validated", state: pkg.windows_validation ? ("done" as const) : ("pending" as const) },
    { id: "linux", label: "Linux validated", state: pkg.linux_parity ? ("done" as const) : ("pending" as const) },
    { id: "ready", label: "Upload ready", state: "done" as const },
    { id: "manual", label: "Manually submitted", state: pkg.lifecycle === "SUBMITTED" ? ("done" as const) : ("pending" as const) },
    { id: "portal", label: "Portal submission gate passed", state: pkg.portal_verdict === "QUALIFIED" ? ("done" as const) : ("pending" as const) },
  ];

  return (
    <div>
      <PageHeader eyebrow="$ SUBMISSION /" title="Submission" subtitle="Manual upload only. Credentials never enter this application." />
      <div className="banner failure">UPLOADS ARE MANUAL BY DESIGN. File upload must be performed outside this console.</div>
      <div className="grid-2">
        <Panel title="Pipeline">
          <PipelineStepper steps={steps} />
        </Panel>
        <Panel title="Package manifest" actions={<button type="button" className="btn ghost" onClick={() => setRawOpen(true)}>Raw</button>}>
          <dl className="kv">
            <dt>Candidate</dt>
            <dd className="mono">{pkg.candidate}</dd>
            <dt>SHA-256</dt>
            <dd className="mono">{pkg.package_sha256}</dd>
            <dt>Config hash</dt>
            <dd className="mono">{pkg.config_hash || "—"}</dd>
            <dt>Policy source</dt>
            <dd className="mono">{pkg.authoritative_policy_source_commit}</dd>
            <dt>Embedded bot_commit</dt>
            <dd className="mono">
              {pkg.embedded_bot_commit} <StatusBadge value={pkg.embedded_metadata_status || "STALE"} />
            </dd>
            <dt>Lifecycle</dt>
            <dd>
              <StatusBadge value={pkg.lifecycle || "SUBMITTED"} />
            </dd>
          </dl>
          <p className="muted">{pkg.metadata_note}</p>
        </Panel>
      </div>
      <RawRecordDrawer open={rawOpen} onClose={() => setRawOpen(false)} title="Submission raw" recordId="submission" schema="SUBMISSION_DASHBOARD" provenance="SUBMITTED_PACKAGE" raw={data} />
    </div>
  );
}
