import { useEffect, useState } from "react";
import { useDataSource } from "../data/DataSourceContext";
import { ApiError } from "../data/types";
import { PageHeader, Panel } from "../components/data-display/Panel";
import { BackendUnavailable, LoadingState } from "../components/feedback/States";

export default function SubmissionPage() {
  const ds = useDataSource();
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [down, setDown] = useState(false);

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

  return (
    <div>
      <PageHeader title="Submission" subtitle="Immutable submitted package record." />
      <div className="warning" style={{ marginBottom: "0.75rem" }}>
        Uploads are manual by design. Credentials never enter this application.
      </div>
      <Panel title="Package identity">
        <div className="pre">
          {`candidate: ${pkg.candidate}
package_path: ${pkg.package_path}
package_sha256: ${pkg.package_sha256}
config_hash: ${pkg.config_hash}
authoritative_policy_source_commit: ${pkg.authoritative_policy_source_commit}
embedded_bot_commit: ${pkg.embedded_bot_commit}
embedded_metadata_status: ${pkg.embedded_metadata_status}
repository_completion_commit: ${pkg.repository_completion_commit}
lifecycle: ${pkg.lifecycle}
portal_gate: ${pkg.portal_gate_name} (${pkg.portal_verdict})
`}
        </div>
        <p>
          Package policy bytes match commit <code>{pkg.authoritative_policy_source_commit}</code>. The
          embedded <code>bot_commit</code> contains stale metadata and is retained for auditability only.
        </p>
        <p className="muted">{pkg.metadata_note}</p>
      </Panel>
    </div>
  );
}
