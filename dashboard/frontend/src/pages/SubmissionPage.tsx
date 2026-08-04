import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { SubmissionPipeline, PackageAction } from "../types/submission";
import PipelineStepper from "../components/data-display/PipelineStepper";
import DataSourceBadge from "../components/status/DataSourceBadge";
import { ShieldAlert, Info } from "lucide-react";
import { toast } from "sonner";

export default function SubmissionPage() {
  const ds = useDataSource();
  const [pipeline, setPipeline] = useState<SubmissionPipeline | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => { ds.getSubmissionPipeline().then(setPipeline); }, [ds]);

  const runAction = async (action: PackageAction, label: string) => {
    if (pipeline?.kind !== "DEMO" && pipeline?.kind !== "IMPORTED_PROJECT_EVIDENCE") return;
    setRunning(true);
    try {
      const updated = await ds.runDemoPackageAction(action);
      setPipeline(updated);
      toast.success(`DEMO: ${label} simulated`);
    } catch {
      toast.error("Demo action failed");
    } finally {
      setRunning(false);
    }
  };

  const pkg = pipeline?.activePackage;

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ submission/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Submission</h1>
      </header>

      <div className="border-2 border-[#F85149] border-opacity-60 rounded-sm p-4 bg-[#0C1116] flex items-start gap-3">
        <ShieldAlert size={20} className="text-[#F85149] flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-[#F85149] font-mono text-xs font-bold uppercase tracking-wider mb-1">Uploads Are Manual By Design</p>
          <p className="text-[#8593A1] font-mono text-xs">Credentials never enter this application. File upload to the competition portal must be performed manually outside this console. The console only tracks package state and provides manual upload instructions.</p>
        </div>
      </div>

      {pipeline && <DataSourceBadge kind={pipeline.kind} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-3">Pipeline</p>
          {pipeline ? <PipelineStepper steps={pipeline.steps} /> : <p className="text-[#4A5568] font-mono text-xs">Loading…</p>}
        </section>

        <section>
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-3">Package Manifest</p>
          {pkg ? (
            <div className="border border-[#1E2630] rounded-sm overflow-hidden">
              {[
                ["Candidate", pkg.candidateName],
                ["Checkpoint", pkg.checkpoint],
                ["SHA-256", pkg.sha256 ?? "NOT BUILT"],
                ["Size", pkg.sizeBytes != null ? `${(pkg.sizeBytes / 1024 / 1024).toFixed(1)} MB` : "NOT BUILT"],
                ["Built At", pkg.builtAt ?? "NOT BUILT"],
                ["Windows Validated", pkg.validatedWindowsAt ?? "NOT VALIDATED"],
                ["Linux Validated", pkg.validatedLinuxAt ?? "NOT VALIDATED"],
              ].map(([k, v]) => (
                <div key={k} className="flex border-b border-[#1E2630] last:border-b-0">
                  <div className="px-3 py-2 w-44 flex-shrink-0 bg-[#0C1116]" style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#6F7C89", textTransform: "uppercase" }}>{k}</div>
                  <div className="px-3 py-2 flex-1 font-mono text-xs" style={{ color: v?.startsWith("NOT") ? "#4A5568" : "#CDD6DF" }}>{v}</div>
                </div>
              ))}
            </div>
          ) : <p className="text-[#4A5568] font-mono text-xs">No package.</p>}

          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-2 mt-4">DEMO Actions</p>
          <div className="flex flex-wrap gap-2">
            {([
              { action: "simulate_build" as PackageAction, label: "Simulate Build" },
              { action: "simulate_validate_windows" as PackageAction, label: "Simulate Win Validate" },
              { action: "simulate_validate_linux" as PackageAction, label: "Simulate Linux Validate" },
              { action: "mark_upload_ready" as PackageAction, label: "Mark Upload Ready" },
            ]).map(({ action, label }) => (
              <button
                key={action}
                onClick={() => runAction(action, label)}
                disabled={running}
                className="px-3 py-1.5 rounded-sm border border-[#FFB000] border-opacity-60 text-[#FFB000] hover:bg-[#1A1608] transition-colors disabled:opacity-40"
                style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase" }}
              >
                {label}
              </button>
            ))}
          </div>
          <DataSourceBadge kind="DEMO" />

          <div className="border border-[#1E2630] rounded-sm p-3 mt-4 bg-[#0C1116]">
            <div className="flex items-center gap-2 mb-2">
              <Info size={12} className="text-[#22D3EE]" />
              <span className="text-[#22D3EE] font-mono text-xs uppercase">Manual Upload Instructions</span>
            </div>
            <p className="text-[#8593A1] font-mono text-xs">
              1. Build the package locally using the project build script.<br />
              2. Validate on Windows and Linux using the provided test runner.<br />
              3. Navigate to the competition portal web UI.<br />
              4. Upload the package file manually.<br />
              5. Record the submission in the Competition tab.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
