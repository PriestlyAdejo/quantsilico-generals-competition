import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { RepositoryStatus, CommitRecord, CiRun, EnvironmentLock } from "../types/repository";
import DataSourceBadge from "../components/status/DataSourceBadge";
import { Check, X, Clock, AlertTriangle, Minus } from "lucide-react";

const statusIcon = (s: string) => {
  const u = s.toUpperCase();
  if (u === "PASS" || u === "PASSED" || u === "VALIDATED") return <Check size={12} className="text-[#3FB950]" />;
  if (u === "FAIL" || u === "FAILED") return <X size={12} className="text-[#F85149]" />;
  if (u === "RUNNING") return <Clock size={12} className="text-[#FFB000]" />;
  if (u === "NOT_RUN" || u === "NOT_RECORDED" || u === "NOT_CONFIGURED" || u === "NOT_APPLICABLE" || u === "UNKNOWN") {
    return <Minus size={12} className="text-[#6F7C89]" />;
  }
  return <AlertTriangle size={12} className="text-[#6F7C89]" />;
};

const statusColor = (value: string) => {
  const u = value.toUpperCase();
  if (u === "PASS" || u === "PASSED" || u === "VALIDATED") return "#3FB950";
  if (u === "FAIL" || u === "FAILED") return "#F85149";
  return "#CDD6DF";
};

export default function RepositoryPage() {
  const ds = useDataSource();
  const [repoStatus, setRepoStatus] = useState<RepositoryStatus | null>(null);
  const [commits, setCommits] = useState<CommitRecord[]>([]);
  const [ciRuns, setCiRuns] = useState<CiRun[]>([]);
  const [locks, setLocks] = useState<EnvironmentLock[]>([]);

  useEffect(() => {
    ds.getRepositoryStatus().then(setRepoStatus);
    ds.listRecentCommits().then(setCommits);
    ds.listCiRuns().then(setCiRuns);
    ds.getEnvironmentLocks().then(setLocks);
  }, [ds]);

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ repository/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Repository</h1>
        <p className="text-[#8593A1] text-sm mt-1">Read-only — no mutation controls.</p>
        <p className="text-[#6F7C89] font-mono text-[10px] mt-1">
          <a className="text-[#22D3EE] hover:underline" href="/documentation/repository">About Repository</a>
          {" · "}
          Statuses use PASS / FAIL / NOT_RUN / NOT_RECORDED — never generic &quot;skipped&quot;.
        </p>
      </header>

      {repoStatus && (
        <>
          <DataSourceBadge kind={repoStatus.kind} />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { label: "Branch", value: repoStatus.branch, mono: true },
              { label: "Engine SHA", value: repoStatus.engineSha, mono: true },
              { label: "Hardware", value: repoStatus.hardware, mono: false },
              { label: "Linux Parity", value: repoStatus.linuxParityStatus, mono: true, status: true },
              { label: "Tests", value: repoStatus.testStatus, mono: true, status: true },
              { label: "Package", value: repoStatus.packageStatus, mono: true },
            ].map(({ label, value, mono, status }) => (
              <div key={label} className="border border-[#1E2630] rounded-sm p-3 bg-[#0C1116]">
                <p className="text-[#6F7C89] font-mono text-xs uppercase mb-1">{label}</p>
                <div className="flex items-center gap-1.5">
                  {status && statusIcon(String(value))}
                  <p className={`${mono ? "font-mono" : ""} text-sm font-bold break-all`} style={{ color: statusColor(String(value)) }}>
                    {value}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-3">Recent Commits</p>
          <div className="border border-[#1E2630] rounded-sm overflow-hidden">
            {commits.length === 0 ? (
              <p className="p-3 text-[#6F7C89] font-mono text-xs">NOT RECORDED</p>
            ) : commits.map((commit, i) => (
              <div key={commit.id} className={`p-3 ${i < commits.length - 1 ? "border-b border-[#1E2630]" : ""}`}>
                <div className="flex items-center justify-between">
                  <span className="text-[#FFB000] font-mono text-xs">{commit.sha}</span>
                  <DataSourceBadge kind={commit.kind} pill />
                </div>
                <p className="text-[#CDD6DF] font-mono text-xs mt-1">{commit.message}</p>
                <p className="text-[#6F7C89] font-mono text-xs mt-0.5">
                  {commit.author} · {commit.committedAt === "NOT RECORDED" ? "NOT RECORDED" : new Date(commit.committedAt).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section>
          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mb-3">CI Runs</p>
          <div className="border border-[#1E2630] rounded-sm overflow-hidden">
            {ciRuns.length === 0 ? (
              <p className="p-3 text-[#6F7C89] font-mono text-xs">NOT RECORDED — no cached CI runs available locally.</p>
            ) : ciRuns.map((run, i) => (
              <div key={run.id} className={`p-3 flex items-center justify-between ${i < ciRuns.length - 1 ? "border-b border-[#1E2630]" : ""}`}>
                <div className="flex items-center gap-2">
                  {statusIcon(run.status)}
                  <div>
                    <p className="text-[#CDD6DF] font-mono text-xs">{run.suite}</p>
                    <p className="text-[#6F7C89] font-mono text-xs">{run.commitSha} · {run.durationSecs ?? "—"}s</p>
                  </div>
                </div>
                <DataSourceBadge kind={run.kind} pill />
              </div>
            ))}
          </div>

          <p className="text-[#8593A1] font-mono text-xs uppercase tracking-widest mt-4 mb-3">Environment Locks</p>
          <div className="border border-[#1E2630] rounded-sm overflow-hidden">
            {locks.length === 0 ? (
              <p className="p-3 text-[#6F7C89] font-mono text-xs">NOT RECORDED</p>
            ) : locks.map((lock, i) => (
              <div key={lock.id} className={`p-3 ${i < locks.length - 1 ? "border-b border-[#1E2630]" : ""}`}>
                <div className="flex items-center justify-between">
                  <span className="text-[#CDD6DF] font-mono text-xs">{lock.name}</span>
                  <DataSourceBadge kind={lock.kind} pill />
                </div>
                <p className="text-[#8593A1] font-mono text-xs mt-1">{lock.reason}</p>
                <p className="text-[#6F7C89] font-mono text-xs">source: {lock.lockedBy}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
