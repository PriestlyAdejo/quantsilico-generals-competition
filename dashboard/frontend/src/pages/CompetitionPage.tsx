import React, { useEffect, useState } from "react";
import { useDataSource } from "../app/DataSourceProvider";
import { PortalObservation, ManualSubmissionRecord } from "../types/competition";
import { DataSourceKind } from "../types/common";
import SourceSelector from "../components/data-display/SourceSelector";
import DataSourceBadge from "../components/status/DataSourceBadge";
import DateTimeCell from "../components/data-display/DateTimeCell";
import { Info } from "lucide-react";

type Source = "OFFICIAL_PORTAL_OBSERVATION" | "MANUALLY_RECORDED";

export default function CompetitionPage() {
  const ds = useDataSource();
  const [source, setSource] = useState<Source>("OFFICIAL_PORTAL_OBSERVATION");
  const [portalObs, setPortalObs] = useState<PortalObservation[]>([]);
  const [manualRecs, setManualRecs] = useState<ManualSubmissionRecord[]>([]);

  useEffect(() => {
    ds.listPortalObservations().then(setPortalObs);
    ds.listManualSubmissionRecords().then(setManualRecs);
  }, [ds]);

  return (
    <div className="p-6 space-y-6">
      <header>
        <p className="text-[#FFB000] font-mono text-xs uppercase tracking-widest mb-1">$ competition/</p>
        <h1 className="text-2xl font-bold text-[#EAF0F6]" style={{ fontFamily: "var(--font-heading)" }}>Competition</h1>
        <p className="text-[#8593A1] text-sm mt-1">Official portal observations and manually recorded submissions.</p>
      </header>

      <SourceSelector
        options={[
          { kind: "OFFICIAL_PORTAL_OBSERVATION", label: "Official Portal Obs", count: portalObs.length },
          { kind: "MANUALLY_RECORDED", label: "Manually Recorded", count: manualRecs.length },
        ]}
        value={source as DataSourceKind}
        onChange={v => setSource(v as Source)}
      />

      {source === "OFFICIAL_PORTAL_OBSERVATION" && (
        <div>
          {portalObs.length === 0 ? (
            <div className="border border-[#1E2630] rounded-sm p-8 bg-[#0C1116] text-center">
              <Info size={24} className="text-[#4A5568] mx-auto mb-3" />
              <p className="text-[#6F7C89] font-mono text-xs uppercase tracking-wide mb-1">No Portal Observation Imported</p>
              <p className="text-[#4A5568] font-mono text-xs max-w-sm mx-auto">
                No official portal observation has been imported for this candidate. Portal observations must be manually copied from the competition website and entered into the system.
              </p>
              <DataSourceBadge kind="OFFICIAL_PORTAL_OBSERVATION" pill />
            </div>
          ) : (
            <div className="space-y-3">
              {portalObs.map(obs => (
                <div key={obs.id} className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
                  <DataSourceBadge kind="OFFICIAL_PORTAL_OBSERVATION" />
                  <p className="text-[#CDD6DF] font-mono text-sm font-bold">{obs.candidateName}</p>
                  <div className="grid grid-cols-3 gap-3 mt-2">
                    <div><p className="text-[#6F7C89] font-mono text-xs">Rank</p><p className="text-[#CDD6DF] font-mono text-lg font-bold">{obs.rank ?? "—"}</p></div>
                    <div><p className="text-[#6F7C89] font-mono text-xs">Score</p><p className="text-[#CDD6DF] font-mono text-lg font-bold">{obs.score?.toFixed(3) ?? "—"}</p></div>
                    <div><p className="text-[#6F7C89] font-mono text-xs">Observed</p><p className="text-[#CDD6DF] font-mono text-xs">{obs.observedAt}</p></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {source === "MANUALLY_RECORDED" && (
        <div className="space-y-3">
          {manualRecs.length === 0 ? (
            <p className="text-[#4A5568] font-mono text-xs">No manual records.</p>
          ) : manualRecs.map(rec => (
            <div key={rec.id} className="border border-[#1E2630] rounded-sm p-4 bg-[#0C1116]">
              <DataSourceBadge kind={rec.kind} />
              <p className="text-[#CDD6DF] font-mono text-sm font-bold">{rec.candidateName}</p>
              <div className="grid grid-cols-2 gap-3 mt-2">
                <div><p className="text-[#6F7C89] font-mono text-xs">Submitted</p><p className="text-[#CDD6DF] font-mono text-xs"><DateTimeCell iso={rec.submittedAt} /></p></div>
                <div><p className="text-[#6F7C89] font-mono text-xs">Method</p><p className="text-[#CDD6DF] font-mono text-xs">{rec.method}</p></div>
              </div>
              {rec.notes && <p className="text-[#8593A1] font-mono text-xs mt-2">{rec.notes}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
