import React, { useState } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "../../app/components/ui/sheet";
import { Copy, Check } from "lucide-react";
import DataSourceBadge from "../status/DataSourceBadge";
import { DataSourceKind } from "../../types/common";

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  kind?: DataSourceKind;
  record: Record<string, unknown>;
}

export default function RawRecordDrawer({ open, onClose, title, kind, record }: Props) {
  const [tab, setTab] = useState<"kv" | "json">("kv");
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(record, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const entries = Object.entries(record).filter(([k]) => k !== "saliencyMap" && k !== "beliefMap");

  return (
    <Sheet open={open} onOpenChange={o => { if (!o) onClose(); }}>
      <SheetContent side="right" className="w-full max-w-xl bg-[#0C1116] border-l border-[#1E2630] p-0">
        <SheetHeader className="px-4 pt-4 pb-3 border-b border-[#1E2630]">
          <div className="flex items-center justify-between">
            <SheetTitle className="text-[#CDD6DF]" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{title}</SheetTitle>
            <button onClick={handleCopy} className="flex items-center gap-1 text-[#6F7C89] hover:text-[#FFB000] transition-colors" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
              {copied ? <Check size={12} className="text-[#3FB950]" /> : <Copy size={12} />}
              {copied ? "COPIED" : "COPY"}
            </button>
          </div>
          {kind && <DataSourceBadge kind={kind} pill />}
        </SheetHeader>
        <div className="flex gap-0 border-b border-[#1E2630]">
          {(["kv", "json"] as const).map(t => (
            <button key={t} onClick={() => setTab(t)} className={`px-4 py-2 transition-colors ${tab === t ? "border-b-2 border-[#FFB000] text-[#FFB000]" : "text-[#6F7C89] hover:text-[#8593A1]"}`} style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase" }}>
              {t === "kv" ? "Key / Value" : "Raw JSON"}
            </button>
          ))}
        </div>
        <div className="overflow-y-auto p-4" style={{ maxHeight: "calc(100vh - 140px)" }}>
          {tab === "kv" ? (
            <dl className="space-y-2">
              {entries.map(([k, v]) => (
                <div key={k} className="grid grid-cols-2 gap-2 py-1.5 border-b border-[#1E2630]">
                  <dt className="text-[#6F7C89]" style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase" }}>{k}</dt>
                  <dd className="text-[#CDD6DF] break-all" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                    {v === null ? <span className="text-[#4A5568]">null</span> : String(v)}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <pre className="text-[#CDD6DF] text-xs overflow-x-auto" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
              {JSON.stringify(record, null, 2)}
            </pre>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
