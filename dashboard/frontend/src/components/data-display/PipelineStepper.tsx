import React from "react";
import { PipelineStep } from "../../types/submission";
import { Check, Clock, AlertTriangle, Loader } from "lucide-react";

interface Props {
  steps: PipelineStep[];
  className?: string;
}

export default function PipelineStepper({ steps, className = "" }: Props) {
  return (
    <div className={`flex flex-col gap-0 ${className}`}>
      {steps.map((step, i) => (
        <div key={step.stage} className="flex items-stretch gap-3">
          <div className="flex flex-col items-center">
            <StepIcon status={step.status} />
            {i < steps.length - 1 && (
              <div className={`w-px flex-1 mt-1 ${step.status === "complete" ? "bg-[#3FB950]" : "bg-[#1E2630]"}`} style={{ minHeight: 16 }} />
            )}
          </div>
          <div className="pb-4 min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span
                className={`text-xs font-semibold uppercase tracking-wide ${
                  step.status === "complete" ? "text-[#3FB950]"
                  : step.status === "active" ? "text-[#FFB000]"
                  : step.status === "blocked" ? "text-[#F85149]"
                  : "text-[#6F7C89]"
                }`}
                style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
              >
                {step.label}
              </span>
              {step.completedAt && (
                <span className="text-[#4A5568]" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                  {new Date(step.completedAt).toLocaleDateString("en-GB", { day: "2-digit", month: "short" })}
                </span>
              )}
            </div>
            {step.blockerReason && (
              <p className="text-[#F85149] mt-0.5" style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}>
                {step.blockerReason}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function StepIcon({ status }: { status: PipelineStep["status"] }) {
  const base = "flex-shrink-0 w-5 h-5 rounded-full border flex items-center justify-center mt-0.5";
  if (status === "complete") return (
    <div className={`${base} bg-[#3FB950] border-[#3FB950]`}>
      <Check size={10} className="text-black" />
    </div>
  );
  if (status === "active") return (
    <div className={`${base} border-[#FFB000]`}>
      <Loader size={10} className="text-[#FFB000] animate-spin" />
    </div>
  );
  if (status === "blocked") return (
    <div className={`${base} border-[#F85149]`}>
      <AlertTriangle size={10} className="text-[#F85149]" />
    </div>
  );
  return (
    <div className={`${base} border-[#2D3748]`}>
      <Clock size={10} className="text-[#6F7C89]" />
    </div>
  );
}
