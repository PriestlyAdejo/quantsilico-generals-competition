import React from "react";

type Variant = "success" | "warning" | "error" | "neutral" | "info";

interface Props {
  variant: Variant;
  label: string;
  dot?: boolean;
}

const VARIANT_STYLES: Record<Variant, { border: string; text: string; dot: string; bg: string }> = {
  success: { border: "border-[#3FB950]", text: "text-[#3FB950]", dot: "bg-[#3FB950]", bg: "bg-[#0C1116]" },
  warning: { border: "border-[#FFB000]", text: "text-[#FFB000]", dot: "bg-[#FFB000]", bg: "bg-[#0C1116]" },
  error:   { border: "border-[#F85149]", text: "text-[#F85149]", dot: "bg-[#F85149]", bg: "bg-[#0C1116]" },
  neutral: { border: "border-[#1E2630]", text: "text-[#8593A1]", dot: "bg-[#6F7C89]", bg: "bg-[#0C1116]" },
  info:    { border: "border-[#22D3EE]", text: "text-[#22D3EE]", dot: "bg-[#22D3EE]", bg: "bg-[#0C1116]" },
};

export default function StatusBadge({ variant, label, dot = false }: Props) {
  const s = VARIANT_STYLES[variant];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm border ${s.border} ${s.text} ${s.bg}`}
      style={{ fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.06em" }}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${s.dot}`} />}
      {label}
    </span>
  );
}
