import React, { ReactNode } from "react";

interface Props {
  eyebrow: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}

export default function PageHeader({ eyebrow, title, subtitle, actions }: Props) {
  return (
    <div className="border-b border-[#1E2630] pb-4 mb-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div
            className="mb-1 text-[#FFB000] uppercase tracking-widest"
            style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
          >
            $ {eyebrow}
          </div>
          <h1
            className="text-xl font-bold text-[#EAF0F6]"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {title}
          </h1>
          {subtitle && (
            <p className="mt-1 text-sm text-[#8593A1]" style={{ fontFamily: "var(--font-body)" }}>
              {subtitle}
            </p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
      </div>
    </div>
  );
}
