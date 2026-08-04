import React, { ReactNode } from "react";

interface Props {
  title?: string;
  eyebrow?: string;
  children: ReactNode;
  className?: string;
  badge?: ReactNode;
  actions?: ReactNode;
}

export default function Panel({ title, eyebrow, children, className = "", badge, actions }: Props) {
  const hasHeader = title || eyebrow || badge || actions;
  return (
    <div className={`bg-[#11161C] border border-[#1E2630] rounded-sm ${className}`}>
      {hasHeader && (
        <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[#1E2630]">
          <div className="flex items-center gap-2 min-w-0">
            {eyebrow && (
              <span
                className="text-[#FFB000] uppercase tracking-widest flex-shrink-0"
                style={{ fontFamily: "var(--font-mono)", fontSize: 10 }}
              >
                $ {eyebrow}
              </span>
            )}
            {title && (
              <span
                className="text-[#EAF0F6] font-bold truncate"
                style={{ fontFamily: "var(--font-display)", fontSize: 13 }}
              >
                {title}
              </span>
            )}
            {badge}
          </div>
          {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
