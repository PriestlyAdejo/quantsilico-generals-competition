import React, { useState } from "react";
import { Search, X } from "lucide-react";

export interface FilterChip {
  key: string;
  label: string;
  value: string;
  onRemove: () => void;
}

interface Props {
  search: string;
  onSearch: (v: string) => void;
  chips?: FilterChip[];
  placeholder?: string;
  actions?: React.ReactNode;
}

export default function FilterBar({ search, onSearch, chips = [], placeholder = "Search…", actions }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 pb-3 mb-3 border-b border-[#1E2630]">
      <div className="relative flex-1 min-w-40">
        <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#6F7C89]" />
        <input
          type="text"
          value={search}
          onChange={e => onSearch(e.target.value)}
          placeholder={placeholder}
          className="w-full bg-[#0C1116] border border-[#1E2630] rounded-sm pl-7 pr-3 py-1.5 text-[#CDD6DF] placeholder-[#4A5568] focus:outline-none focus:border-[#FFB000] transition-colors"
          style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
        />
      </div>
      {chips.map(chip => (
        <span
          key={chip.key}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-sm bg-[#1E2630] border border-[#2D3748]"
          style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "#CDD6DF" }}
        >
          <span className="text-[#6F7C89]">{chip.label}:</span>
          {chip.value}
          <button onClick={chip.onRemove} className="text-[#6F7C89] hover:text-[#F85149] ml-0.5">
            <X size={9} />
          </button>
        </span>
      ))}
      {actions}
    </div>
  );
}
