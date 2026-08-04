import React from "react";

export default function LoadingState() {
  return (
    <div className="flex items-center justify-center py-16">
      <span
        className="text-[#FFB000]"
        style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}
      >
        Initialising
        <span className="animate-pulse">_</span>
      </span>
    </div>
  );
}
