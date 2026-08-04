import React from "react";

interface Props {
  error: string;
}

export default function ErrorState({ error }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-8">
      <div
        className="text-[#F85149] font-bold mb-2"
        style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}
      >
        ERROR
      </div>
      <div
        className="text-[#F85149] opacity-70 text-center max-w-md"
        style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}
      >
        {error}
      </div>
    </div>
  );
}
