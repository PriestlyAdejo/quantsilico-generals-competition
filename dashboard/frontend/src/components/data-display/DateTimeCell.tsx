import React from "react";
import { fmtDate, fmtTime } from "../../utils/formatting";

interface Props {
  iso: string | null | undefined;
  /** Show date only, time only, or both (default). */
  mode?: "date" | "time" | "both";
  className?: string;
}

/** Honest date/time cell — never renders Invalid Date. */
export default function DateTimeCell({ iso, mode = "both", className = "" }: Props) {
  const date = fmtDate(iso);
  const time = fmtTime(iso);
  const missing = date === "NOT RECORDED";

  let text: string;
  if (missing) text = "NOT RECORDED";
  else if (mode === "date") text = date;
  else if (mode === "time") text = time;
  else text = `${date} ${time}`;

  return (
    <span
      className={`font-mono text-xs ${missing ? "text-[#6F7C89]" : "text-[#CDD6DF]"} ${className}`}
      title={iso && !missing ? iso : undefined}
    >
      {text}
    </span>
  );
}
