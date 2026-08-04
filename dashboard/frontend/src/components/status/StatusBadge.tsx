export function StatusBadge({ value }: { value: string }) {
  const v = value.toUpperCase();
  let cls = "badge missing";
  if (v === "PASS" || v === "SUBMITTED" || v === "COMPLETED") cls = "badge pass";
  else if (v === "FAIL" || v === "FAILED" || v === "FAILURE") cls = "badge fail";
  else if (v === "NONE" || v === "NOT_EVALUATED" || v === "DRAW") cls = "badge none";
  else if (v.includes("QUALIFIED") || v === "INFO") cls = "badge info";
  return (
    <span className={cls} title={value}>
      {value}
    </span>
  );
}

export function ProvenanceBadge({
  provenance,
  observedAt,
}: {
  provenance: string;
  observedAt?: string;
}) {
  return (
    <span className="badge accent" title={observedAt || provenance}>
      {provenance}
      {observedAt ? ` · ${observedAt}` : ""}
    </span>
  );
}
