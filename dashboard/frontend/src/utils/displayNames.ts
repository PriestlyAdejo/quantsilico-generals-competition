/** Short, human-readable labels for long opaque IDs. */

export function shortDisplayName(id: string | null | undefined, max = 28): string {
  if (!id) return "NOT RECORDED";
  const trimmed = id.trim();
  if (!trimmed) return "NOT RECORDED";
  if (trimmed.length <= max) return trimmed;
  const head = Math.max(8, Math.floor((max - 1) / 2));
  const tail = Math.max(6, max - head - 1);
  return `${trimmed.slice(0, head)}…${trimmed.slice(-tail)}`;
}

export function titleFromId(id: string | null | undefined): string {
  if (!id) return "NOT RECORDED";
  return id
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
