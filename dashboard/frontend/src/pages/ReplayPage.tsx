import { useEffect, useState } from "react";

type ReplayItem = { id: string; name: string };

export default function ReplayPage() {
  const [items, setItems] = useState<ReplayItem[]>([]);
  const [view, setView] = useState<string>("");
  useEffect(() => {
    fetch("/api/replays")
      .then((r) => r.json())
      .then((d) => setItems(d.replays ?? []));
  }, []);
  async function openReplay(id: string) {
    const res = await fetch(`/api/replays/${encodeURIComponent(id)}`);
    setView(JSON.stringify(await res.json(), null, 2));
  }
  return (
    <section className="panel">
      <h1>Replay Lab</h1>
      <p className="hint">Real replays from replays/private only.</p>
      <ul className="list">
        {items.map((item) => (
          <li key={item.id}>
            <button type="button" onClick={() => openReplay(item.id)}>
              {item.name}
            </button>
          </li>
        ))}
      </ul>
      {!items.length && <p className="hint">No replays recorded yet.</p>}
      <pre>{view}</pre>
    </section>
  );
}
