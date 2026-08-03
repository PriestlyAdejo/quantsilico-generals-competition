import { useEffect, useState } from "react";

export default function OverviewPage() {
  const [data, setData] = useState<unknown>(null);
  useEffect(() => {
    fetch("/api/overview")
      .then((r) => r.json())
      .then(setData)
      .catch((err) => setData({ error: String(err) }));
  }, []);
  return (
    <section className="panel">
      <h1>Overview</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </section>
  );
}
