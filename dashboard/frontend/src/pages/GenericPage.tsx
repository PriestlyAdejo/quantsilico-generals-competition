import { useEffect, useState } from "react";

export default function GenericPage({
  title,
  endpoint,
}: {
  title: string;
  endpoint: string;
}) {
  const [data, setData] = useState<unknown>(null);
  useEffect(() => {
    fetch(endpoint)
      .then((r) => r.json())
      .then(setData)
      .catch((err) => setData({ error: String(err) }));
  }, [endpoint]);
  return (
    <section className="panel">
      <h1>{title}</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </section>
  );
}
