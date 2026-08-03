import { FormEvent, useState } from "react";

export default function ArenaPage() {
  const [result, setResult] = useState<string>("");
  async function onSubmit(ev: FormEvent<HTMLFormElement>) {
    ev.preventDefault();
    const fd = new FormData(ev.currentTarget);
    setResult("running…");
    const res = await fetch("/api/jobs/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_type: "MATCH",
        candidate: fd.get("candidate"),
        opponent: fd.get("opponent"),
        seed: Number(fd.get("seed")),
        max_turns: Number(fd.get("max_turns")),
        record_replay: true,
      }),
    });
    const body = await res.json();
    setResult(JSON.stringify(body, null, 2));
  }
  return (
    <section className="panel">
      <h1>Arena</h1>
      <form onSubmit={onSubmit} className="form">
        <label>
          Candidate
          <select name="candidate" defaultValue="heuristic_v1">
            <option>heuristic_v1</option>
            <option>heuristic_v0</option>
            <option>pass</option>
            <option>legal_random</option>
            <option>expander</option>
          </select>
        </label>
        <label>
          Opponent
          <select name="opponent" defaultValue="expander">
            <option>expander</option>
            <option>pass</option>
            <option>legal_random</option>
            <option>heuristic_v1</option>
          </select>
        </label>
        <label>
          Seed
          <input name="seed" type="number" defaultValue={0} />
        </label>
        <label>
          Max turns
          <input name="max_turns" type="number" defaultValue={40} />
        </label>
        <button type="submit">Run match</button>
      </form>
      <pre>{result}</pre>
    </section>
  );
}
