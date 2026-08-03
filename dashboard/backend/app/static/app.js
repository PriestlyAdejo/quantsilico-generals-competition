async function loadOverview() {
  const res = await fetch("/api/overview");
  const data = await res.json();
  document.getElementById("overview-data").textContent = JSON.stringify(data, null, 2);
}

async function loadReplays() {
  const res = await fetch("/api/replays");
  const data = await res.json();
  const ul = document.getElementById("replay-list");
  ul.innerHTML = "";
  for (const item of data.replays) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = item.name;
    btn.addEventListener("click", async () => {
      const r = await fetch(`/api/replays/${encodeURIComponent(item.id)}`);
      const body = await r.json();
      document.getElementById("replay-view").textContent = JSON.stringify(body, null, 2);
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

document.getElementById("match-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fd = new FormData(ev.target);
  const payload = {
    job_type: "MATCH",
    candidate: fd.get("candidate"),
    opponent: fd.get("opponent"),
    seed: Number(fd.get("seed")),
    max_turns: Number(fd.get("max_turns")),
    record_replay: true,
  };
  const out = document.getElementById("match-result");
  out.textContent = "running…";
  const res = await fetch("/api/jobs/match", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  out.textContent = JSON.stringify(data, null, 2);
  await loadReplays();
});

loadOverview();
loadReplays();
