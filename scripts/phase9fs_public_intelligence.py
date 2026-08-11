"""PUBLIC_INTELLIGENCE_TIMEBOX — public leaderboard/replay discovery (≤90 min)."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIMEBOX_S = 90 * 60

URLS = [
    "https://www.generals.bot/leaderboard",
    "https://www.generals.bot/api/leaderboard",
    "https://www.generals.bot/player?name=QuantSilico&id=88151",
    "https://www.generals.bot/about",
]


def _fetch(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "QuantSilicoResearchBot/0.1 (public read-only; no auth)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read(200_000).decode("utf-8", errors="replace")
            return {
                "url": url,
                "ok": True,
                "status": getattr(resp, "status", 200),
                "content_type": resp.headers.get("Content-Type", ""),
                "bytes": len(body),
                "mentions_quantsilico": bool(re.search(r"quantsilico|QuantSilico", body, re.I)),
                "looks_like_json": body.lstrip().startswith(("{", "[")),
                "replay_hrefs": re.findall(r'href=["\']([^"\']*replay[^"\']*)["\']', body, re.I)[:20],
                "api_hrefs": re.findall(r'href=["\']([^"\']*/api/[^"\']*)["\']', body, re.I)[:20],
            }
    except Exception as exc:  # noqa: BLE001
        return {"url": url, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    started = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    pages = []
    for url in URLS:
        if time.perf_counter() - started > TIMEBOX_S:
            break
        pages.append(_fetch(url))

    has_entry = any(p.get("mentions_quantsilico") for p in pages if p.get("ok"))
    lb_ok = any(p.get("ok") and "leaderboard" in p.get("url", "") for p in pages)
    status = "SNAPSHOT_RECORDED" if lb_ok else "LEADERBOARD_UNREACHABLE"
    if lb_ok and not has_entry:
        entry_status = "NO_ACTIVE_LEADERBOARD_ENTRY"
    elif has_entry:
        entry_status = "ACTIVE_OR_MENTIONED"
    else:
        entry_status = "UNKNOWN"

    partitions = {
        "TRAIN": {"policy": "PUBLIC_REPLAY_TRAIN_ONLY_AFTER_EPOCH", "leakage": "FORBIDDEN_UNTIL_PARTITIONED"},
        "DEVELOPMENT": {"policy": "LOCAL_OR_HELD_OUT_DEV", "leakage": "NO_TRAIN_OVERLAP"},
        "MONITORING": {"policy": "POST_UPLOAD_PUBLIC_OBSERVATION", "leakage": "N/A"},
        "CONFIRMATION": {"policy": "HELD_CONFIRMATION_SEEDS", "leakage": "NO_TRAIN_OVERLAP"},
    }

    doc = {
        "schema_version": 1,
        "kind": "PUBLIC_INTELLIGENCE_TIMEBOX",
        "created_at": now,
        "timebox_s": TIMEBOX_S,
        "elapsed_s": time.perf_counter() - started,
        "status": status,
        "leaderboard_entry_status": entry_status,
        "pages": pages,
        "replay_partitions": partitions,
        "auth_scraping": "FORBIDDEN_NOT_USED",
        "notes": [
            "Public read-only probes only; no login.",
            "Replay train/eval leakage forbidden until partitions assigned after public epoch.",
        ],
    }
    out = REPO / "experiments" / "manifests" / "phase9fs_public_intelligence.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (REPO / "experiments" / "reports" / "phase9fs_public_intelligence.md").write_text(
        "\n".join(
            [
                "# Public intelligence timebox",
                "",
                f"Created: {now}",
                "",
                f"- Status: **{status}**",
                f"- Leaderboard entry: **{entry_status}**",
                f"- Pages probed: {len(pages)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "entry": entry_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
