"""Public-only probe for portal version attribution fields.

Does not log in, submit, delete, or mutate portal state.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "manifests" / "portal_public_attribution_probe.json"

URLS = [
    "https://www.generals.bot/about",
    "https://www.generals.bot/leaderboard",
    "https://www.generals.bot/player?name=QuantSilico&id=88151",
]

INTERESTING = re.compile(
    r"(submission|package|sha256|hash|version|build|deploy|bot_id|playerId|elo|rank)",
    re.I,
)


def _fetch(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "QuantSilicoResearchBot/0.1 (public read-only probe)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read(250_000).decode("utf-8", errors="replace")
            ctype = resp.headers.get("Content-Type", "")
            return {
                "url": url,
                "status": getattr(resp, "status", 200),
                "content_type": ctype,
                "bytes": len(body),
                "ok": True,
                "interesting_snippets": [
                    m.group(0) for m in INTERESTING.finditer(body[:50_000])
                ][:40],
                "has_json_ld": "application/ld+json" in body.lower(),
                "mentions_package_hash": bool(re.search(r"sha[-_]?256|package.?hash", body, re.I)),
                "mentions_submission_id": bool(re.search(r"submission[_ -]?id", body, re.I)),
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "url": url,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "mentions_package_hash": False,
            "mentions_submission_id": False,
        }


def main() -> None:
    pages = [_fetch(u) for u in URLS]
    exact = any(p.get("mentions_package_hash") or p.get("mentions_submission_id") for p in pages)
    report = {
        "schema_version": 1,
        "kind": "PORTAL_PUBLIC_ATTRIBUTION_PROBE",
        "recording": "MANUALLY_RECORDED",
        "pages": pages,
        "finding": (
            "EXACT_PORTAL_VERSION_FIELDS_VISIBLE"
            if exact
            else "NO_PUBLIC_PACKAGE_HASH_OR_SUBMISSION_ID_BESIDE_MATCHES"
        ),
        "recommended_attribution_method": (
            "EXACT_PACKAGE_HASH" if exact else "MANUAL_OPERATOR_ASSIGNMENT"
        ),
        "warning": None
        if exact
        else (
            "WARNING: public pages inspected do not expose package hash / submission id "
            "beside matches; do not claim exact per-match version attribution."
        ),
        "notes": [
            "Probe is read-only HTTP GET of public URLs only.",
            "No authentication, upload, delete, or destructive actions.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("finding", "recommended_attribution_method", "warning")}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
