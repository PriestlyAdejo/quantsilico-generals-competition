"""Dashboard capability catalogue — enabled flag + backend-owned reason."""

from __future__ import annotations

from typing import Any


def capability(enabled: bool, reason: str) -> dict[str, Any]:
    return {"enabled": enabled, "reason": reason}


def build_capabilities(*, arena_match: bool = True) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "DASHBOARD_CAPABILITIES",
        "capabilities": {
            "arena_match_launch": capability(
                arena_match,
                "Allowlisted local evaluator match via JobService."
                if arena_match
                else "Arena match launch is unavailable.",
            ),
            "arena_cancel": capability(
                False,
                "In-flight match cancellation is not safely supported for the current evaluator wrapper.",
            ),
            "environment_inspect": capability(
                True,
                "Read-only official/replay-derived observation inspection is available.",
            ),
            "environment_reset": capability(
                True,
                "Official Environment Lab sessions support reset within TTL and concurrency limits.",
            ),
            "environment_step": capability(
                True,
                "Official Environment Lab sessions support typed step with legal-action validation.",
            ),
            "training_launch": capability(
                False,
                "Long training campaigns cannot be started from the dashboard integration.",
            ),
            "training_pause": capability(False, "Training control is not exposed in this console."),
            "training_cancel": capability(False, "Training control is not exposed in this console."),
            "training_resume": capability(False, "Training control is not exposed in this console."),
            "package_build": capability(
                False,
                "Package builds remain CLI/operator-only to protect immutable submitted artefacts.",
            ),
            "package_validate": capability(
                True,
                "Allowlisted submission validation may be invoked via JobService when configured.",
            ),
            "package_reveal": capability(
                True,
                "Reveal uses a server-resolved record ID; browser-supplied paths are rejected.",
            ),
            "portal_upload": capability(
                False,
                "Competition uploads are manual by design.",
            ),
            "git_mutation": capability(
                False,
                "The console is read-only for repository state.",
            ),
        },
    }
