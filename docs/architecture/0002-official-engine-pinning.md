# 0002 — Official engine pinning

## Status

Accepted (bootstrap)

## Context

The EquiLibre × ÚFAL Generals Competition evaluates bots against an official
engine and a pinned sandbox dependency set.

## Decision

Pin https://github.com/strakam/generals-bots as a Git submodule at
`third_party/generals-bots`.

Bootstrap pin:

- commit `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462`
- branch `master`
- retrieved `2026-08-03`

Authoritative runtime lock:

`third_party/generals-bots/competition/requirements.txt`

Install order inside `.venv`:

1. upgrade pip / setuptools / wheel
2. install official requirements
3. install the engine editable with `--no-deps`
4. install this private package with `[dev]`
5. `pip check`

## Consequences

- Engine upgrades are explicit submodule bumps plus documentation updates.
- Local silent patches to the engine are forbidden.
- Native Windows success is not claimed as Linux sandbox parity.
