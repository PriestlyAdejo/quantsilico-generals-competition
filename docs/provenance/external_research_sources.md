# External research sources (provenance only)

External repositories and papers are **research inputs**. They are not the
QuantSilico product identity. Do not name packages, modules, candidates, or
branches after external bots or authors.

## Pinning policy

- Clone external sources **outside** this canonical repository.
- Do not vendor unlicensed code into `src/`.
- Record URL, commit, and licence hash when a clone is performed.
- Until cloned, pins remain `UNPINNED_PENDING_CLONE`.

## Intended sources

| ID | Role | Licence |
|----|------|---------|
| official_generals_bots | Rules/sim authority (already submodule) | As in submodule |
| external_paper_method | Methodological inspiration for PPO/transformer ideas | Verify at access time |
| external_public_code | Optional inspection of public PPO examples | Verify at access time |

See `external_research_sources.json` for machine-readable fields.
