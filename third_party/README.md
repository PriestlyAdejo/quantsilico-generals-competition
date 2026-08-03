# Official Generals engine (third-party)

## Official repository

https://github.com/strakam/generals-bots

## Pin recorded at bootstrap

| Field | Value |
|-------|-------|
| Branch | `master` |
| Commit | `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462` |
| Commit date | `2026-07-27T17:04:29+02:00` |
| Subject | Competition: pin the sandbox's Python library versions in requirements.txt |
| Retrieval date | `2026-08-03` |

## Why a submodule

The official engine is the source of truth for rules, protocols, agent
runners, and the competition sandbox dependency lock. Pinning it as a Git
submodule:

- records an exact upstream commit;
- avoids vendoring a modified fork by accident;
- makes upgrades an explicit submodule bump.

## Rule: no silent local engine modifications

Do **not** edit files under `third_party/generals-bots` for strategy work.
If a local change to the engine is ever required for diagnosis:

1. document it;
2. keep it ephemeral;
3. do not commit silent patches into this private repository as if they were
   upstream behaviour.

Competition runtime versions come from:

`third_party/generals-bots/competition/requirements.txt`
