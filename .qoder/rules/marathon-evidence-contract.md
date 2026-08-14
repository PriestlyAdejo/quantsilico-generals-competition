---
trigger: always_on
---

# Marathon Evidence Contract

Applies to every session in this repository (plan `MARATHON_REDESIGN_LOCKED_V1`).

## Evidence before claims

- Never fabricate or infer results: tests, benchmarks, training metrics, GPU utilisation, tournament outcomes, leaderboard results, or artefacts.
- No `PASS`/`COMPLETE` without a real command run and captured output. Evidence hierarchy: implementation -> execution -> machine-readable result -> validation -> captured evidence -> documentation.
- Test code is not a passing test; a report is not the underlying result; state claiming PASS is not actual PASS.

## State discipline

- Update `experiments/marathon/ACTIVE_STATE.json` (protocol §7 exact key set) before stopping any session.
- Empty lists mean "audited and none found"; unaudited state is `UNKNOWN` with a reason.
- Material conflicts between code/config/evidence/docs are surfaced (`PLAN_CONFLICT`), never silently resolved.

## Git safety

- Forbidden: `git push --force`/`-f`, `git reset --hard` over valuable work, destructive `git clean`, deleting unmerged unique branches/worktrees, rewriting shared history.
- Preserve LF and executable bits on `*.sh`. Do not modify `third_party/generals-bots` (pinned engine).

## Human-controlled boundaries (from `configs/marathon/programme.yaml`)

Require explicit human authorization: new cash spend / credit purchases / payment methods / known overage, repository visibility changes (`REPOSITORY_PRIVACY_CUTOVER`), competition uploads, destructive removal of unique evidence, force push / history rewrite, pinned-engine modification, credential operations. Existing prepaid/promotional/trial compute credits may be used autonomously within the available balance.

## Experimental integrity

- Anything touching action selection declares exactly one `PPO_SEMANTICS` (`UNCHANGED`, `PRE_SAMPLING_MASK`, `OFF_POLICY_AUXILIARY`, `EVAL_ONLY`); missing/ambiguous semantics invalidates the experiment before training.
- Experiments record commit, config, seed, command, environment, raw logs, and machine-readable results. Failed experiments stay discoverable.
- Do not kill processes whose identity is unknown (WSL workload rule); profile before optimising.
