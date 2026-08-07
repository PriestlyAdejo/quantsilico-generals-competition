# Cloud A100 forensic performance v1

## Outcome

`PERF_V1_REUSABLE_ROLLOUT_SCAN` is a narrow, semantics-preserving candidate.
Its frozen-fixture `PERFORMANCE_EQUIVALENCE_GATE` passed on CPU JAX 0.11.0, but
no A100 benchmark was run because production PPO was active. Do not interrupt or
promote into the production trainer until an identical controlled benchmark
shows a material sustained TPS gain.

Candidate runtime implementation hash:
`d66eb6f1c4f9809c27cdefaaa333083324f079e8e7ededb815e6e822470e3371`.

## Evidence-backed change

The frozen production collector constructed a new local `jax.jit` wrapper around
the full rollout `lax.scan` on every call. The candidate moves that transform to
module scope and makes `rollout_len` static. Array shapes and the static horizon
remain part of JAX's cache key, while repeated updates can reuse the same
top-level executable identity.

The candidate is 15 added and 9 removed lines relative to the frozen rollout
source. The CPU fixture observed one candidate executable-cache entry after two
identical calls at the same horizon.

## Performance equivalence gate

The gate compares the exact frozen pre-candidate wrapper with the reusable scan
using seed `20260807`. At `rtol=1e-6`, `atol=1e-6`, it covers final environment
state, recurrent observation memory, RNG, reset pool/cursor, observations, legal
masks, actions, log-probabilities, policy values, rewards, done semantics, GAE,
returns, PPO loss/metrics, gradients, updated parameters, optimizer state, and
an on-disk checkpoint save/load round trip.

Measured local results:

- equivalence gate: `1 passed in 169.44s`;
- targeted JAX/legal/PPO regressions: `4 passed in 34.16s`;
- backend: CPU, JAX 0.11.0;
- A100 benchmark: **not run**.

## Static forensic audit

- Collector: already uses a fully device-side `vmap` + `lax.scan`; there is no
  Python timestep or environment loop. The per-call top-level JIT construction
  was the clearest recompilation/cache risk and is the only implementation
  change in this candidate.
- Host/device synchronization: rollout collection explicitly waits for rewards
  and bootstrap values; the update waits for parameters and transfers small
  reward/done/return telemetry plus scalar metrics. There is no per-step
  `device_get`, `.item()`, or `np.asarray` in the scanned path. Fusing or reducing
  update telemetry may be benchmarked later, but current evidence does not
  justify changing it during the authoritative run.
- Reset pool: pool construction is expensive setup work, but the production
  runner builds it outside the update loop and resets by device-side indexed
  selection. The helper named `_make_pool_batch_cached` does not itself memoize
  its JIT wrapper; this matters mainly for repeated pool builds, not the current
  hot loop, so it was not changed.
- Masks/observations: both seats are generated under the scan on device. Legal
  masks are concatenated for one `2N` policy forward; no host reconstruction was
  found.
- Recurrent copies: both observation memories remain in the scan carry and are
  cleared with device-side `where` on episode boundaries. No host recurrent-state
  copy was found.
- PPO/allocations: GAE is a reverse `lax.scan`; PPO supports static gradient
  accumulation but production uses the full logical batch. Repartitioning it
  could change floating-point reduction order and needs separate equivalence and
  memory evidence, so it is not part of PERF_V1.
- Telemetry/checkpoints: heartbeat telemetry is update-level and checkpoints are
  milestone-level. Checkpoint serialization is intentionally host-bound and not
  in the per-update hot path.

## 768x32 and 1024x32

Neither geometry was tested locally or on the A100. Their feasibility cannot be
established from source inspection alone because PPO activation/gradient memory,
not just stored trajectory bytes, determines peak VRAM. Probe `768x32` first in
the controlled milestone window with identical seeds and measurement policy.
Only consider `1024x32` if `768x32` leaves verified headroom. Do not infer safety
from the A100's nominal 80 GB.

## Recommendation

Keep production unchanged. At a COMPLETE milestone, benchmark frozen baseline
and PERF_V1 identically within the approved 10–15 minute interruption. Promote
only if the sustained gain meets the existing threshold and health metrics,
VRAM, compilation behavior, and checkpoint compatibility remain valid.
