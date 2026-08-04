# Arena configuration and running a match

Production Arena launches allowlisted local matches:

1. Select a registered candidate (default submitted heuristic v2).
2. Select an opponent (Expander / Hunter / heuristics).
3. Choose a deterministic seed.
4. Launch → receive persistent job ID.
5. Lifecycle: QUEUED → RUNNING → COMPLETED / FAILED.
6. Open the recorded replay when a replay ID exists.

Live board telemetry may be unavailable: the evaluator often records outcomes without streaming frames. That is not a failure by itself.
