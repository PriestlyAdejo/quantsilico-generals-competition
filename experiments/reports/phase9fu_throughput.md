# Throughput lane (non-blocking)

Status: documented only during Stage 3 paired evaluation (resource isolation).

- Competition track does **not** wait on Tier-1 TPS.
- Current repaired PPO smoke ≈ 4 valid learning transitions/sec — research-only.
- Target remains ≥ 100 valid PPO learning transitions/sec via vectorisation / batching / sync reduction.
- Pause throughput/PPO processes during competitive CPU/latency/paired measurements.
