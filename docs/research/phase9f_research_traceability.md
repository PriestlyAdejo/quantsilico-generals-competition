# Phase 9F research traceability

Maps each major Phase 9F decision to the research principle motivating it,
the QuantSilico implementation, intentional exclusions, and deviations.

| Family | Principle | QuantSilico | Exclusion / deviation |
| --- | --- | --- | --- |
| AlphaStar/SCC | Imitation before RL; structured actions; league; recurrent memory | Teacher→BC→DAgger→teacher-anchored sync PPO; legal candidate ranking; opponent league; CNN+GRU | No TPU/transformer/full league roles |
| R2D2 | Sequence handling; burn-in; hidden lifecycle | Persistent actor hidden; episode windows; burn-in; truncated BPTT | On-policy PPO not off-policy DQN |
| IMPALA | Persistent actors; batched accelerator learning | Persistent CPU actors; CUDA batches; sync epochs | No uncontrolled async lag; no V-trace tonight |
| Option-Critic | Temporally extended behaviours | Persistent modes with initiation/termination/hysteresis | Explicit rules, not learned options |
| DAgger | Correct learner state distribution | ≤1 aggregation round; immutable dataset versions | Overnight-bounded |
| Reverse curriculum | Near success → harder starts | Ordered scenario ladder near-win→full game | No unstructured mixture |
| Potential shaping (Ng et al.) | Policy-invariant F_t=c(γΦ′-Φ); Φ(term)=0 | Curriculum arm only; no clip when claiming invariance | HEURISTIC_BOUNDED_SHAPING if clipped |
| Invalid-action masking | Optimise over valid actions | Pre-sample executable mask; candidate-set hash replay; >0.1% fail gate | — |
| AlphaGo search | Exact search where tactical | 1–2 ply forced defence/convert/DT/capture | No whole-game MCTS |
| OpenAI Five PPO | Recurrent PPO after competent start | Matched+overnight CNN-PPO | Laptop budget |
| Exploration | Strategic info | Frontier/belief reduction/stale refresh | No RND novelty |
| Imperfect-info | Compact belief | Visible belief + candidate-general constraints | No ReBeL/DeepNash |
