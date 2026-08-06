# Competition-native JAX discrepancies

| Topic | External / older path | QuantSilico decision |
|-------|----------------------|----------------------|
| Rules | Standard Generals.io assumptions | Official competition simulator |
| Primary learner | PyTorch proposal-ranking Hybrid / invalid overnight PPO | Full-support competition_native_jax |
| Advantage filter | Absolute advantage filtering in some public code | Ablation only; default none/entropy path |
| Policy regularizer | Reverse-KL heuristics | Default `entropy_only` |
| Official experimental JAX PPO | WIP equinox example in generals-bots | Not copied; may inspire structure only |
| Naming | External bot/author names | Neutral `competition_native_jax` IDs only |
| Deployment | Train-time JAX | Do not assume jaxlib in package |

Official frozen packages and the competition simulator override paper/code when they conflict.
