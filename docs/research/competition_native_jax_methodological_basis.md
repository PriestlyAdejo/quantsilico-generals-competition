# Competition-native JAX methodological basis

QuantSilico's pre-overnight rebuild uses:

- Official `GeneralsEnv(mode="competition")` as rules authority
- Full legal action support (PASS + 8 moves/cell + BUILD/cell)
- Exact castle economics and Deathtouch/1200 horizon features
- Deterministic observation memory (not opaque RNN as primary)
- Transformer with 3×3 patch tokens unpatchified to cells
- Symmetric self-play with shared raw parameters
- Masked PPO with collection/update support identity
- HL-Gauss value distribution; EMA for evaluation/selection
- NumPy CPU deployment path preferred over packaging jaxlib
- End-to-end JAX GPU training as the *intended* hot path when CUDA JAX is available

Default policy regularizer: `entropy_only`. Advantage filters and reverse-KL are ablations only.
