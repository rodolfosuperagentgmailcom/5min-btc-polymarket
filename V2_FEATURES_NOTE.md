# V2 Replay Feature Layer

The replay feature layer is intentionally split into two stages:

1. Build causal features using only market data available at each snapshot timestamp.
2. Attach terminal UP/DOWN resolution labels only after feature generation.

This separation is designed to reduce accidental future leakage between research and live/paper code paths.

Primary CLOB-derived features include midpoint, normalized midpoint share, top-three depth imbalance, combined bid/ask cost, selected-side book quality, receive latency, quote age, and 1/5/15-second causal midpoint deltas and velocities.

Chainlink settlement-aligned BTC features remain a separate feed layer and must not be synthesized from Polymarket prices.
