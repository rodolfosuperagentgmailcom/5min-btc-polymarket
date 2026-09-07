# Research layer

Research transforms must be causal: features may only use data observable at or before each snapshot timestamp. Terminal resolution labels are attached only after feature construction.

Current replay features are CLOB-only. Settlement-aligned BTC/Chainlink features belong in the Chainlink feed pipeline and must not be inferred from Polymarket prices.
