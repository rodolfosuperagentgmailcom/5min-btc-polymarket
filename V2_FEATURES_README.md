# Building replay datasets

For one resolved market directory:

```bash
python -m btc5m_v2.research.replay_v2 runtime/data/YYYY-MM-DD/btc-updown-5m-...
```

For all resolved markets under the recorder root:

```bash
python -m btc5m_v2.research.replay_v2 runtime/data --root
```

Use `--allow-unresolved` only for feature inspection; unresolved rows intentionally have no terminal outcome label.

The replay builder reads immutable `clob_snapshots_part-*.parquet` files, builds causal CLOB features, then attaches the terminal resolution label in a separate pass.
