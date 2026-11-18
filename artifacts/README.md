# artifacts

Local analysis tables that can be regenerated from logs you have permission to use.

| File | Contents | Built by |
| --- | --- | --- |
| `deal_in_table_v0.json` (gitignored) | Deal-in rate against a riichi, bucketed by tile class, safety class and turn. Used for the v5 defence channels. | [`scripts/build_deal_in_table.py`](../scripts/build_deal_in_table.py) with `--out artifacts/deal_in_table_v0.json` |
