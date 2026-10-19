# artifacts

Small derived tables that features are computed from. Each one can be rebuilt from your own logs.

| File | Contents | Built by |
| --- | --- | --- |
| `deal_in_table_v0.json` | Deal-in rate against a riichi, bucketed by tile class (honor, 1/9, 2/8, 3/7, 4/6, 5), safety class (genbutsu / suji / no suji) and turn (1–6, 7–12, 13+). Counted on Phoenix-room mjai logs; it feeds the defence channels of the v5 observation | [`scripts/build_deal_in_table.py`](../scripts/build_deal_in_table.py) |
