# common

Shared cores of the Sichuan line.

| File | Contents |
| --- | --- |
| `suit_table.py` | O(1) lookup tables for winning-hand and shanten checks. Sichuan has three numbered suits and no honors, so one suit's hand is 9 counts in 0..4, encoded as a base-5 index into a precomputed table. Verified against the recursive reference on 30k cases with zero mismatches. The `.npy` table is a build product (about 2 s to rebuild) and is not checked in |
| `shaping.py` | Potential-based reward shaping. Under a uniform random policy, 76.8% of Sichuan games end with all four scores at zero, so a cold start sees no gradient. Shaping adds signal without changing the optimal policy (the telescoping is checked numerically in `sichuan/check_shaping.py`) |
