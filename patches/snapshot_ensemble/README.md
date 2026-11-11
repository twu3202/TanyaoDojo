# Snapshot ensemble: is the mean-Q ensemble of our best models stronger than 408k?

Pre-registered 2026-10-29, before the run. 5060 Ti. Chosen by the user after the
[sample-freshness test](../online_freshness/README.md) closed.

## Question

Nothing trained past tpt 408k has beaten it:

- every variant from 408k was ≤ 0 paired against it at 100k (−0.13 to −0.40);
- the 5060 Ti replays of the v11 line level off about 1.1 below it;
- neither a better offline base nor a bigger network nor fresher samples helped.

The historical line's gains came from stages anchored to the current best. If our best models make partly independent
errors, the mean of their Q values plays better than any one of them. That needs no training, and a stronger
ensemble is a candidate anchor (teacher) for the next online stage, which would turn it into a single network.

## Design

- **Members.** The four best own models by the sacred 100k (the October table in [RESULTS.md](../../RESULTS.md)).
  Member 0 is 408k.

  | Member | 100k vs v4 |
  | --- | --- |
  | tpt 408k | −0.637 ± 0.458 |
  | Stage 1 448k (targeted exploration from 408k) | −0.765 ± 0.458 |
  | v4 pool 448k (408k trained on with a v4-heavy opponent pool) | −0.863 ± 0.462 |
  | s368000 (λ 0.1 anchor, old reward; 408k's predecessor) | −0.886 ± 0.461 |

  - All four are Mortal 192 × 40 networks with the same observation version.
  - None is derived from v4's weights; v4 appears only as an opponent in training.
- **Policy.** The action is the argmax of the mean of the four masked Q vectors. There is no temperature and no
  weighting, and nothing is tuned.
- **Engine check first.** The ensemble path with 408k as its only member must play seeds 10000–10099 (400 games)
  byte-identically to the single-model path, apart from the player name and each decision's wall-clock
  `eval_time_ns`. If it does not, the chain stops. The first run of the check compared the timing field too and
  stopped. With the field dropped, all 400 logs matched, Q values included.
- **Screening: E − 408k.**
  - SPRT 0 vs +1, α 0.025, β 0.2.
  - One look per 1,000 groups, in ascending seed order from 10000.
  - Cap 25,000 groups (100k games).
  - Paired with 408k's existing 100k on the same walls.

## Verdicts

- **Promote.** Play on to the full 100k (seeds 10000–34999), for E vs v4 and E − 408k on all of it. Then propose an
  online stage anchored to the ensemble. That step needs a trainer change and is the user's decision.
- **Drop.** The neighbourhood of 408k carries no complementary information that averaging can use.
- **Undecided at the cap.** Read the fixed 100k. A result of ≥ +0.5 with z ≥ 2 counts as a promote, ≤ 0 as a drop.
- **Reported as an ensemble.** An ensemble is not a single network. Its result is listed separately from
  single-model records.

## Also reported

- E vs v4.
- E − each member, paired on the same walls.
- The share of engine decisions on which E's action differs from 408k's.
- Mechanism readouts (`diag_gap` / `diag_place` / `diag_bust`) against 408k's.

## Result (2026-10-29): dropped at the first look

- **Engine check passed (12:58).** With 408k as the only member, the ensemble path played 400 games identical to
  the single-model path and chose differently on 0 of 62,385 decisions.
  - The first run of the check had compared the wall-clock `eval_time_ns` field too, and stopped.
- **The SPRT dropped at 1,000 groups**, s10000–10999, with LLR −2.18:

  | | 1,000 groups |
  | --- | ---: |
  | E vs v4 | −2.07 ± 2.19 |
  | E − 408k | −1.67 ± 1.96 (z −1.67) |
  | E − v4 pool 448k | −2.03 ± 2.06 |
  | E − s368000 | −2.01 ± 2.15 |

- **The ensemble changes almost nothing.** It chose differently from 408k on 8,240 of 608,023 engine decisions
  (1.36%). A +1-point effect from that few changes was never likely.
- **"E is worse" is not established either.** One thousand groups cannot tell −1.7 from noise around a small effect.
- **Reading by the pre-set rule: dropped.** The models around 408k are too alike for averaging to help.
- The default main-line continuation started at 13:19.

## Cost and follow-on

- **Cost.** The challenger's network runs four times per decision. That is estimated at 18 minutes per 1,000 groups,
  so at most about 8 hours to the cap.
- **Follow-on.** The default main-line continuation (`patches/online_transfer/mainline_next.sh`, from A240k).
- **Tools.**
  - `jax_rl/mjai_bot/run_eval.py --challenger-type ensemble` takes comma-separated checkpoints. The single-model
    paths are unchanged.
  - `ens_eval.sh` plays one segment with the same layout and acceptance as `xfer_eval.sh`.
  - `ens_chain.sh` runs the whole test.
