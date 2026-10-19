# Evaluation throughput: the first automated-search object

2026-10-19. Every decision in this project is bought with sacred evaluations: about 7.5 hours per 100k on the 5060 Ti
(3.9 games/s). That noisy, expensive fitness is why automated research has nothing to stand on here (see the RSI
survey notes). Throughput is the opposite kind of object. Its fitness is cheap (minutes), nearly noise-free, and
directly useful: twice the games per hour halves the cost of every future readout. So it is the first component handed
to a keep/revert search.

## Fixed runner, fixed workload, fixed judge

- **Workload.** q25_final vs mortal_v4, seeds 10000–10399 (400 groups = 1,600 games), seed_key 20260711, GPU fp32 /
  AMP off. These are exactly the games of the q25 100k evaluation of 2026-10-15, whose logs are kept.
- **Judge** (`bench_compare.py`, outside anything a candidate changes):
  - wall-clock games/s;
  - identical games, i.e. decompressed mjai logs equal byte for byte to the reference;
  - the per-group placement-point difference to the reference;
  - GPU utilisation and memory, CPU load.
- **Admissible candidates.** A layout or implementation change is admissible only if all 1,600 games stay identical.
  The new engine can then be paired with every existing 100k dump; a "faster" engine that changes games would need
  all controls re-evaluated. The first runs test whether today's layout reproduces its own 10-15 logs. If it does
  not, identity is relaxed to the reference's own run-to-run agreement, stated with the number.
- **Not admissible.** Lower precision, fewer games, skipped validation, or anything touching the judge, the seeds or the
  logs it reads.

## Round 0 (`bench_chain.sh`): baseline layouts in the GPU gap after q12one

| name | processes | games per iteration (batch width) | expected effect |
| --- | --- | --- | --- |
| b0 | 1 | 400 | today's layout; run-to-run determinism vs 10-15 |
| b1 | 2 | 400 | same 100-seed batches, CPU game logic of one overlaps GPU work of the other |
| b2 | 4 | 400 | same, more overlap; GPU memory limit |
| b3 | 1 | 1,600 | wider batches; may change kernel choice, so identity is checked |
| b4 | 2 | 800 | between the two |

b1 and b2 keep every batch exactly as in the reference, so they should be identical by construction if the engine is
deterministic. b3 and b4 are allowed to fail identity; they tell whether batch width alone changes games.

## After round 0: the search loop

- **Loop.** The loop keeps one ledger, `runs/bench_eval/summary.txt`, of every candidate, including failures and
  timeouts. Each candidate is one change against the best admissible layout so far. It is kept only if it is faster
  by more than the measured run-to-run timing spread and still passes identity.
- **Candidate space** (in order):
  - process count and seed split;
  - CPU thread caps per process (OMP / torch threads);
  - pinned host memory and a single host-to-device copy per batch in `MortalEngine`;
  - CUDA-graph capture of the fixed-width forward pass.
  Each must pass identity.
- **Budget.** At most 12 candidates or 2 GPU-hours, whichever comes first. Each candidate runs in a gap between
  scheduled jobs, never alongside a training run.
- **Done when** the best layout is written into the evaluation chains and one full 12k with it matches the
  reference 12k of the same model game for game.

The same harness shape (fixed workload, identity judge) is meant for self-play generation next: the online workers'
games/s at an equal policy decides how much fresh data each online batch sees (2.0–2.3× reuse on the 5060 Ti against
1.3× on trx).
