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

## Round 0 result (2026-10-20 07:10–07:44, 5060 Ti, nothing else on the GPU)

| layout | games/s | vs b0 | identical games | readout shift | GPU util | GPU memory | CPU load (20 threads) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b0 1 × 400 (today's) | 3.72 | 1.00 | 1600 / 1600 | 0 | 40% | 1.2 GB | 5.0 |
| b1 2 × 400 | 4.96 | 1.33 | 1600 / 1600 | 0 | 58% | 1.8 GB | 10.9 |
| **b2 4 × 400** | **5.46** | **1.47** | **1600 / 1600** | 0 | 65% | 2.8 GB | 22.4 |
| b3 1 × 1600 | 4.76 | 1.28 | 0 / 1600 | −0.03 pt | 49% | 1.7 GB | 5.1 |
| b4 2 × 800 | 5.43 | 1.46 | 0 / 1600 | −0.06 pt | 61% | 2.0 GB | 12.4 |

- **Determinism holds across days.** b0 reproduces the 10-15 logs game for game, q_values included.
- **Kept: b2.** It gives 1.47× the throughput with every game identical, so a 100k drops from about 7.5 to about
  5.1 hours. The online transfer test's evaluation (`xfer_eval.sh`) uses it from 10-20. The seeds are split by whole
  100-seed iterations, so every batch is the one a single process would play.
- **Rejected: wider batches.** b3 and b4 change every game: the batch width moves kernel choices, so near-tie decisions
  flip. They are also no faster than b2.
- **Next.** CPU load is now above the 20 threads, so the next candidates are:
  - per-process thread caps (OMP / torch threads);
  - 6 processes;
  - pinned host memory in `MortalEngine`.

  They run in the next GPU gap.

## Round 1 result (2026-10-21 17:26–18:05, `bench_round1.sh`, 5060 Ti, nothing else on the GPU)

Each candidate is one change against b2 (4 × 400). b2r repeats b2 to measure the timing spread a candidate must beat.

| layout | games/s | vs b2 | identical games | GPU util | GPU memory | CPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b2r: repeat of b2 | 5.42 | 0.99 | 1600 / 1600 | 65% | 2.8 GB | 21.0 |
| b5: `RAYON_NUM_THREADS=5` | 5.68 | 1.04 | 1600 / 1600 | 69% | 2.8 GB | 11.3 |
| b6: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | 5.45 | 1.00 | 1600 / 1600 | 69% | 2.8 GB | 26.0 |
| **b7: both caps** | **5.78** | **1.06** | **1600 / 1600** | 68% | 2.8 GB | 10.6 |
| b2w: b2 on 800 groups | 5.58 | (base) | 3200 / 3200 | 69% | 2.9 GB | 21.7 |
| b8: 8 × 400 on 800 groups | 5.67 | 1.02 vs b2w | 3200 / 3200 | 59% | 4.9 GB | 42.9 |

- **Spread.** b2 and its repeat differ by 0.04 games/s (0.7%).
- **Kept: b7.** It gives 1.06× b2, about 1.55× b0 (today's layout before 10-20). Every game stays identical, since
  thread caps do not change any batch.
  - libriichi's arena runs its games on a rayon pool that defaults to all 20 threads in each of the 4 processes.
    Capping it at 5 per process halves the CPU load and is the useful part.
  - torch's own CPU threads matter little on their own.
- **Rejected: 8 processes.** They are only 1.6% faster than 4 on the same 800-group workload. They also use 4.9 GB of
  the card's 8 GB, and the load reaches 42.9 on 20 threads.
- **Adopted.** b7 runs in `xfer_eval.sh` and in the early-end-gap generation (`patches/early_end_gap`) from 10-21. A
  100k drops from about 7.5 hours (b0) to about 4.8.
- **Budget used.** 11 of 12 candidates in about 80 GPU-minutes. CUDA-graph capture and pinned memory would need
  engine changes for at most the GPU-side share, while GPU utilisation is already 68%. They wait until throughput
  matters again.

## X1: inference architecture on the real card (2026-11-04 02:24–02:46, `x1_5060.sh`, 5060 Ti, nothing else on the GPU)

The variants were developed on trx (`research/2026-11-03_rl_infra_survey`, section 8, gitignored) and timed here on
the b7 workload.

**Single process, 400 games, timed variant:** 2.97 games/s.
- Inference takes 58% of the wall time (35% on trx).
- Of that, the forward pass is 82–86%, the host-to-GPU copy 8–9% and stacking 4–8%.

**4 × 400 (b7):**

| Variant | games/s | vs base | Games | GPU util |
| --- | ---: | ---: | --- | ---: |
| base | 5.75 | (base) | hash 756a3285 | 69% |
| base2: repeat | 5.71 | 0.99 | identical | 69% |
| pinned: pinned host buffers + async copies | 5.70 | 0.99 | identical | 69% |
| free: one inference server, 4 arena clients | 5.98 | 1.04 | not reproducible | 64% |

- **Free server batching.** It merges 183 decisions per forward for v4 and 76 for the challenger. That buys only 4%.
- **Spread.** base and its repeat differ by 0.6%.
- **CUDA graphs.** They were left out. On trx they were slower than base at 4 × 400. They also reserve 5.6–9.2 GiB per
  process, more than four processes can have on this 8 GB card.
- **Verdict: nothing adopted.** The adoption bar was 1.3×. As on trx, the inference architecture is worth at most a
  few percent: four processes already overlap one process's inference with another's CPU work.
- **Where the time goes.** The remaining cost is CPU work in the arena. The likeliest part is the v4 'sp' feature
  encoding, which libriichi itself marks as CPU-intensive; it has not been profiled yet.
