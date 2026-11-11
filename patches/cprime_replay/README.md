# P1 short: the C′ jump step replayed from its own start

Pre-registered 2026-11-07, before the run. 5060 Ti. Chosen by the user on 11-07, after P2 phase 1 missed its gate.

## Why

- **The jump is real.** The C′ re-check ([cprime_recheck](../cprime_recheck/README.md)) measured C′ final − C392k
  = +0.98 ± 0.63 (z 3.1) on 13,000 new groups, and +1.08 ± 0.56 on all 16,000. It is the largest single-step gain in
  the project.
- **It ran once.** It ran on trx on 07-23/24, and nobody has run it again.
  - Four rebuilds on the replay line never reproduced it, but they started elsewhere.
  - Two continuations from C′ final's own endpoint on 09-13 went backwards, one of them with July's seven-worker pool
    restored.
- **The question:** retrained from the same start with the same recipe, on our current pipeline, does the step give
  the gain again?
  - Yes: the operator works here, and the remaining problem is how to grow a start like rl1_best.
  - No: the July step was a training draw, or it depends on something our pipeline does not have.

## Design

Arm cr (`runs/xfer_cr`, port 5598). Everything below is copied from the July run's own files on trx.

| | July C′ (trx, 07-23/24) | Replay |
| --- | --- | --- |
| Start | C392k, step 215,800, full state (Adam moments, scheduler) | the same file, md5 7a679fb4 |
| Anchor and base opponent | rl1_best (`BASE.pth`, md5 b9c0ff1f) | the same file |
| λ | 0.5 up to step 240,000, then 0.2 up to 278,600 | the same steps. The switch is read from the trainer log: weight 0.2 from its restart at 240,000 |
| LR | 5e-6 in effect. The config said 1e-5, but the scheduler restored from rl1's state kept 5e-6, the LR trap | 5e-6, the scheduler state unchanged; the config is set to 5e-6 so that nothing contradicts it |
| Reward | [6, 4, 2, 0] | the same |
| Boltzmann temperature / ε | 0.1 / 0.005 (the trainer's embedded config) | the same |
| Batch, accumulation, capacity, reuse rate, submit | 512, 8, 4,000, 0, 400 | the same |
| Pool spec | base,base,v4,snap over 7 workers | base,base,v4,C392k over 4 workers |

**The config check.** The generated config may differ from the replay control v11s2's only in pts, peak, final and,
during the first phase, the anchor weight. The chain checks this.

### Departures from the July run (process rule R8)

1. **Workers and freshness.**
   - **July:** 7 GPU workers on trx. The single-process loader held the trainer at about 1.46 batches/s. This gives
     about 2.5 game files per batch, from the trainer log's drains (0.40 batches per file), and about 1.3 uses per
     fresh sample, assuming about 162 trainee samples per file as in our runs.
   - **Here:** the standard harness, with 4 GPU workers on the 8 GB card, an 8-process loader, about 1.8 files per
     batch and reuse about 1.8.
   - **Why not match it:** there is no room for more workers. Slowing the trainer would add 3 to 5 hours. And the
     freshness test (10-29) found no gain from reuse 1.0 over 1.8 on the control's stage (−0.56 ± 0.89).
2. **Pool composition.**
   - July's spec, cycled over 7 workers, gave 4 base, 2 v4 and 1 snap. This was fingerprinted on 09-13.
   - Over 4 workers it gives 2, 1 and 1, i.e. 50/25/25 instead of 57/29/14.
   - July's snap was the trainee at its 07-23 16:31 launch, about step 203k, in the same λ 0.5 phase. That file was not
     kept. C392k stands in for it, as in the 09-13 restoration.
3. **Buffer.** The buffer starts empty: the trainer waits for 4,000 games from C392k's policy. July's stream simply
   continued.
4. **test_play** uses 800 games against July's 2,000. It is only the noisy safety reading, and the trainer restarts at
   each test_play boundary in both runs.
5. **Code.** `train.py` 67bce509 is trx's July trainer, ba587de4 (unchanged since 07-22 at the 09-13 check), plus the
   M1 calibration options. The options are off by default, and the diff only adds them.
6. **Hardware.** RTX 5060 Ti against RTX 6000 Ada, with AMP on in both. Rounding differs.
7. **The training draw.** The games differ. The control rerun put one stage's spread at ±0.55.

## Readouts and verdict

**Evaluation:**
- 12k of CR240k (the end of the λ 0.5 part);
- the 16,000 groups of CR278.6k (seeds 10000–25999), the groups on which C392k and C′ final were measured in P0.

The primary is **CR278.6k − C392k on the 16,000 groups**. The historical step measured +1.08 ± 0.56 there.

| Reading | Outcome |
| --- | --- |
| ≥ +0.7 with z ≥ 2 | **Reproduced.** The step works on this pipeline from this start |
| ≤ +0.3 | **Not reproduced.** The July step does not repeat here |
| In between | Undecided at this size |

**Also reported:**
- the same pairing on the 13,000 new groups only;
- CR278.6k − C′ final, − rl1_best, − tpt 408k and − U40k, and vs v4;
- on 12k, the λ 0.5 and λ 0.2 parts (CR240k − C392k, CR278.6k − CR240k);
- the mechanism of CR278.6k beside C392k's and C′ final's. In July the relaxation took riichi from +1.98 to +0.58 pp,
  deal-in from +0.78 to +0.31 and defence from −57 to −30.

**Prior:** about 35%. The step is real, but the two continuations from its endpoint went backwards, and the gain may
depend on the data stream.

**Cost:**
- training: 62,800 batches, about 6.5 hours;
- evaluation: 12k plus 16,000 groups, about 4 hours.

## Files

- `cr_chain.sh` writes the log `runs/xfer_tools/cr_chain.log`, with the markers `CR_CHAIN_DONE` / `CR_CHAIN_STOPPED`.
- Results go to `cr_final.txt` and `cr_mech.txt`.
- It rewrites `GPU_STATUS_mortal.txt` and lends the idle card for a day after it ends.
- Two shared harness options were added for this arm. Their defaults leave every other arm unchanged:
  - `xfer_arm.sh` takes `POOL`;
  - its archiver also snapshots at `STOP_AT` (`ALSO`), because 278,600 is off the 8,000 grid.

## Successor: the same step from U40k (pre-registered 2026-11-07, before the replay's result)

Chosen by the user on 11-07. It runs only if the replay's verdict above is **reproduced** or **undecided**. If the
replay does not reproduce, or its chain stops, nothing is trained and the card waits for the user.

**Why.** P2 phase 1's trajectory ([unanchored_first_stage](../unanchored_first_stage/README.md)) showed that our
unanchored stage from v11 does grow a milder rl1_best. U40k, the snapshot July's best.pth rule picks, has attack +20,
riichi +1.40 and −1.61 against v4. This arm applies the replay's exact step to it. It is the whole path on our own
pipeline: v11 → unanchored stage → anchored repair.

Arm cu (`runs/xfer_cu`, port 5599):

| | Replay (cr) | This arm (cu) |
| --- | --- | --- |
| Start | C392k, step 215,800 | U40k, step 40,000 (md5 b6fefbe2), full state, scheduler at 5e-6 |
| Anchor and base opponent | rl1_best | U40k itself |
| λ 0.5, then λ 0.2 | 24,200 batches, then 38,600 | the same: to 64,200, then to 102,800 |
| Reward, LR, temperature | [6, 4, 2, 0], 5e-6, 0.1 | the same; the config check is the same |
| Pool | base, base, v4, C392k | base, base, v4, base: the start takes the snap seat, and here the start is the base |

**Departures beyond the replay's own list:**
- **The λ 0.5 history.** In July, λ 0.5 ran about 180,000 batches from rl1_best, but the policy barely moved: decision
  agreement with rl1_best was within 0.1 pp. This arm has only the replay's last 24,200 of them.
- **The pool** is 3 base (U40k) to 1 v4, against the replay's 2 rl1_best, 1 v4 and 1 C392k.

**Primary: CU102.8k − U40k on the 16,000 groups.**

| Reading | Outcome |
| --- | --- |
| ≥ +0.7 with z ≥ 2 | **Holds:** our own pipeline reproduces the jump path from v11 |
| ≤ +0.3 | **Fails** |
| In between | Undecided at this size |

**Also reported:**
- the replay's and history's steps beside it;
- CU102.8k vs v4, and − CR278.6k, − C′ final, − the pooled control and − tpt 408k;
- the λ 0.5 and λ 0.2 parts on 12k;
- the mechanism of CU102.8k, U40k, CR278.6k and C′ final.

**Prior:** about 40% if the replay reproduces. The start is milder than rl1_best, so there is less of a defect to
repair.

**Cost:** about 6.5 hours of training and 4 hours of evaluation.

`cu_chain.sh` writes the log `runs/xfer_tools/cu_chain.log`. Its markers are `CU_CHAIN_DONE`, `CU_CHAIN_STOPPED` and
`CU_CHAIN_SKIPPED`, and its results go to `cu_final.txt` and `cu_mech.txt`.

## Result (2026-11-07 21:43): not reproduced

Training ran as pre-registered:
- 215,800 → 240,000 at λ 0.5, then 240,000 → 278,600 at λ 0.2;
- 38,800 batches in the λ 0.2 part on 11.2 M fresh trainee samples, reuse 1.77;
- the trainer log read "anchor enabled: weight=0.2" from 240,000;
- no guard fired; test_play stayed between −3.3 and +0.8.

**Strength** (16,000 groups, seeds 10000–25999, unless marked 12k):

| Pairing | Mean ± 95% CI | z |
| --- | ---: | ---: |
| **CR278.6k − C392k (primary)** | **+0.07 ± 0.54** | +0.25 |
| the same, 13,000 new groups only | +0.09 ± 0.60 | +0.29 |
| history: C′ final − C392k | +1.08 ± 0.56 | +3.78 |
| CR278.6k − C′ final | −1.02 ± 0.59 | −3.40 |
| CR278.6k − rl1_best | +0.30 ± 0.55 | +1.07 |
| CR278.6k − tpt 408k | −1.41 ± 0.66 | −4.19 |
| CR278.6k − U40k | −0.33 ± 0.70 | −0.92 |
| CR278.6k vs v4 | −1.93 ± 0.58 | −6.53 |
| 12k: CR240k − C392k (λ 0.5 part) | +0.24 ± 1.15 | +0.41 |
| 12k: CR278.6k − CR240k (λ 0.2 part) | −0.26 ± 1.22 | −0.41 |

**Verdict: not reproduced** (+0.07 ≤ +0.3). The successor from U40k was therefore skipped (`CU_CHAIN_SKIPPED`, 21:45),
as pre-registered.

**Mechanism** (against the v4 seats in the same games; per kyoku, pp):

| | call (v4) | deal-in | riichi | net (attack / defence) | bust (v4) | 4th |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| C392k, the start (16k) | 29.10 (30.43) | +0.78 | +1.98 | −32 (+25 / −57) | 7.65 (6.66) | +2.35 |
| CR240k (12k) | 29.05 (30.55) | +0.69 | +2.04 | −31 (+24 / −55) | 7.51 (6.49) | +2.27 |
| **CR278.6k, the replay (16k)** | 29.05 (30.45) | **+0.86** | **+2.23** | −30 (+32 / −61) | 7.88 (6.67) | +2.39 |
| C′ final, July (16k) | 30.02 (30.42) | +0.31 | +0.58 | −18 (+12 / −30) | 7.44 (6.72) | +1.41 |

**Reading:**
- **The replay's policy did not move.** In July the same 38,600 batches at λ 0.2 moved C392k's profile to C′ final's:
  riichi −1.4 pp, deal-in −0.5 pp, defence +27. The replay stayed on C392k's profile, a little more aggressive if
  anything. So this is not a smaller gain: the July change did not happen at all.
- **The trainer side matches July.** I checked the configs embedded in the four states (C392k, C′ final, CR240k,
  CR278.6k) key by key. Apart from paths and the port, they differ only in:
  - the loader workers, 0 in July and 8 here; the loader shards files across workers, so the samples are the same;
  - test_play games, 2,000 against 800;
  - the config's LR lines, while the LR in effect is 5e-6 in both: optimizer lr 5e-6, scheduler peak/final 5e-6.
  The anchor file and weight schedule match.
- **What is left.** The causes left are the data stream, the hardware, or a training draw:
  - the data stream: 7 workers and reuse ~1.3 in July, against 4 and 1.8 here; the pool mix 4 : 2 : 1 with a ~203k
    snapshot, against 2 : 1 : 1 with C392k;
  - the hardware: RTX 6000 Ada against the 5060 Ti;
  - a training draw: the control rerun put one stage's spread at ±0.55. A whole-profile shift in one draw and none in
    the other is a large gap for chance, but chance is not excluded.
- **Self-play readings** (trainee pt per 800-game session in the λ 0.2 part, about ±0.6 per worker):
  - against rl1_best: July +0.38 over four workers, the replay −0.41 over two;
  - against v4: July −1.6, the replay −2.1.
  They point the same way, but within noise.
- **Consequence:** the anchored-repair step does not transfer to our pipeline as is. The route "grow rl1_best, then
  repair it" has no repair step we can run.

## Second round: a second draw here, and July's topology on trx (pre-registered 2026-11-07, before either runs)

Chosen by the user on 11-07, after the result above. The first replay leaves three explanations: July's data stream,
July's hardware, or a training draw. Two arms run at once, and every snapshot is evaluated on the 5060 Ti, the device
of all the other readings in this comparison.

| | First replay (cr) | Second draw (cr2) | July topology (ct) |
| --- | --- | --- | --- |
| Machine | 5060 Ti | 5060 Ti | trx, RTX 6000 Ada (July's card) |
| Workers, loader processes | 4, 8 | 4, 8 | **7, 0** (July's) |
| Pool | base, base, v4, C392k | the same | base, base, v4, snap over 7: 4 : 2 : 1 (July's) |
| test_play | 800 games | 800 | **2,000** (July's) |
| λ switch at 240,000 | whole run stopped and restarted, buffer emptied | the same | **config edited before the trainer's own respawn; workers and buffer go on** (July's) |
| Self-play seeds | random per worker | new random draws | new random draws |

Everything else is the first replay's: C392k's full July state, anchor rl1_best, LR 5e-6, reward [6, 4, 2, 0],
temperature 0.1, the same step counts and the same evaluation.

**The trx arm's config** is written from the July config embedded in C392k's state (`trx/ct_setup.py`). It is then
checked key by key against the configs embedded in C392k and C′ final. The only differences allowed are the run
directory, the port and the LR lines; the LR in effect is 5e-6 in both runs.

**Low priority on trx** (the user's terms):
- The arm uses the trx GPU only while nobody else does. While it does, it keeps `~/GPU_BUSY_mortal_lowprio`, which
  says it is low priority and may be interrupted at any time.
- It yields within about a minute to any other `~/GPU_BUSY_*` file or any GPU process that is not its own. It resumes
  from `state.pth` after the GPU has been free for 10 minutes.
- Each interruption is logged. A resume starts with a fresh buffer, as the first replay did at 240,000.
- It gives up after 72 hours.

**Departures of the trx arm from July (R8):**
1. **The start of the stream.** The buffer starts empty, while July's stream continued from its launch at about 203k.
   The snap seat is C392k in place of July's ~203k snapshot, which was not kept.
2. **Code.**
   - `train.py` 67bce509 is July's ba587de4 plus the M1 options, off by default.
   - `dataloader.py` skips unreadable files: July crashed once on a missing drain file at 237,400 and went on.
   - `server.py` and `client.py` match the 5060 Ti's; July's versions were not recorded.
   - libriichi is the same build on both machines.
3. **Interruptions**, if any, as above.
4. **The training draw.**

**Primaries.** CR2_278.6k − C392k and CT278.6k − C392k on the 16,000 groups, with the same thresholds:
≥ +0.7 with z ≥ 2 means reproduced, ≤ +0.3 means not reproduced, in between is undecided.

**Reading, set in advance:**

| cr2 | ct | Reading |
| --- | --- | --- |
| not | reproduced | July's topology or card matters: the data stream (workers, freshness, pool) or the hardware, to be split next |
| not | not | With three misses, two here and one on July's own topology and card, the July step was most likely a draw. The C′ line closes. |
| reproduced | either | The step can happen on our pipeline. It is a draw with a large miss rate, and how often it hits becomes the question |
| undecided cases | | reported as they are, with CT − mean(CR, CR2) as the topology contrast |

**Also reported:**
- CR2 − CR, the spread of two draws;
- CT − mean(CR, CR2);
- − C′ final, vs v4, the λ parts on 12k;
- the mechanism of all three arms beside C392k's and C′ final's. "The policy moved" means riichi fell by at least
  0.7 pp from C392k's +1.98; July's fell by 1.4.

**Priors:** cr2 reproduces about 15%, ct about 30%.

**Cost and timing (if trx is not interrupted):**
- cr2 trains for about 7.5 hours, and its evaluation takes about 4.5;
- ct trains for about 13 hours on trx; it is evaluated on the 5060 Ti after cr2, in about 4.5 hours;
- results are expected around 16:00 on 11-08.

**Files.**
- `cr2_chain.sh` (5060 Ti): log `runs/xfer_tools/cr2_chain.log`, markers `CR2_CHAIN_DONE` / `CR2_CHAIN_STOPPED`, results
  `cr2_final.txt` and `cr2_mech.txt`. It also evaluates ct once its archives arrive, waiting up to 48 hours.
- `trx/ct_chain.sh` (trx, `runs/cprime_trx`): log `ct_chain.log`, markers `CT_CHAIN_DONE` / `CT_CHAIN_STOPPED`. It
  writes `CT_READY` with the archives' md5.
- A Mac relay copies the archives to `runs/xfer_ct/archive` on the 5060 Ti and adds `READY`, or `STOPPED`. The two
  machines cannot reach each other directly.

## Result of the second round (2026-11-09 13:01): neither arm reproduced

**Training.**
- **cr2** ran as pre-registered on the 5060 Ti and reached 278,600 at 06:44 on 11-08, with reuse 1.76. Its config
  equalled the first replay's at both weights.
- **ct** ran on trx with July's topology and reached 278,600 at 01:30 on 11-09. It yielded four times, each logged:
  - to another `GPU_BUSY_` file at 00:59 on 11-08;
  - to a foreign GPU process at 02:01;
  - to `GPU_BUSY_omni1108` at 02:59, and again at 10:38 while starting;
  - it resumed for good at 12:45 on 11-08.
- The λ 0.2 part ran without a break: the weight was written at 239,800 and read at the trainer's 240,000 respawn.
  Over all 21 trainer starts, none read the wrong weight.
- ct's reuse was 1.37 per drain from 228k, the same 9,600 files per 4,000 batches as July; 1.18 over the whole run.
  Its trainer ran at 1.32–1.37 batches/s, against July's 1.46.

**Strength** (16,000 groups, seeds 10000–25999, unless marked 12k):

| Pairing | Mean ± 95% CI | z |
| --- | ---: | ---: |
| **CR2_278.6k − C392k (primary, second 5060 Ti draw)** | **−0.12 ± 0.55** | −0.44 |
| **CT278.6k − C392k (primary, trx with July's topology)** | **−0.15 ± 0.53** | −0.55 |
| the same, 13,000 new groups only: CR2 / CT | −0.30 ± 0.61 / −0.32 ± 0.59 | −0.97 / −1.07 |
| first replay: CR278.6k − C392k | +0.07 ± 0.54 | +0.25 |
| history: C′ final − C392k | +1.08 ± 0.56 | +3.78 |
| CR2 − CR (two draws on the same machine) | −0.19 ± 0.59 | −0.64 |
| CT − mean(CR, CR2) (topology and card) | −0.12 ± 0.48 | −0.49 |
| CR2 − C′ final / CT − C′ final | −1.21 ± 0.60 / −1.23 ± 0.59 | −3.92 / −4.11 |
| CT − tpt 408k | −1.62 ± 0.66 | −4.85 |
| CT − rl1_best | +0.08 ± 0.54 | +0.31 |
| CR2 vs v4 / CT vs v4 | −2.13 ± 0.59 / −2.15 ± 0.58 | −7.12 / −7.27 |
| 12k: CR2_240k − C392k (λ 0.5 part) | +1.24 ± 1.16 | +2.10 |
| 12k: CR2 − CR2_240k (λ 0.2 part) | −0.60 ± 1.29 | −0.91 |
| 12k: CT240k − C392k (λ 0.5 part) | −0.27 ± 1.18 | −0.45 |
| 12k: CT − CT240k (λ 0.2 part) | +0.87 ± 1.23 | +1.38 |

**Verdicts:** CR2 not reproduced, CT not reproduced (both ≤ +0.3).

**Mechanism** (against the v4 seats in the same games; per kyoku, pp):

| | call (v4) | deal-in | riichi | net (attack / defence) | bust (v4) | 4th |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| C392k, the start (16k) | 29.10 (30.43) | +0.78 | +1.98 | −32 (+25 / −57) | 7.65 (6.66) | +2.35 |
| CR278.6k, first replay (16k) | 29.05 (30.45) | +0.86 | +2.23 | −30 (+32 / −61) | 7.88 (6.67) | +2.39 |
| CR2_278.6k, second draw (16k) | 28.38 (30.44) | +0.93 | +2.95 | −25 (+38 / −64) | 7.79 (6.73) | +2.66 |
| CT278.6k, trx July topology (16k) | 29.05 (30.44) | +0.96 | +2.07 | −37 (+29 / −66) | 7.83 (6.66) | +2.42 |
| C′ final, July (16k) | 30.02 (30.42) | +0.31 | +0.58 | −18 (+12 / −30) | 7.44 (6.72) | +1.41 |
| CR240k / CR2_240k / CT240k (12k) | 29.05 / 28.44 / 28.91 | +0.69 / +0.52 / +0.52 | +2.04 / +1.87 / +1.09 | −31 / −14 / −38 | | +2.27 / +1.89 / +2.48 |

**Reading** (the pre-registered row "not / not"):
- **The July step was most likely a draw, and the C′ line closes.**
  - The step has now been drawn four times from C392k's state: July's run and three replays. Only July's jumped.
  - The three replays agree with each other: −0.12, −0.15 and +0.07, all within the ±0.55 spread of one stage.
- **July's topology and card add nothing measurable.** CT − mean(CR, CR2) = −0.12 ± 0.48. ct had July's 7 workers,
  loader, test_play, reuse, pool mix, λ switch and card.
- **No replay's policy moved toward C′ final.**
  - Riichi stayed at +2.07 to +2.95 against C392k's +1.98, so the "moved" criterion (a fall of 0.7 pp) failed in all
    three. July's fell to +0.58.
  - Defence got slightly worse in all three (−61 to −66 against −57), where July's improved to −30.
  - ct's λ 0.5 snapshot read riichi +1.09 on 12k, but the λ 0.2 part brought it back to +2.07.
- **What is left from July cannot be rerun:** the buffer that the July stream carried into 215,800, and the ~203k
  snapshot in the snap seat. Both belong to that run's past, not to its recipe.
- **Consequence for the project:**
  - The record lineage, C′ final and then tpt 408k, rests on a step that our pipeline did not repeat in three tries.
  - One hit in four draws gives a hit rate between about 1% and 81% (95%, exact), so rerunning until it hits is not a
    recipe.
  - The route "grow rl1_best, then repair it with an anchor" has no repair step we can run.

**Operations: the relay was lost.** The Mac relay ran as a background task of the session and died with it. ct
finished on trx at 01:30, but its archives reached the 5060 Ti only at 09:06, after a manual relay. The 5060 Ti sat
idle for those 7.5 hours. A relay that matters should run on a machine that stays up, or the consuming chain should
pull the files itself.
