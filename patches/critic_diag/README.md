# Critic upper-bound diagnostic: how much of the return can a critic explain?

Pre-registered 2026-11-09, before the run. 5060 Ti, about 2 GPU hours. Chosen by the user on 11-09, after the C′ line
closed ([cprime_replay](../cprime_replay/README.md)): prepare this and the four-seat self-play probe, and run this
first.

## Why

- The policy-gradient (PG) probe ([policy_critic](../policy_critic/README.md)) lost to the Q stage. Its critic explained
  under 10% of the return variance (explained variance, EV, 0.08–0.10), so its advantage was mostly noise.
- The 11-05 design audit (`research/2026-11-05_rl_design_audit`, gitignored) found the critic was its largest
  departure from convention:
  - a small head on detached policy features, where others use a separate network or a shared trunk with value
    gradients;
  - public information only, where imperfect-information agents usually give the critic the hidden state
    (a privileged critic): PerfectDou's value net sees the opponents' hands, and AlphaStar's ablation gives 82% against
    22%.
- **The question:** is 0.1 the ceiling that mahjong's noise sets, or was it our critic? If it was the critic, a PG v2
  with a conventional critic is worth a probe. If not, the PG direction closes.

## Data

The PG probe's own self-play games, `runs/xfer_p1`: 14,400 unique games left in its buffer, drain and worker folders.
- They were played at the end of its run (policy at about 152k–160k), sampled at T 0.1 with ε 1, so they follow the
  probe's behavior policy exactly.
- Duplicates across the three folders were removed by game identity; 60 sampled duplicate pairs were byte-identical.

Split by seed, so that games on the same wall never cross splits (md5 of the seed, salt `critic_diag_20261109`):

| Split | Games | Seeds | Use |
| --- | ---: | ---: | --- |
| test | 2,236 | 559 | the readout; about 300,000 trainee decisions |
| val | 596 | 149 | early stopping: each critic's best step is chosen on these |
| train | 11,568 | | plus 43,200 games of the other A80k stage-2 arms (C rerun, R, W), training only |

Each game gives about 134 trainee decisions; the training set is about 7.3 M decisions.

**The return** is the one the probe's critic and the Q loss regress on: the kyoku's GRP reward (pts
[2.39, 1.195, 0, −3.585], γ 1). So every decision in a kyoku has the same target.

## Critics

All start from P160k: its trunk (`mortal`), and its value head as trained online. All are trained on the same batches
with MSE to the return.

| Critic | Input | What trains | Corresponds to |
| --- | --- | --- | --- |
| head0 | public | nothing: P160k's online head on its own features | the probe as it ran |
| head | public | the head only, on P160k's frozen features | the probe's design, with more fitting |
| sep | public | a copy of the trunk plus the head | a separate public-information critic |
| shared | public | trunk, head and the policy head; the policy is held at P160k's by a KL loss (×10) | a shared trunk with value gradients |
| oracle | public + hidden | trunk with extra input channels, plus the head | a privileged critic |
| hands | public + opponents' hands | as oracle, with the wall rows zeroed | a PerfectDou-style privileged critic |

- **The hidden input** is Mortal's own oracle encoding (217 rows), rebuilt from the logs:
  - opponents' hands, red fives, shanten, waits and furiten (rows 0–50);
  - the remaining wall in draw order, the dead wall, dora and ura indicators (rows 51–216). Drawn tiles are exact; the
    positions nobody draws are a random fill, so they carry no information.
- The extra input channels start at zero, so oracle and hands start equal to sep. Any difference comes from the
  hidden input.
- A privileged critic stays an unbiased baseline: the hidden state is fixed before the action and does not depend on
  it.

**Training.** 15,000 batches of 512 (about 7.7 M decisions, about one pass over the training set), the same stream for
all critics.
- AdamW: LR 1e-4 for trunks, 1e-3 for heads. Warm-up 200 batches, then cosine to 10%. No weight decay, gradient clip 1.0.
- BN statistics frozen, as in every online stage; bf16 autocast.
- Validation every 1,500 batches and at the end. Each critic is scored on the test games at its own best validation
  step.

The budget comes from a speed test (2.18 batches/s for all six critics together, data wait 2% with 6 loader workers),
which printed no explained variance. The run uses 4 loader workers and 2 for evaluation, because the 6 workers left
only 2 GB of WSL's 15 GB free.

## Readouts

**Primary:** EV on the test games, 1 − Σ(G − V)² / Σ(G − Ḡ)², for each critic, with 95% bootstrap intervals over games
(2,000 resamples).

**Also reported:**
- the paired differences head − head0, sep − head, shared − sep, oracle − sep, hands − sep, oracle − hands;
- the residual-variance factor (1 − EV(head0)) / (1 − EV): how many times fewer samples a PG step would need for the
  same gradient noise;
- EV by decision (discard, riichi, calls, agari, pass) and by turn (junme 0–5, 6–10, 11–14, 15+);
- shared: KL to P160k's policy and greedy agreement on the test games, to show what holding the policy cost;
- each critic's validation curve. A critic whose curve still rises at its last look is flagged: its EV is then a
  lower bound.

**Replication check.** head0 on the test games should be near the probe's online reading, 0.095. A gap over 0.03 means
the offline setup differs from the online one, and is reported first.

## Verdicts, set in advance

The rule comes from the 11-05 audit.

| Result | Reading | Next |
| --- | --- | --- |
| EV(oracle) ≥ 0.40 | A privileged critic would cut the advantage's noise variance at least 1.5-fold | A PG v2 probe is worth designing: privileged critic, behavior log-probs logged at sampling, entropy bonus, return normalization. Its run needs the user's go |
| every critic < 0.20 | No critic we can build cuts the noise by more than 1.13-fold: mahjong's noise, not our critic, held the probe back | The PG direction closes |
| otherwise | The critic can be improved, but by under 1.33-fold | Reported as it is; PG v2 is not recommended by default; the user decides |

**Secondary readings:**
- sep − head ≥ +0.05: the detached head was a weak design even without hidden information.
- shared within 0.02 of sep, with KL under 0.02 and greedy agreement over 98%: a shared trunk can carry the value, and
  PG v2 does not need a second network.
- oracle − hands: how much comes from the wall rather than the opponents' hands. The hands critic is the conventional
  privileged critic; seeing the wall goes further.

**Priors:** EV(oracle) ≥ 0.40 about 40%; every critic under 0.20 about 25%.

## Departures (process rule R8)

| | Conventional, or the online probe | Here | Why, and the cost |
| --- | --- | --- | --- |
| 1 | The critic trains online, on-policy | Offline on fixed games; 79% of training games are from three Q-line arms (ε 0.005), not from the probe | The probe's 11,568 training games alone would be re-read many times. The other arms share its start A80k and its temperature; the mismatch adds a little bias to a mixture value. The test is on the probe's own games |
| 2 | A tuned learning rate | One budget and one LR per part, no sweep | Cost. Early stopping guards against overfitting; a curve still rising is flagged, and its EV is a lower bound |
| 3 | A critic trained from scratch or from a pretrained net | All from P160k's trunk and online head | What a PG v2 would do; from scratch needs far more data |
| 4 | Mortal: fp16 with a scaler, weight decay 0.1 | bf16 autocast, no weight decay | Simpler for a one-pass fine-tune; neither is expected to move EV |
| 5 | A shared trunk trained by the PG loss and the value loss | Value loss plus KL to P160k's policy | The PG loss cannot be run offline; the KL stands in for "the trunk must keep the policy" |
| 6 | A privileged critic sees the opponents' hands | oracle also sees the wall order | Mortal's encoder includes it. The hands critic is the conventional variant |
| 7 | Return normalization | None | EV does not depend on the return's scale |
| 8 | Readings over the whole run | Test games from the end of the probe's run only | Only those games were kept |

## Cost and timing

- About 1.9 hours of training, plus about 10 minutes of validation and testing. The card is marked BUSY with an ETA.
  Active bookings are served before the start and after the end.
- The report runs on the CPU.

## Files

- `critic_diag.py`, deployed as `Mortal/mortal/critic_diag.py`, with commands `split`, `smoke`, `bench`, `train` and
  `report`.
- `cd_chain.sh` (`runs/xfer_tools/cd_chain.sh`): log `cd_chain.log`, markers `CD_CHAIN_DONE` / `CD_CHAIN_STOPPED`.
- Outputs in `runs/critic_diag`:
  - `split.json`;
  - `train.log` and `val_curve.json`;
  - `best_critics.pth`;
  - `test_preds.npz` and `test_summary.json`;
  - `report.txt`.

## Glossary

| 中文 | English | Meaning here |
| --- | --- | --- |
| 评论家 | critic / value function | predicts the return from a state; the PG baseline |
| 解释方差 | explained variance (EV) | the share of the return's variance the critic predicts; 1 is perfect, 0 is useless |
| 特权评论家 | privileged / asymmetric critic | a critic that also sees hidden state during training; the policy never does |
| 优势 | advantage | return minus the critic's prediction |
| 共享主干 | shared trunk | one network body for both the policy and the critic |
| 早停 | early stopping | keeping the step where the validation score was best |
| 自助法 | bootstrap | resampling games to get an interval |

## Result of the first run (2026-11-09 19:57): a failed fit, no verdict

15,000 batches in 1.9 hours, as planned. Test: 359,130 trainee decisions in 2,236 games, return SD 0.678.

| Critic | EV on test | 95% CI | Best validation look |
| --- | ---: | --- | ---: |
| head0 | 0.098 | [0.090, 0.106] | (no training) |
| head | 0.098 | [0.091, 0.106] | 12,000 |
| sep | 0.008 | [−0.006, 0.021] | 13,500 |
| shared | 0.104 | [0.097, 0.113] | 12,000 |
| oracle | 0.010 | [−0.003, 0.024] | 13,500 |
| hands | 0.055 | [0.043, 0.067] | 15,000 |

- **Replication passes.** head0 reads 0.098 on the test games, against 0.095 online.
- **The three trunk critics failed to fit.** sep, oracle and hands started equal to head0 (0.085 on validation) and
  stayed below it at all ten looks; on test they end at 0.01–0.06. So they measured a broken fit, not a ceiling.
- **The cause is my learning rate.**
  - The probe trained its value online at LR 1e-5, with an optimizer step every 8 batches. This run used 1e-4 for the
    trunks with a step every batch: about 80 times more movement per sample. Departure 2 named "one LR, no sweep" but
    did not compare it with the probe's own LR.
  - The stream made it worse. A batch held decisions from about 16 games, about 13 per kyoku, all with the same target.
    The trunks memorized the kyokus in the stream: their EV on the training window, measured before each step, was
    0.15–0.26, and it fell as the LR annealed. The held-out EV stayed near zero.
  - Step 0 was not a validation look, so early stopping could not fall back to the start.
- **Why the pre-registered rule is not applied.** Read literally, "every critic < 0.20" holds, and the PG direction
  would close. But three of the six critics never beat their own start. Had step 0 been a look, they would all read 0.098
  and the rule would still say "close", without having measured a privileged critic at all.
- **What the run does show:**
  - head − head0 = −0.000 [−0.002, +0.001]. The probe's detached head was already fitted to its features; more fitting
    gives nothing.
  - shared is 0.007 above head0, at KL 0.045 to P160k and greedy agreement 96.3%. The secondary criterion (KL under 0.02,
    agreement over 98%) is not met.
  - hands − sep = +0.047 [+0.033, +0.062] under the same broken training. The opponents' hands carry value information
    that even this fit picked up. Adding the wall lost it again (oracle − hands −0.045): more input to memorize.
  - Every critic explains more late in the kyoku (head0: junme 0–5 0.05, 15+ 0.26), and agari decisions are mostly
    explained (0.71).

## Second run: pre-registered 2026-11-09 20:15, before it ran

Same data, split, test games, return and readouts as the first run. `critic_diag2.py`, outputs in `runs/critic_diag2`,
chain `cd2_chain.sh` (log `cd2_chain.log`).

**What changes:**
- **Learning rates:** trunks at 1e-5, the probe's own; heads at 1e-4.
- **A real shuffle buffer:** each loader worker keeps 6,000 decisions (about 45 games) and yields at random from it.
  A batch of 512 then holds about one decision per kyoku, instead of about 13.
- **Step 0 is a validation look,** so every critic can fall back to its start. Looks every 1,000 batches.
- **Two side critics replace head and shared.** head is settled (−0.000), and shared was close to its start. Each side
  critic takes P160k's frozen features plus a fresh small network on the public and hidden input (two conv layers of
  128 channels, then 256 features), fused into the head. The fusion's output layer starts at zero.
  - They cannot damage the public features, which is how the first run failed.
  - A small fresh network may underfit, so they give a lower bound.

| Critic | Input | What trains |
| --- | --- | --- |
| head0 | public | nothing (the reference) |
| sep | public | trunk copy + head |
| hands | public + opponents' hands | oracle trunk (wall rows zeroed) + head |
| oracle | public + all hidden | oracle trunk + head |
| hands_side | public + opponents' hands | head + side network, on P160k's frozen features |
| oracle_side | public + all hidden | head + side network, on P160k's frozen features |

The smoke test confirmed that all five trained critics start exactly at head0 (max |ΔV| = 0).

**Training:** 15,000 batches of 512; warm-up 200, then cosine to 10%; AdamW without weight decay, clip 1.0, BN frozen,
bf16. Side networks at LR 1e-4. Speed test: 2.86 batches/s with 4 loader workers, data wait 2%, about 6.4 GB of WSL memory free; so about 1.5 hours. A memory guard stops the run if free memory falls under 2 GB.

**Verdicts (thresholds unchanged):**
- In each family (hands: hands, hands_side; oracle: oracle, oracle_side), the variant with the higher best validation
  EV is the family's critic. Its test EV is the reading.
- EV(oracle) ≥ 0.40: a PG v2 probe is worth designing. Its run needs the user's go.
- Every critic < 0.20: the PG direction closes.
- Otherwise: reported as it is, PG v2 not recommended by default.
- **Lower-bound flag.** If a critic's best look is the last one and its last three looks rise, its EV is a lower bound.
  If such a critic is the one under 0.20, the reading is "not closed, lower bound", and the run is reported as it is.

**Departures added to the first run's table (R8):**

| | Conventional, or the online probe | Here | Why, and the cost |
| --- | --- | --- | --- |
| 9 | The probe: LR 1e-5 with an optimizer step every 8 batches | 1e-5 with a step every batch | At one step per 8 batches, a pass would move the trunk only ~1,900 steps. Each step moves as far as online, with 8 times the steps per sample |
| 10 | A privileged critic is a full network | Also two side critics: frozen public features + a small fresh network | Robust to the first run's failure; a small network may underfit, so it gives a lower bound |
| 11 | Inputs in float32 | Stored as float16 in the shuffle buffer | Error at most 2.4e-4, 16 times below the bfloat16 rounding the model applies to its input anyway; it keeps 6,000 decisions per worker in memory |
| 12 | The first run: head LR 1e-3 | 1e-4 | At 1e-3 the first run's head wandered between 0.048 and 0.087 on validation and never beat head0 |

**Priors, revised after the first run:** EV(oracle) ≥ 0.40 about 10%; every critic under 0.20 about 60%.

**Cost:** about 1.5–2 hours on the 5060 Ti. It runs as a test on an idle card; nothing else was queued.

## Result of the second run (2026-11-09 21:55): no critic reaches 0.20, the PG direction closes

15,000 batches in 1.6 hours at 2.67 batches/s; the lowest free WSL memory was 3.7 GB, and the guard never fired.

| Critic | EV on test | 95% CI | − head0 | Best validation look |
| --- | ---: | --- | --- | ---: |
| head0 | 0.098 | [0.090, 0.106] | | |
| sep | 0.098 | = head0 | 0 | step 0: training never beat the start |
| hands | 0.098 | = head0 | 0 | step 0 |
| oracle | 0.098 | = head0 | 0 | step 0 |
| hands_side | 0.113 | [0.105, 0.122] | +0.015 [+0.012, +0.018] | 14,000 |
| oracle_side | 0.114 | [0.105, 0.123] | +0.016 [+0.012, +0.019] | 14,000 |

- **Verdict, as set in advance:** every critic is under 0.20, so the PG direction closes. The family critics are sep
  (0.098), hands_side (0.113) and oracle_side (0.114). Neither side critic's curve was still rising: both peaked at
  14,000 and fell at 15,000, so the lower-bound flag does not apply.
- **The best critic hardly cuts the noise.** Its residual-variance factor is 1.02: the advantage would carry 98% of the
  gradient noise it carries now. About 89% of the return's variance is out of reach of any critic we built, hidden
  state included.
- **The wall adds nothing beyond the hands:** oracle_side − hands_side = +0.0002 [−0.001, +0.002].
- **The gain is small everywhere.** By decision, the side critics add 0.015 on discards, 0.04 on riichi and 0.06 on
  agari. By turn, they add 0.012 at junme 0–5 and 0.022 at 15+.
- **The full-trunk critics overfit even at the probe's LR.** On the training stream, measured before each step, they
  read 0.13–0.16; on validation they stayed under the start at every look, and recovered toward it only as the LR
  annealed (hands 0.063 at the last look). 55,000 training games give about 550,000 independent kyoku outcomes. A
  40-block trunk learns to recognize the kyokus and memorizes their outcomes.

**Limits:**
- There is no public-only side critic. So part of the side critics' +0.015 may come from added capacity rather than
  from the hidden input. It does not matter for the verdict: even the whole gain leaves the best critic at 0.11.
- A privileged trunk trained on far more games might do better. The hands trunk was still recovering at the end, but
  it never got back to its start. An online stage plays well over 100,000 games, a few times this training set. To
  reach 0.20, though, a trunk would have to gain about 0.10 over the start, where the side critics, which do not
  overfit, gained 0.015.
- All of this is for the kyoku return the probe used (γ 1). A critic for a discounted or a shorter-horizon return would
  explain more, but it would change the PG estimator too, so it is a different design.
