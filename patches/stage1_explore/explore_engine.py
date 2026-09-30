"""Branch B, Stage 1: targeted exploration at the push/fold and riichi decision boundary (worker side).

Stage 0 (TanyaoDojo jax_rl/mjai_bot/stage0_stale.py) found that at push/fold and riichi decisions the taken action's
Q is calibrated but the untaken option's Q sits ~49 pt too low (offline CQL legacy, frozen since rl1): with a
99.5%-greedy worker the untaken option never gets a target. Near the boundary (our own margin < ~5 pt) pushes
are wrong on average (flip to fold: first order +0.15 pt/game on held-out games). This engine gives the untaken
option targets there:

  at a decision where the greedy choice and its push/fold or riichi/dama counterpart are within `margin` Q units,
  play the counterpart with probability `prob`.

Pairs (same definitions as riichi-eval's policy probe, except that genbutsu here is the riichi players' own
discards only — tiles passed after a riichi are not counted; the v5 defence rows carry exactly that set):
  push -> fold   exposed (an opponent's riichi accepted, we are not in riichi), greedy = non-genbutsu discard or
                 riichi, a genbutsu discard legal; counterpart = best genbutsu discard
  fold -> push   exposed, greedy = genbutsu discard, a non-genbutsu discard legal; counterpart = best non-genbutsu
  riichi -> dama riichi legal, greedy = riichi; counterpart = best discard
  dama -> riichi riichi legal, greedy = a discard; counterpart = riichi
If two pairs apply, the one with the smaller margin is used.

The engine asks libriichi for v5 observations (v4 rows + 10 defence rows; the first 1012 rows are bit-identical
to v4, checked 26,850/26,850) and feeds only the v4 rows to the v4 network. `is_greedy` is reported as
"action == argmax Q", so the trainer's loader can find every real deviation (this also covers Boltzmann samples
that happened to hit the argmax, which upstream marks non-greedy).
"""
import numpy as np
import torch

from engine import MortalEngine, sample_top_p

V4_ROWS = 1012
ROW_OPP_RIICHI = [857, 858, 859]     # relative opponents 1..3, riichi accepted (whole row = 1)
ROW_SELF_RIICHI_ACCEPTED = 869
ROW_SELF_RIICHI_DECLARED = 878       # set at discard decisions once we declared riichi
ROW_AT_KAN_SELECT = 870
ROW_GENBUTSU = V4_ROWS + 4           # 3 rows, per relative opponent, only for riichi-accepted opponents

# discard action -> tile kind (aka fives 34/35/36 -> 5m/5p/5s)
_KIND = torch.tensor(list(range(34)) + [4, 13, 22])
KINDS = ('push2fold', 'fold2push', 'riichi2dama', 'dama2riichi')


def frontier(q, obs, masks):
    """Vectorised boundary pairs. q (B,46) raw Q, obs (B,1022,34) v5, masks (B,46) bool.
    Returns greedy (B,), alt (B,), margin (B,) [inf where no pair], kind (B,) index into KINDS or -1."""
    B = q.shape[0]
    dev = q.device
    neg = torch.finfo(q.dtype).min
    qm = q.masked_fill(~masks, neg)
    greedy = qm.argmax(-1)
    ar = torch.arange(B, device=dev)
    q_g = qm[ar, greedy]

    disc = masks[:, :37]
    opp_riichi = obs[:, ROW_OPP_RIICHI, 0] > 0.5                                   # (B,3)
    self_riichi = (obs[:, ROW_SELF_RIICHI_ACCEPTED, 0] > 0.5) | (obs[:, ROW_SELF_RIICHI_DECLARED, 0] > 0.5)
    kan_select = obs[:, ROW_AT_KAN_SELECT, 0] > 0.5
    exposed = opp_riichi.any(1) & ~self_riichi & ~kan_select
    genb = obs[:, ROW_GENBUTSU:ROW_GENBUTSU + 3, :] > 0.5                          # (B,3,34)
    safe_kind = (genb | ~opp_riichi[:, :, None]).all(1)                              # (B,34)
    kind_idx = _KIND.to(dev)
    safe_act = safe_kind[:, kind_idx] & disc                                         # (B,37)
    unsafe_act = ~safe_kind[:, kind_idx] & disc

    g_disc = greedy < 37
    g37 = greedy.clamp(max=36)
    g_safe = g_disc & safe_act[ar, g37]
    g_unsafe = g_disc & unsafe_act[ar, g37]
    g_riichi = greedy == 37

    def best_of(sel):
        v = q[:, :37].masked_fill(~sel, neg)
        return v.argmax(-1), v.max(-1).values, sel.any(1)

    a_safe, q_safe, has_safe = best_of(safe_act)
    a_unsafe, q_unsafe, has_unsafe = best_of(unsafe_act)
    a_disc, q_disc, has_disc = best_of(disc)

    inf = torch.full((B,), float('inf'), device=dev, dtype=q.dtype)
    # push -> fold
    p2f = exposed & (g_unsafe | g_riichi) & has_safe
    m_p2f = torch.where(p2f, q_g - q_safe, inf)
    # fold -> push
    f2p = exposed & g_safe & has_unsafe
    m_f2p = torch.where(f2p, q_g - q_unsafe, inf)
    # riichi -> dama / dama -> riichi (riichi decisions are not limited to exposed states)
    can_r = masks[:, 37] & ~kan_select
    r2d = can_r & g_riichi & has_disc
    m_r2d = torch.where(r2d, q_g - q_disc, inf)
    d2r = can_r & g_disc
    m_d2r = torch.where(d2r, q_g - q[:, 37], inf)

    margins = torch.stack([m_p2f, m_f2p, m_r2d, m_d2r], 1)                         # (B,4)
    alts = torch.stack([a_safe, a_unsafe, a_disc, torch.full_like(greedy, 37)], 1)
    margin, kind = margins.min(1)
    alt = alts[ar, kind]
    kind = torch.where(torch.isfinite(margin), kind, torch.full_like(kind, -1))
    return greedy, alt, margin, kind


class FrontierExploreEngine(MortalEngine):
    def __init__(self, *args, explore_margin, explore_prob, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.version == 4, 'frontier exploration reads v5 defence rows appended to v4 observations'
        self.version = 5                       # what libriichi encodes for this engine; the network sees v4 rows
        self.explore_margin = explore_margin   # Q units
        self.explore_prob = explore_prob
        self.stats = np.zeros((2, len(KINDS)), dtype=np.int64)   # [eligible, explored] per kind

    def _react_batch(self, obs, masks, invisible_obs):
        obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        batch_size = obs.shape[0]

        phi = self.brain(obs[:, :V4_ROWS])
        q_out = self.dqn(phi, masks)

        if self.boltzmann_epsilon > 0:
            use_greedy = torch.full((batch_size,), 1 - self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(use_greedy, q_out.argmax(-1), sampled)
        else:
            actions = q_out.argmax(-1)

        greedy, alt, margin, kind = frontier(q_out.float(), obs, masks)
        eligible = (kind >= 0) & (margin < self.explore_margin) & (actions == greedy)
        explore = eligible & (torch.rand(batch_size, device=self.device) < self.explore_prob)
        actions = torch.where(explore, alt, actions)
        is_greedy = actions == greedy

        k = kind.clamp(min=0)
        self.stats[0] += np.bincount(k[eligible].cpu().numpy(), minlength=len(KINDS))
        self.stats[1] += np.bincount(k[explore].cpu().numpy(), minlength=len(KINDS))
        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()
