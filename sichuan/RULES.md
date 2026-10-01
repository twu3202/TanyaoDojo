# Sichuan mahjong (Xuezhan Daodi, "bloody to the end"): rules frozen at v0 (2026-07-24)

**Purpose**: the single source of truth for the pure-RL environment. Both the reference implementation (reference_impl.py)
and the future JAX environment follow this document; rule disputes are settled by it, and any change requires a version
bump and a rerun of the differential tests. Baseline: the mainstream Chengdu Xuezhan Daodi rules, including "wind and
rain" (immediate kong payments) and the ready-hand check with kong refunds at an exhaustive draw; variants deliberately
excluded from v0 are listed at the end.

## 1. Tiles and players

- 108 tiles: characters / dots / bamboos (wan / tong / tiao), 1-9 × 4 each. **No honor tiles, no flowers.**
- 4 players. The dealer sits in seat 0 for the first hand (dealer rotation in the RL environment is controlled by the outer layer).

## 2. Opening

- The dealer holds 14 tiles, the others 13 (implementation: everyone is dealt 13, then the dealer draws the first tile, i.e. the dealer moves first).
- **Exchanging three tiles (huan san zhang): not in v0** (see §9).
- **Declaring a void suit (ding que, mandatory)**: after the deal and before play, every player simultaneously declares one suit as their "void suit".
  - RL action: choose 1 of 3 (void characters / dots / bamboos). The reference implementation provides a heuristic default (the suit with the fewest tiles in hand; ties go to the lower index).
  - Constraint A (winning): at the time of winning, hand + melds **must contain no tiles of the void suit**.
  - Constraint B (play): while the hand still holds void-suit tiles, **only void-suit tiles may be discarded** (the discard mask opens only the void-suit tiles).
  - Void-suit tiles cannot be used for pungs or kongs.

## 3. Play and responses

- Order: draw → (optional: concealed kong / added kong / self-drawn win) → discard → responses from the other players.
- **No chow.** The only responses are: pung, exposed kong (on a discard), win on the discard, pass.
- Response priority: **win > pung/kong**; several players able to win at once = **multiple winners on one discard** (all of them stand; see §6).
- A pung or kong must be followed by a discard (after a kong, first draw a replacement "on the kong" tile).
- Three kinds of kong: exposed kong (another player discards the 4th tile), added kong (self-drawing the 4th tile after a pung), concealed kong (4 tiles in hand).
  - An added kong can be robbed: a player waiting on that tile can win on it (**robbing the kong**, counted as a fan type in §5). When the robbing win stands, the added kong does not (the tile goes to the winner,
    and the player who declared the kong gets no wind-and-rain payment).

## 4. Winning hands

- Standard: 4 sets (pungs / chows; a chow is only 3 consecutive tiles of the same suit) + 1 pair. Melded pungs and kongs occupy set slots.
- **Seven pairs**: 7 pairs (concealed hands only; 4 identical tiles count as 2 pairs, and any "gen" among them is counted separately, see §5).
- The void-suit constraint (§2A) must hold. **There is no minimum fan to win** (a plain hand can win).

## 5. Scoring: fan and multipliers

Base = 1. Score = base × 2^fan, **capped at 4 fan (2^4 = 16 × base)** (call transfer, flower-pig penalties and the like are not in v0).

| Fan type | Fan | Notes |
|---|---|---|
| Plain win (ping hu) | 0 | |
| All pungs (dui dui hu) | 1 | 4 pungs (kongs included) + a pair |
| Full flush (qing yi se) | 2 | All tiles of one suit |
| Seven pairs (qi dui) | 2 | |
| Dragon seven pairs (long qi dui) | 3 | Seven pairs containing ≥1 set of 4 identical tiles (those 4 are not counted again as a gen) |
| Golden hook (jin gou diao) | 1 (additive) | Only 1 tile left in hand as a single wait (everything else melded) |
| Bloom on the kong (gang shang hua) | 1 (additive) | Self-drawn win on the replacement tile after a kong |
| Cannon on the kong (gang shang pao) | 1 (additive, counted for the winner) | Winning on a discard made right after a kong |
| Robbing the kong (qiang gang hu) | 1 (additive, counted for the winner) | |
| Moon from the sea floor (hai di lao yue) | 1 (additive) | Self-drawn win on the last tile |
| Self-draw (zi mo) | 1 (additive) | |
| Gen | +1 per gen | Every set of 4 identical tiles (kongs included) in the winning hand is one gen |

Fan types stack (e.g. full flush + all pungs = 3 fan), and the cap still applies after stacking.

Settlement: self-draw = each of the other three players (those who have not won and are still in play, see §6) pays 2^fan;
win on a discard = the discarder alone pays 2^fan.

## 6. Bloody to the end (play continues after wins)

- A winner reveals the hand and leaves play: no more drawing or discarding, **hand and melds are frozen**, and the winner takes no further part in responses or settlements (except kong payments; see the refund in §7).
- Multiple winners on one discard: if several players can win on the same discard → **all of them stand**, and the discarder settles with each winner separately.
- End condition: **3 players have won**, or **the wall is exhausted**.
- Could the discarder be a player who has already won? No — players who have won no longer discard. Players who have won do not pay for later self-draws or discards.

## 7. Wind and rain (immediate kong payments) and refunds

When a kong stands, it is paid **immediately** (only by players who have not won and are still in play):
- Exposed kong (on a discard): the player who discarded the tile pays 2 × base.
- Added kong (ba gang): every player pays 1 × base.
- Concealed kong: every player pays 2 × base.

**Refund (tui shui)**: at an exhaustive draw (the wall is exhausted), any player who is **not ready** must refund all kong
payments received during this hand (each back to whoever paid it, even if the payer has since won). Ready players keep
their kong payments. Players who have won count as ready and refund nothing.

## 8. Ready-hand check at an exhaustive draw (cha da jiao)

At an exhaustive draw: **each player who is not ready pays each ready player the value of "the maximum possible fan of
the hand that ready player is waiting on"** (at 2^fan, subject to the cap). Ready players do not pay one another, and
neither do players who are not ready. Players who have won take no part in the check.

## 9. Explicitly excluded from v0 (to be revisited in a later version)

- Exchanging three tiles (swapping tiles at the start) — planned for v1; it affects strategic depth but not the skeleton of the state machine.
- Call transfer, flower-pig (cha hua zhu) penalties, regional multiplier variants for rain/wind.
- Base-score tiers / table fees, dealer-repeat rules (the RL setup plays single hands; the outer layer controls dealer rotation).

## 10. Invariants (test anchors)

1. Zero-sum scoring: at any moment, the four players' score changes sum to 0.
2. Tile conservation: wall + four hands + melds + discard rivers = 108.
3. A winner's state is frozen: hand and melds no longer change.
4. Void-suit constraint: no winning set of 14 (or 14+3k) tiles contains the void suit; while a hand holds void-suit tiles, the discard mask contains only void-suit tiles.
5. Termination is guaranteed: any legal action sequence reaches the end of the game (empty wall or 3 wins) in a finite number of steps.
