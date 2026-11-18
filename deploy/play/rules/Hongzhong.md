# Hongzhong 112 and Fangchenggang rooms

Rule ID: `hongzhong-dojo-2ma-v1-20261116`. This is one explicit four-player room profile, not a claim that every
local game called 红中麻将 has these rules. Policy: public-information wildcard-aware tile-efficiency heuristic.
No Sichuan model is silently reused for this different game.

## Source and room choices

Primary source checked 2026-11-16: [GameTea, 红中麻将怎么玩？ (2018-09-18)](https://gametea.net/news/201809/7405.html).
The publisher specifies the 112-tile set, wildcard restrictions, no chi, no ordinary discard win, self-draw and
multiple robbed-added-kong wins, four-red win, dealer succession, bonus hits and the four-player scoring table.
The ordinary N-tile bonus mode and its separate one-tile face-value mode are separate room choices.

- Three suits plus four red dragons; dealer 14 tiles and others 13.
- Reds substitute in a winning hand; they do not substitute other tiles in open sets. Calling reds themselves is legal.
- Standard wins: each loser pays 2 + 2 per hit. Bonus hits are suited 1/5/9 and red dragons.
- Draw two bonus tiles, plus one for a no-red winner, limited to the remaining wall.
- Exposed kong: discarder pays 3; added kong: each opponent pays 1; concealed kong: each opponent pays 2.
- Winner deals next; after a draw, the last drawer deals next.

The following **Dojo choices fill unspecified boundaries** and are displayed in the game:

1. Standard four-set-and-pair wins only; seven pairs is off. Four concealed reds may win on the player's own
   draw turn (including the dealer's initial 14), using ordinary win/bonus scoring. Open red melds do not trigger it.
2. No-red bonus eligibility includes the full hand and open melds. Wins may be declined; there is no pass-win lock.
3. Kongs require a remaining replacement tile. Pon must be followed by a discard, not a self-draw claim or another kong.
4. An unwon last draw is discarded, then the hand ends. Prior kong payments stand; no ready-hand penalties/refunds.
5. Multiple robbed-kong winners share one bonus sequence; each uses the first 2 or 3 tiles according to eligibility.
   The kong maker pays three shares to each winner; the nearest winner in response order deals next.
6. Only the offered fourth tile can be robbed, no kong payment occurs on a robbed kong, and that physical tile is
   counted once even when multiple winners show it. Concealed kongs and ordinary discards cannot be robbed/won on.

## GameTea one-ma room

Rule ID `hongzhong-dojo-one-ma-v1-20261118`. This uses the same GameTea source and hand rules. After a win, reveal
at most one tile from the available wall. In addition to 2 base points per share, a suited tile adds its rank plus 1
and a red dragon adds 11. Empty walls contribute zero bonus. A no-red hand also draws at most one tile in this
Dojo profile. The kong maker pays three shares per winner for a robbed added kong, using the same revealed tile.

## Fangchenggang four-ma room

Rule ID `fangchenggang-hongzhong-four-ma-v1-20261118`. The source is the operator's [Fangchenggang product rules](https://www.hzbianqu.com/game/1123.html).
This preset selects ordinary four-tile bonus scoring, no no-red extra bonus, and seven pairs. Seven pairs is closed,
can use wildcards, and receives regular scoring. A red upgrade cannot be robbed. A natural added kong is available
only while its matching tile is the latest draw. Kong replacements and bonus flips come from the opposite end of the
wall to normal draws. Four remaining tiles are reserved for bonuses. If a kong is robbed, its maker deals next.

Dojo conventions here are to count a quad as two pairs, allow calling reds themselves and passing on a win, and
settle the last playable discard as a draw with the reserve untouched. A kong needs a replacement while preserving
the reserve. Kong payments stand on drawn hands. Multiple robbers share the single bonus sequence. Four-red wins
get normal bonus scoring in this preset, with the optional all-wall bonus disabled.

## State and invariants

`hongzhong.py` owns the complete hand. `hongzhong_game.py` adapts it to the same web-session interface as riichi
and Sichuan. This core may support future self-draw/bonus families, but each new variant still needs a frozen
specification and independent tests; removing reds alone does not implement Zhuanzhuan.

Every transition validates its action, keeps exactly four physical copies of each of the 28 tile types, preserves
zero-sum scoring, and is bounded by a turn guard. Bonus and robbed tiles are separate physical zones. The heuristic
receives only its own tiles, exposed tiles, and legal actions. Its distance is a progress estimate; winning legality
uses a separate exhaustive wildcard decomposition, not the heuristic.
