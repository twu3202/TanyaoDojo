# Hangzhou wildcard rooms

The seven rooms play eight hands at base stake one with the 136-tile set and no flowers.
The human hosts and deals the first hand. The wall is live to the last tile; a kong needs a replacement left.
Kongs draw from the other end of the wall and have no immediate payment. The winner deals next and the dealer
repeats on a draw. On a draw all deltas are zero. Chun’an also allows discard wins and robbing an added kong.
Songyang keeps 20 tiles in reserve and allows robbing an added kong from any seat.

| Room | Calls | Dealer share | Feed liability |
| --- | --- | --- | --- |
| Three-player, fixed dealer | Pon and kong | Always ×8 | None |
| Four-player, three feeds | Chow from the preceding seat; pon and kong from any seat | ×2, ×4 on first repeat, ×8 on further repeats | Dealer and non-dealer only |
| Tonglu | Chow, pon and kong subject to two separate call limits | ×2, ×4, ×8 | None |
| Chun’an | Chow, pon and kong, with the guard seat’s pon restriction | Always ×8 | None |
| Two-player, flip wildcard | Chow, pon and kong against a single opponent | One share | None |
| Xiaoshan | Two calls per supplier, passed-pon lock and a restricted second chow | ×4, then ×8 | None |
| Songyang | Chow, pon and kong, with a 20-tile reserve | Always ×2 | Three feeds between any pair of seats |

## Wildcard and win multipliers

White Dragon is the default wildcard for the concealed part of a hand, including honors. No open meld
or kong uses a wildcard. The last draw is counted separately when checking whether the hand was waiting for
any tile to pair with a wildcard. That wait is a **Burst** (爆头); six pairs with the same wait make a
**Seven-pairs Burst** (七客). A wildcard can fill a set for either route. An ordinary completed wildcard hand
can also win in these rooms, except in Tonglu when the hand contains a wildcard.

| Pattern | Multiplier |
| --- | --- |
| Regular hand / Burst | ×1 / ×2 |
| One, two, three or four consecutive kong replacement draws ending in a win | ×2, ×4, ×8, ×16 |
| One, two or three consecutive wildcard floats ending in a Burst | ×4, ×8, ×16 including the Burst |
| Seven pairs / natural seven pairs | ×2 / ×4 |
| Each natural four-of-a-kind used as two pairs | An additional ×2 |

Here natural seven pairs (清七对) means seven pairs without a wildcard. Honors and mixed suits count.
A seven-pairs Burst doubles the seven-pairs base. Kong and Burst multipliers stack, including a kong followed
by a float (杠飘), or a float followed by a kong (飘杠), each ×8 with one kong and one float. When a hand
qualifies in two ways, use the higher score. This room counts only actual natural four-of-a-kinds for luxury
pairs and treats the drawn tile in a seven-pairs Burst as the partner of the reserved wildcard.

## Wildcard float and calls

After a draw, use **Float wildcard** to discard an extra White Dragon while keeping a Burst or seven-pairs Burst
wait. An ordinary White Dragon discard when the hand cannot float creates no bonus. Until the floater draws
again, other seats can win or make concealed kongs on their turns and otherwise have to discard their exact draw.
An ordinary discard by the floater breaks the consecutive float bonus. The floater can still make concealed kongs.

The three-player room blocks the floater's pon and open kongs for the rest of the hand. The three-feed room
allows the floater to chow or pon during the forced circuit, ending the circuit early, and still allows chow or
pon afterwards. Open kongs stay blocked for a seat after its first float. Calls from forced discards do not
count towards three-feed liability. These choices make the published float and feed exceptions explicit; the
three-player room also uses the documented Hangzhou forced-draw convention for the other seats.

## Payers

For a self-draw, compute the share for each loser. A dealer-involved payment uses the room's dealer multiplier;
a payment between non-dealers uses ×1. For instance a non-dealer's Burst in a three-player room costs the dealer
16 and the other seat 2. In the four-player room each chow, pon or direct kong from the same supplier is one
feed; upgrading a pon preserves its count. Non-dealers can take up to two calls from each other. In a
dealer/non-dealer pair, after the same player receives three eligible feeds from the same opponent, that opponent
owes all three normal shares if the recipient wins. If the feeder wins, the recipient pays twice the sum of
the feeder's normal shares. Other seats pay zero. For multiple applicable links, use the first formed link.

Beginner AI weighs wildcard-aware tile efficiency, visible copies and exact self-draw waits. It sometimes floats
early with a guaranteed Burst wait, and prioritizes winning when the wall is shorter. Concealed opposing kongs
are shown face-down and their identities are excluded from the AI's visible counts.

## Tonglu

This room combines the two-chow and two-pon limits, mandatory Burst with a wildcard, the ordinary dealer start,
and a fully live wall. A natural seven-pairs win is legal. A seven-pairs hand holding a wildcard must use it
as a seven-pairs Burst wait. A direct kong counts against the pon limit and upgrading a pon keeps its existing
slot. Both caps are per supplier. The floater may take an open kong during the circuit, or upgrade a pon or
make a concealed kong after the next draw. This room has no feed liability. It counts only natural quads
for luxury seven-pairs scoring.

## Chun’an

The room takes the fixed ×8, two-call cap without liability, dealer-linked discard wins, and guard-seat options.
Dealer and non-dealer may win on one another’s discard or added kong. Only the first seat to accept in turn
order wins. The tile supplier pays the sole ×8 share. A discard win can use a regular completed wildcard hand.
Burst timing and kong replacement bonuses apply to self-draw. The float circuit suspends discard and robbing
wins. The guard seat, just before the dealer, can pon only its own preceding seat while holding no wildcard.
The restriction applies to pon. The separate call cap counts chows, pons and direct kongs together.
The scope of the guard restriction and the nearest-winner choice are table conventions where the source is brief.

## Two-player flip wildcard

One tile from the undrawn wall is revealed for the hand. Its other three copies are wild. If a tile other
than White Dragon was revealed, all four White Dragons act as the revealed tile’s regular face. That face
is used for sequences and calls, and the board still shows the physical White Dragon. If White Dragon is
revealed, its other three copies are wild with no substitute. The indicator stays out of the draw pile.
Both seats are blocked from exposed calls during a float circuit. Natural seven pairs double the seven-pairs
base, with another doubling for each actual natural quad. This practice room fixes a single payer at one
share and uses the full 136-tile set with all undealt tiles live apart from the indicator.

## Xiaoshan Ten Winds

The room starts at second dealer, rises from ×4 to ×8 on the first repeat, and caps at third dealer.
The winner deals next and an exhausted wall repeats the deal. The selected profile uses the published
128 limit and no score buy-in or running loss cap. The detailed payout formula applies the hand multiplier
to each seat’s dealer factor, adds one for each of the winner’s kongs, then caps that payer at 128.
Kongs have no immediate payment in this profile.

The first chow allows a swap discard. A chow reaching the second permitted call from that supplier locks
its tile and the opposite end of the same sequence until the discard. Declining a pon locks that tile until
one’s next draw. Chows, pons and direct kongs share the two-call limit. Upgrading a pon keeps its slot.

With a wildcard, an ordinary win must score at least ×2. Hand multipliers can combine when the selected
shape supports them. A replacement win uses ×2 for the last open kong or ×4 for the last concealed kong.
Seven-pairs multipliers start at ×4 and are ×8, ×16 and ×64 with one, two or three natural quads.
Natural seven pairs double these values. All-honor pungs (正风), other all-honor hands (乱风), opening wins
and Ten Winds use their own values independently of other hand patterns.

From the first discard, keep discarding honors, the wildcard or a physical White Dragon. Chows, pons and
open kongs end the run. Concealed kongs can keep it. Choose **Discard and win · Ten Winds** on the tenth
eligible discard or a later eligible discard in the same run. The source’s formula notation is ambiguous.
This table starts at ×4 on discard ten and doubles on each additional eligible discard, before the same
per-payer cap. Ten Winds can settle without a standard winning shape.

## Songyang Kong Liability

The base room uses fixed dealer ×2, the flipped wildcard, and a 20-tile reserve aside from the displayed
indicator. Replacement tiles leave the same 20 tiles untouched. Only self-draw and robbing an added kong
can win. For a robbery, the first accepting seat wins and the kong player pays all three normal shares of
the ×4 robbery-adjusted win. The original pon stands. For a self-draw, the first applicable three-feed link
sets the liable payer’s bill to all three shares, or double for reverse liability. Other losers pay normally.

Hand and event bundles multiply where compatible. Ordinary forms include pungs, pairs, suit and honor
patterns, wildcard-head waits, floats, single-tile hands after four calls, and replacement wins. Three
opening wildcards can declare a win even without a complete hand. A wildcard used in thirteen orphans
or Seven Winds lowers it to the operator’s irregular tier. The published score table does not include
complete tile examples for those names. This room counts them by physical face and permits the wildcard
to fill missing members. Seven Winds means one of each honor plus seven distinct suited tiles, with at
least three ranks between tiles of a suit. The public display shows the wildcard and the number of live tiles.

## Sources

- [Zhejiang Game Hall three-player rules and multiplier table](https://www.hzbianqu.com/game/581.html)
- [Zhejiang Game Hall three-player dealer and call rules](https://www.hzbianqu.com/article/1476.html)
- [Zhejiang Game Hall three-player pattern examples](https://www.hzbianqu.com/article/1548.html)
- [Zhejiang Game Hall four-player room options](https://www.hzbianqu.com/game/37.html)
- [Bianfeng GameABC forced discards, feed counting and liability](https://www.gameabc.com/news/201911/6191.html)
- [Zhejiang Game Hall example for reserving a wildcard and floating the second](https://www.hzbianqu.com/game/1611.html)
- [Zhejiang Game Hall Tonglu options and scoring](https://www.hzbianqu.com/game/62.html)
- [Zhejiang Game Hall Chun’an dealer link and guard seat](https://www.hzbianqu.com/game/1055.html)
- [Zhejiang Game Hall Xiaoshan hand and payout table](https://www.hzbianqu.com/game/55.html)
- [Zhejiang Game Hall Songyang scoring and reserve](https://www.hzbianqu.com/game/218.html)
- [Zhejiang Game Hall Seven Winds definition for the related Jinhua room](https://www.hzbianqu.com/game/74.html)
