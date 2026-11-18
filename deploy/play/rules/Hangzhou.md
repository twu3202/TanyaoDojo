# Hangzhou wildcard rooms

The twelve rooms play eight hands at base stake one. The Anji rooms use 144 tiles with flowers. The other rooms use 136 tiles.
The human hosts and deals the first hand. A kong needs a live replacement tile.
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
| Suichang | Chow, pon and kong, with a shrinking reserve | Same as non-dealers | An added-kong robbery charges its supplier the entire bill |
| Deqing Wukang | Chow, pon and kong with a successor wildcard | Dealer pays 10 for Earth | An added-kong robbery charges its supplier three shares per winner |
| Deqing Xinshi | Chow, pon and kong, including an immediate pon upgrade | Dealer pays 10 for Earth or a non-dealer opening special | A later robbery charges its supplier six shares at each winner’s rate |
| Anji Fifteen Flowers | Chow, pon and kong, including after a call | Same as non-dealers | None |
| Anji Tea House Self-Draw | Chow, pon and kong, with pon upgrades from the next draw | Same as non-dealers | None |

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

## Suichang Shared Discard Bill

This room uses the operator’s ordinary profile at base stake one. It uses the flipped wildcard and White
Dragon substitution. Take the first accepting seat in order for a discard or an added-kong robbery. Use the
winner as the next dealer, with no dealer multiplier. Ordinary self-draws charge each loser two shares.
A discard win charges its supplier two shares and the other losers one each. For the operator’s ×2 robbery
and sole-payer rule, this room combines all four discard shares into a single bill for the kong supplier.

Start with 34 tiles reserved separately from the indicator. This room interprets the operator’s shorter-tail
rule as releasing two reserve tiles per completed kong. The replacement comes from the edge of the adjusted
reserve. A robbed kong releases no tiles. A draw settles at zero.

Compatible hand and event multipliers combine. Each natural quad doubles the value of seven pairs. An
unstructured all-honor hand pays four shares and a structured one starts at eight. Thirteen orphans and
Seven Winds use the same physical-face interpretation as Songyang with Suichang’s own score table. This room
uses the first dealer discard for Earth and allows three wildcards on the dealer’s opening hand or on an
uninterrupted first draw. The operator documents the chip-based match as a separate profile.

## Deqing Wukang

This four-seat room selects Wukang plain scoring with chow, base stake one and no cap. Three-feed liability
and forced Burst are room options. The indicator is removed from play. All four copies of its successor can
be drawn and used as wildcards. Suits wrap after rank nine, winds cycle East–South–West–North, and dragons
cycle Red–Green–White. White Dragon retains its face when it is a natural tile. Wildcards do not enter melds.

Ordinary self-draw pays one share per loser. Seven pairs and Burst pay two, as does a single kong draw.
Successive kongs and floats double the relevant payout. A single float pays four, a single kong Burst pays
four and a single kong followed by a float pays eight. This room pays the higher applicable hand or event
value where the table lists separate values. Floating forces the other seats to discard their draws until
the floater draws again. Four wildcards held at the deal or collected on an uninterrupted first draw pay
ten each, even without a complete hand. A dealer’s opening hand pays twenty each.

Earth applies to a non-dealer’s first draw or a winning discard before the dealer’s second draw, while
the first circuit is uninterrupted by calls or kongs. For each Earth winner, the dealer pays ten and
other losing non-dealers pay five. Several players may win on the same eligible discard or added kong.
For an added-kong robbery, use at least the replacement-win value, doubling for prior kongs in that chain.
The supplier pays three times the highest robbery share to each winner and the original pon stands. The
first accepting winner in turn order deals next.

The Ten Winds declaration opens from the tenth consecutive discard of an East, South, West, North or
wildcard tile. An unrelated discard or open call restarts that seat’s count. This room pays four per loser
on the tenth and doubles for each later qualifying discard. A draw has zero payments.

## Deqing Xinshi

The four-seat plain room shares Wukang’s successor wildcard, Earth window and discard-based Ten Winds.
It uses base stake one, chow and no cap. When a player has never held a wildcard in the deal or on a draw,
ordinary self-draw, seven pairs and replacement-win shares double. The lower rate still applies after a
previously held wildcard has been discarded. Heavenly Hand pays ten per loser.

Opening Ten Winds wins immediately with at least ten East, South, West, North or wildcard tiles at the deal.
A non-dealer with nine may also claim the dealer’s first natural wind discard as the tenth before drawing.
For Opening Ten Winds and Four Opening Wildcards, a dealer winner collects ten from each loser. A non-dealer
collects ten from the dealer and five from the other losers. This room checks dealt hands in dealer order,
gives Four Opening Wildcards priority within the same hand, and permits more than one winner on the discard.

After pon, a player who already holds the fourth copy can add it before making their discard. Draw a kong
replacement immediately, with no robbery window. An added kong in a later turn can be robbed by several
players. For each winner the supplier pays six times that winner’s own share. Use at least their replacement
win share, including the no-wildcard rate if it qualifies. Double for earlier kongs in the supplier’s chain.

## Anji Fifteen Flowers

Start with four copies of the 34 ordinary faces and one each of the four seasons and four plants. Set aside
the indicator before dealing. White Dragons, the eight special tiles and the remaining copies of an ordinary
indicator become flowers. For a White Dragon or special-tile indicator, substitute North as the extra flower
face and use East as the wildcard. In every case exactly 15 flowers remain available to draw. For other
indicators, the wildcard is the next suited rank or wind, or the other member of the Red–Green pair.

Reveal every flower during the deal or a draw, then take a replacement from the opposite end. Repeat until
an ordinary tile is drawn or the wall runs out. Wildcards cannot replace flower faces in a winning shape or
join a meld. Every seat can see the indicator and all exposed flowers. The operator's shared list describes
its thirteen-orphan special with all seven honor faces. That shape is unavailable in this room because White
Dragons are always flowers and a wildcard cannot replace them.

Use the choices that allow chow and pon, have no forced Burst, and score consecutive replacement events as
one group. The dealer's opening win and a non-dealer's first-circuit Earth win pay ten per opponent. Earth
may be won on a first draw or an eligible discard before the dealer's second draw, while no call or kong has
interrupted play. Take the nearest eligible winner on the same discard. Ordinary later wins require self-draw.
Kong or added kong never opens a robbery window. An existing pon can be upgraded immediately after a call.

An ordinary win or seven pairs pays one with wildcards or two without them. Burst pays two, a kong draw
pays three or five depending on whether the hand has wildcards, and a Burst after a kong or flower pays five.
No flowers and no wildcards, a pure one-suit hand and four wildcards each pay ten. Four wildcards can win
without a complete shape on the holder's draw. With exactly three wildcards, an ordinary Burst or discard win
is unavailable. A kong draw, kong Burst or qualifying float can win. Between independent categories use the
higher payout. A float pays ten per floated wildcard, plus five per uninterrupted kong or flower group before
or after a float. Add ten when the floated wildcards plus those still held reach four. All other seats pay the
winning share, with zero payments on a draw.

## Anji Tea House Self-Draw

The four-seat table uses the eight seasons and plants as its only flower faces. Set aside the indicator
before dealing. A flower used as the indicator leaves seven replaceable flowers. Flowers and White Dragon
point to East. Other indicators use the next suit rank, wind or the other member of the Red–Green pair.
White Dragons, North and the remaining copies of the indicator stay in play. The first dealer is chosen
uniformly, then the winner deals next and the dealer repeats on a draw.

This room uses the operator's self-draw description. Earth applies on a non-dealer's first ordinary draw,
including after another seat has called. All other discard win opportunities are disabled. A pon cannot be
upgraded until a subsequent draw. Standard hands, seven pairs and the loose-honors special can win. Interpret
the operator’s named special using the published Anji tile description of all seven honors and seven suited
tiles, spaced at least three ranks apart in each suit. Wildcards may fill missing members.

Plain self-draw pays one with a wildcard, two without it, or five with neither wildcards nor flowers. Seven
pairs pay two with a wildcard or five without. Heaven, Earth and four wildcards pay ten. This room uses one
point per exposed flower on top of the winning share, since the operator specifies the per-flower addition
without a rate. Apply the highest independent hand value, adding the explicitly listed bonuses for consecutive
kongs, floating and flowers during a float. A draw has zero payments. The rules table gives a price for false
float without a trigger. This room uses the defined ordinary wildcard float.

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
- [Zhejiang Game Hall Suichang hand and payment table](https://www.hzbianqu.com/game/47.html)
- [Zhejiang Game Hall Deqing Wukang and Xinshi scoring tables and multi-win rules](https://www.hzbianqu.com/game/919.html)
- [Zhejiang Game Hall Anji flowers, replacements and scoring](https://www.hzbianqu.com/game/886.html)
- [Game Tea House Anji self-draw, calls and scoring](https://www.5158wan.com/news/201809/7408.html)
- [Zhejiang Game Hall Seven Winds definition for the related Jinhua room](https://www.hzbianqu.com/game/74.html)
