# Suited discard-win rooms

These rooms share a four-player, 108-tile state machine. Concealed seven pairs is enabled where listed; four
identical tiles supply two pairs. The first self-draw or winning response window ends the hand.

## Wenzhou Fangpao

Rule ID `wenzhou-fangpao-dojo-v1-20261118`. Source: [GameTea's Wenzhou Fangpao product rules](https://www.gametea.com/games/fangpaomj.html).

The highest applicable base fan is used.

| Fan | Hands |
| ---: | --- |
| 1 | Ordinary four sets and a pair |
| 2 | All triplets |
| 3 | One suit, seven pairs, or golden hook with four called sets |
| 4 | One-suit all triplets |
| 5 | Dragon seven pairs, one-suit seven pairs, or golden hook with all called sets of one suit |
| 6 | One-suit dragon seven pairs |

Dragon seven pairs requires at least one quad. The one-suit golden hook is determined by its called sets,
following the operator's distinction. Self-draw adds one fan in this room; a kong-replacement win uses the
replacement bonus instead. Winning on the discard immediately after a replacement, or robbing an added kong,
also adds one. Heavenly and earthly wins are fixed at six fan here. Heaven is the dealer's initial draw;
earth is a non-dealer's first draw or a discard win before their first draw, with no intervening call or kong.

Each payer owes `2 ** (fan - 1)` points. On self-draw all three opponents pay. For a discard or robbed added
kong, the provider alone pays each winner. The robbery's doubled payment is the extra fan above. Resolved
direct kongs pay 1 from the discarder, added kongs pay 1 from each opponent, and concealed kongs pay 2 from each.
Prior kong payments stand on a draw. Declining a legal discard win locks discard wins in its suit until the
player's next actual draw. This room applies that lock to ordinary discards only.

The first dealer is random. A sole winner deals next; the provider deals after multiple winners. This room
keeps the dealer on an exhaustive draw.

## Changsha Zhuanzhuan (91sui profile)

Rule ID `changsha-zhuanzhuan-91sui-dojo-v1-20261118`. Source: [91sui's Zhuanzhuan operator rules](https://www.91sui.com/Home/GameDetail?id=11).

The payer owes one point plus the winner's completed kong count. All opponents pay for self-draw. The discarder
or added-kong provider is the only payer when an offered tile wins. This profile has no independent kong transfers,
birds, or same-suit passed-win lock. The first dealer is random. The next dealer is chosen randomly among players
who did not win, with all four eligible after a draw. This profile uses the operator's explicit replacement-draw
description for kongs, which takes precedence over a contradictory general bullet on the same page.

## Taishun

Rule ID `taishun-dojo-v1-20261118`. [Zhejiang Game Hall's Taishun rules](https://www.hzbianqu.com/game/63.html).

| Best applicable shape | Discard share | Self-draw share |
| --- | ---: | ---: |
| Ordinary hand | 1 | 2 |
| All triplets | 2 | 4 |
| One suit or seven pairs | 3 | 6 |
| Dragon seven pairs | 6 | 12 |

The dealer's initial self-draw pays 6 from each opponent. A win on the dealer's first discard pays 6 from the
dealer. An ordinary win on the discard after a kong replacement pays a fixed 3. A replacement self-draw doubles
the shape's self-draw share. For an added-kong robbery, this room interprets the extra fan as doubling that same
self-draw share. The kong maker pays three such shares to each winner. Robbery allows multiple winners in this room.

Direct kongs earn 1 from the discarder, drawn-tile added kongs earn 1 from each other seat, and concealed kongs earn
2 from each other seat. The immediate pon of a tile held three times may be upgraded using the held fourth copy.
That upgrade opens a robbery window, requires a replacement draw, and earns no kong points. Finished kong transfers
stand on a draw. Declining an ordinary discard win blocks all ordinary discard wins until the player draws or
successfully claims a pon or kong. An unaccepted or superseded call leaves the lock in place.

The human hosts and deals the first hand. A sole winner deals next. On a multi-win the provider deals next.
On an exhaustive draw the dealer repeats, a room convention for an unspecified case.

## Xuanhua

Rule ID `xuanhua-dojo-v1-20261118`. [Zhejiang Game Hall's Xuanhua rules](https://www.hzbianqu.com/game/309.html).

The published options for multiple discard wins and keeping kong points on a draw are enabled. The option assigning
the entire direct-kong payment to one chair is off. Each opponent pays 1 for a direct or added kong or 2 for a
concealed kong. Each winning hand has base 1 per payer. A closed hand and a self-draw each double the share.
Concealed kongs preserve the closed-hand multiplier. The dealer's initial win pays a fixed 6 from each opponent.
There is no earthly-win premium.

The discarder pays each ordinary winner's share. Only the nearest accepting player can rob an added kong. All
three opponents then pay the self-draw share, including the player who offered the kong. A pass on an ordinary
discard win blocks further ordinary discard wins until that player's next discard. The lock applies only to discard
wins, so the player can accept a later kong robbery. The human hosts and deals first. A sole winner or the discarder
after multiple wins deals next. The dealer repeats on a draw.

## Shaanxi

Rule ID `shaanxi-bianfeng-dojo-v1`. [Bianfeng Shaanxi rules](https://www.hzbianqu.com/game/360.html).

The selected room sets paozi to zero and disables seven pairs. A hand also recognizable as seven pairs may win
if it forms four sets and a pair. Discards and added-kong offers go to the nearest accepting winner. Passing a
discard win locks the same tile only, until drawing, pon or kong. A discard or robbed kong pays 1 from the
provider, and self-draw pays 2 from each opponent. Each win payment involving the dealer doubles. Direct kong
pays 1 from the provider, added kong pays 1 each, and concealed kong pays 2 each. Keep kong points on a draw.
The first dealer is random, the winner deals next following the source's final dealer instruction, and the
dealer repeats on a draw.

## Lishui Fangpao and Wenxi

Rule IDs `lishui-fangpao-dojo-v1` and `wenxi-dojo-v1`. [Lishui room rules](https://www.hzbianqu.com/game/1066.html),
[Wenxi room rules](https://www.hzbianqu.com/game/60.html), [Qingtian scoring comparison](https://www.hzbianqu.com/article/1519.html).

Both allow standard hands and seven pairs and use the highest applicable base share. The single-wait row means
four declared melds and one tile waiting to pair. Four concealed sets includes concealed kongs, following the
[operator's related Lishui definition](https://www.hzbianqu.com/game/1110.html).

| Shape | Lishui self-draw / discard | Wenxi self-draw / discard |
| --- | ---: | ---: |
| Regular | 2 / 1 | 2 / 1 |
| All triplets | 4 / 6 | 4 / 2 |
| Four concealed sets | 12 / 18 | 6 / 3 |
| One suit | 8 / 12 | 6 / 3 |
| Seven pairs | 6 / 9 | 6 / 3 |
| One-suit seven pairs | 12 / 18 | 8 / 4 |
| Dealer initial hand | 16 / — | 8 / — |
| Win on dealer's first discard | — / 20 | — / 4 |
| Four declared melds, single wait | 8 / 12 | 4 / 2 |

An ordinary discard provider pays one share. Each opponent pays a self-draw share. A direct-kong provider
covers three shares on that kong's replacement win. A robbed added kong uses three self-draw shares from its
provider. Wenxi uses the same robbery convention as Lishui, and both allow multiple winners. These rooms
randomize the first dealer, then pass to the winner or the provider after a multi-win, and repeat on draws.

Lishui doubles the share on a replacement win and doubles it on a last-wall self-draw. Wenxi uses the ordinary
self-draw share. A Lishui discard with fewer than four tiles left costs the discarder three self-draw shares.
Three winners on one discard double the final bill, including a last-four bill. Lishui uses the listed table
with the all-flat room and score overrides off. Direct kong pays 3 in Lishui and 1 in Wenxi, from the provider.
Added and concealed kongs pay 1 and 2 respectively from each opponent. Draws reverse all kong transfers.

## Guilin Zhuanzhuan, classic

Rule ID `guilin-zhuanzhuan-classic-dojo-v1`. [Guilin classic rules](https://www.hzbianqu.com/game/1042.html).

This room uses 108 tiles with seven pairs, the upper/lower wake option and base scores without dealer adjustments.
A discard or robbed kong pays 1 from the provider, and self-draw pays 2 from each opponent. Direct kong pays
3 from the provider, added kong pays 1 each and concealed kong pays 2 each. Keep kong transfers on a draw.
A passed ordinary win locks all ordinary discard wins until the player's next draw. You deal first. The winner
deals next, or the provider on multiple wins. Draws keep the dealer.

After a win, reveal an indicator in normal draw order. Its same-suit neighbors, wrapping between 1 and 9,
are the wake tiles. Each matching physical tile in the winner's hand or melds adds 1 to each win share. If a
wake tile matches a kong declared by that winner, extend the reveal. Multiple winners independently use
prefixes of the same indicator sequence. Stop when the current winner no longer extends or the wall is empty.
The results show the shared indicators and each winner's hits.

## Shared room conventions

No chi, wildcard or void-suit selection. The wall is uniformly shuffled. Except for Taishun's unpaid upgrade,
pon is followed by a discard and an added kong requires the newly drawn fourth tile. A kong needs a replacement
drawn from the far end. Wins take
priority over pon and kong and may be declined. All kong tile identities are visible on the table. On the last
discard, wins remain available and there are no further calls. The same physical tile is counted once on a
multi-win, though each winning hand displays it. Only abstract, zero-sum points are used.
