# Suited discard-win rooms

Four rooms share a four-player, 108-tile state machine. Each allows standard hands and concealed seven pairs
(four identical tiles supply two pairs). The first self-draw or winning response window ends the hand. Ordinary
discards can have multiple winners. Xuanhua limits kong robbery to the nearest player who accepts the offer.

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

## Shared room conventions

No chi, wildcard or void-suit selection. The wall is uniformly shuffled. Except for Taishun's unpaid upgrade,
pon is followed by a discard and an added kong requires the newly drawn fourth tile. A kong needs a replacement
drawn from the far end. Wins take
priority over pon and kong and may be declined. All kong tile identities are visible on the table. On the last
discard, wins remain available and there are no further calls. The same physical tile is counted once on a
multi-win, though each winning hand displays it. Only abstract, zero-sum points are used.
