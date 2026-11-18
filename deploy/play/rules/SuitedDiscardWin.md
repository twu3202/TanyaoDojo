# Suited discard-win rooms

Wenzhou Fangpao and Changsha Zhuanzhuan share a four-player, 108-tile state machine. Both allow standard hands
and concealed seven pairs (four identical tiles supply two pairs). The first self-draw, or the first winning
discard or added-kong offer, ends the hand. Every eligible opponent in that response window can win.

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

## Shared room conventions

No chi, wildcard or void-suit selection. The wall is uniformly shuffled. Pon must be followed by a discard.
Added kong uses only the newly drawn fourth tile; a kong needs a replacement drawn from the far end. Wins take
priority over pon and kong and may be declined. All kong tile identities are visible on the table. On the last
discard, wins remain available and there are no further calls. The same physical tile is counted once on a
multi-win, though each winning hand displays it. Only abstract, zero-sum points are used.
