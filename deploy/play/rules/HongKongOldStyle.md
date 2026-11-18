# Hong Kong Old Style, no flowers

Rule ID `hong-kong-old-style-no-flower-dojo-v1-20261118`. The principal source is the
[CUHK Alumni Cup no-flower tournament sheet](https://home.culhk.org/files/ICUD/2016ICUD-Mahjong_Comp-Regulation.pdf).
The [Hong Kong Mahjong Association rule index](https://www.hkmahjong.org/rules?lang=en) supplies definitions for
shared terms and patterns. This room publishes its choices for details the tournament sheet leaves open.

## Play and calls

- Four seats use 136 tiles. The dealer starts with 14; each other seat has 13. The first dealer is random.
- Wins take priority, including multiple winners on one discard. Pong and direct kong outrank chow, which is
  available only to the next seat. After a chow, the called tile and the extension at the opposite end of the
  sequence cannot be discarded immediately. For example, calling 1-2-3 on a 1 blocks discarding 1 or 4.
- Passing an available win, or declining an available chow, pong or direct kong, locks that same tile for the
  corresponding action until that player's actual draw. The other tile of a two-sided wait remains eligible.
- An added kong opens a robbery window. Only Thirteen Orphans can rob a concealed kong. When multiple players
  accept a robbery, the kong maker pays each winner. An unrobbed kong draws a replacement from the far end.
- Take the final live tile. Wins remain available on the last discard. This room offers no further melds after
  that discard. A player may declare a kong on the last draw; if nobody robs it, the hand ends in a draw.
- The dealer repeats on a win, including a shared win, and the seat passes after any other result. The prevailing
  wind advances every four dealer changes. A draw scores zero. The next hand starts with cumulative scores.

## Winning hands

Four sets and a pair or Thirteen Orphans may win at three faan or above. The app checks every valid decomposition
and uses the highest score, capped at eight. Scoring labels below describe the rule terms in this digital room.

| Faan | Patterns and events |
| ---: | --- |
| 1 each | All sequences, closed hand, self-draw, kong replacement self-draw, kong robbery, each dragon triplet, seat wind triplet and prevailing wind triplet |
| 3 each | All triplets, one suit with honors, mixed terminals and honors |
| 4 | Little Three Dragons |
| 7 | One suit without honors |
| 8, fixed | Big or Little Four Winds, Big Three Dragons, four concealed triplets, Thirteen Orphans, all terminals, all honors, four kongs, winning on a second or later consecutive replacement, robbing a consecutive kong, dealer's initial hand or first discard |

This room permits the relevant bonuses to add together, including seat and prevailing wind when they match.
Little Three Dragons includes the two dragon bonuses in its four faan. A concealed kong keeps the hand closed.
Four concealed triplets requires no declared kongs and a self-draw or a discard that finishes the pair.
A replacement self-draw adds its bonus alongside ordinary self-draw. Fixed-score hands take the eight-faan cap.

## Payments

| Faan | Self-draw share from each opponent | Discard payment from provider |
| ---: | ---: | ---: |
| 3 | 16 | 32 |
| 4 | 32 | 64 |
| 5 | 48 | 96 |
| 6 | 64 | 128 |
| 7 | 96 | 192 |
| 8 | 128 | 256 |

The supplier of a player's fourth declared set or third exposed dragon set covers all three shares if that player
later wins by self-draw. Previously declared concealed kongs count toward four sets. Subsequent liability replaces
earlier liability. A discard winner collects only from the discarder; a robbed-kong winner collects three self-draw
shares from the kong maker. Kongs have no stand-alone payments. The interface prevents illegal claims and removes
the need for physical-table call timing, dice placement and false-win penalties.
