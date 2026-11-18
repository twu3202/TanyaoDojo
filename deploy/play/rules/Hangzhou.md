# Hangzhou white-dragon wildcard rooms

The catalog has two separate rooms. Both play eight hands at base stake one, using all 136 tiles with no flowers.
The human hosts and deals the first hand. The wall is live to the last tile; a kong needs a replacement left.
Kongs draw from the other end of the wall and have no immediate payment. The winner deals next and the dealer
repeats on a draw. On a draw all deltas are zero. These are the self-draw rooms, including for kong wins.

| Room | Calls | Dealer share | Feed liability |
| --- | --- | --- | --- |
| Three-player, fixed dealer | Pon and kong | Always ×8 | None |
| Four-player, three feeds | Chow from the preceding seat; pon and kong from any seat | ×2, ×4 on first repeat, ×8 on further repeats | Dealer and non-dealer only |

## Wildcard and win multipliers

All four White Dragons are playable wildcards for the concealed part of a hand, including honors. No open meld
or kong uses a wildcard. The last draw is counted separately when checking whether the hand was waiting for
any tile to pair with a wildcard. That wait is a **Burst** (爆头); six pairs with the same wait make a
**Seven-pairs Burst** (七客). A wildcard can fill a set for either route. An ordinary completed wildcard hand
can also win in these rooms.

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

The three-player room blocks the floater's pon and open kongs for the rest of the hand. The four-player room
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

## Sources

- [Zhejiang Game Hall three-player rules and multiplier table](https://www.hzbianqu.com/game/581.html)
- [Zhejiang Game Hall three-player dealer and call rules](https://www.hzbianqu.com/article/1476.html)
- [Zhejiang Game Hall three-player pattern examples](https://www.hzbianqu.com/article/1548.html)
- [Zhejiang Game Hall four-player room options](https://www.hzbianqu.com/game/37.html)
- [Bianfeng GameABC forced discards, feed counting and liability](https://www.gameabc.com/news/201911/6191.html)
- [Zhejiang Game Hall example for reserving a wildcard and floating the second](https://www.hzbianqu.com/game/1611.html)
