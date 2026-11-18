# Sixteen-Tile Mahjong

Both four-player rooms deal sixteen tiles to each player plus one to the dealer, use five sets and a pair
for ordinary wins, and finish after eight hands. Chows come from the immediately preceding seat. Pon and
kong can use discards from other players. The first accepting winner after the discarder takes the tile.
The dealer stays after a draw or dealer win. Otherwise the next seat deals. Both tables use a fixed reserve
and tail-side replacements immediately outside it. The AI uses its own hand, the public calls, flower
and gold information, and five-set shape and live-tile estimates.

## Fuzhou 36 Flowers

The [operator flow](https://www.hzbianqu.com/game/160.html) and
[Fuzhou language reference](https://resource.zingzeu.org/kb/word-lists/mahjong) describe the flower rounds
and opening gold claim. The [published Fuzhou scores](https://xinyueyouxi.com/game/216-22?channel_id=0&product_id=33&region=)
give the flower and gold points and special awards.

- All winds, dragons and the eight single flowers are replaced. At the opening, each seat replaces only
  the flowers it starts its turn with. After all seats have had a turn, return to the dealer for a further
  pass if needed. Afterward, the dealer exposes any flowers reached while turning the first suited gold.
  The displayed tile occupies a public indicator slot. The three remaining copies are wild.
- A seat dealt all three golds gets its opening win choice first. Each other seat has one opportunity to
  take the indicator. The dealer must exchange it for a non-gold tile; a nondealer adds it to sixteen tiles.
  Acceptance immediately ends the hand. The Dojo order for competing claims begins at the dealer and
  proceeds in seat order. Both opening choices use the self-draw payment schedule. A gold claim earns its
  30-point award even when the added indicator is the third gold in the completed hand.
- A gold cannot be called or used as a natural component of an exposed chow, pon or kong. A seat that
  discards a gold may only self-draw until its next ordinary draw. A ready hand with a free gold to form
  a pair with any draw also requires self-draw. After passing a discard win, the same face is locked until the
  next draw. This room does not rob kongs and uses a fixed eighteen-tile reserve.
- Each flower, gold, exposed kong and dealer repeat adds one. Each concealed kong adds two. Multiply this
  subtotal by two for self-draw and add the largest special award. A discard win charges the supplier once;
  other winners receive one share from each opponent. The scoring schedule has no separate standalone kong
  transfers.

| Special | Points |
| --- | ---: |
| Dealer opening win, opening gold claim, or a win with no flowers and no kongs | 30 |
| One flower | 15 |
| Three golds, with no other shape required on a draw | 40 |
| Two gold tiles as the pair on a self-draw | 60 |
| Three gold tiles as a natural pung on a self-draw | 120 |
| Pure suit | 240 |

## Taiwanese Sixteen Tiles

The table follows the [published 16/17-tile and eight-flower deal](https://www.mjweb.com.tw/htm/rule_mj.htm)
and uses the [flower-win, hand and dealer-tai schedule](https://diam.ngct.net/mj_rules.php) for the awards below.
The Dojo room uses a fixed sixteen-tile reserve, accepts a direct kong from any opponent, and ends after a
fixed number of hands.

- A player who has all eight flowers can claim a win against the other three seats. Seven flowers can
  claim against the holder of the eighth. Each is worth eight tai without a hand-shape requirement. Flowers
  stay with the seats that drew them. Declining the flower win resumes play. During play, concealed kong
  faces are hidden from opponents.
- An added kong can be robbed. An immediate discard of the face just called in a chow is forbidden.
  After passing a discard or added-kong win, or discarding instead of accepting a self-draw, a player must
  complete an ordinary draw that could not win and discard before requalifying. A flower win also observes
  this lock.
- Each payment is three points plus tai. Dealer involvement adds one tai plus two for each preceding hand
  of the current dealer streak. On a discard or kong robbery only the supplier pays; on self-draw each
  opponent pays individually. Dealer opening win and win on the dealer's first discard are 32 and 16 tai and
  replace other hand awards. The dealer's own payment adjustment still applies.
- Other awards include closed self-draw (three), strict two-sided all-chow discard win (two), single wait (one),
  seat flower (one), a complete four-flower group (two), hand pungs and honors, and latest wall draw or discard.
  A self-draw after a flower, concealed kong or added kong receives one replacement tai. Dealer, seat wind
  and round wind are tracked separately. The room has no opening ready declaration or separate four-kong win.
