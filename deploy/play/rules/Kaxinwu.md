# Kaxinwu · Continuous Wins

This room follows the [operator's four-player, two-suit rules](https://www.hzbianqu.com/game/1122.html).
It fixes the choices below for a single published room.

## Play after the first win

- Use 72 tiles, four copies of ranks 1–9 in dots and bamboo. There is no void suit or chow.
  An ordinary completed hand or seven pairs can win by self-draw, discard or robbing an added kong.
- The dealer starts with 14 tiles, and the other seats start with 13. All four continue playing after every win.
  A seat that has won discards only its drawn tile and cannot pon. The table automatically discards that
  tile when it is the only legal action. The winning tile is shown separately;
  a multi-win moves one physical tile to the win display and pays each winner independently.
- A single win passes play to the seat after the drawing or supplying seat. If multiple players win on a discard
  or a robbed added kong, that supplier draws again. The final drawer deals next.
- Passing a discard or robbing win prevents another offer of the same tile at equal or lower fan until a draw or a
  completed pon/kong. After passing a self-draw, one equal-fan offer is available before the next such action.
  Higher-fan offers remain eligible; lower offers are blocked for that cycle. Passing a pon locks only that tile.
- Before winning, open and added kongs are optional. An added or concealed kong may be called immediately
  after pon; a player who chose pon when they could have made an open kong cannot later upgrade that pon.
  After winning, concealed and added kongs may use the drawn tile only. A kong must retain all prior winning
  tile types as waits. Other waits can disappear, as in the operator's example. An eligible open kong
  after winning is mandatory when no one takes the win. Added and concealed kongs remain optional.
- Kongs use a replacement tile. Once the wall is empty, resolve wins on the final discard and then settle.

## Points

- A win pays `2 ** total_fan` points per payer. A discard or robbery bills the supplier; on self-draw all three
  opponents pay, including earlier winners. Choose one base pattern: regular hand 1, all triplets or middle-five
  wait 2, seven pairs 2 plus one for each concealed quad. A middle-five wait needs the winning five to complete
  a 4–5–6 sequence in a valid hand decomposition.
- Pure suit adds 1. Each concealed, unmelded quad adds 1 outside seven pairs. Robbing an added kong and a
  win on the discard immediately after a kong each add 1. A kong replacement self-draw adds one fan per
  consecutive kong in that chain.
- An open kong bills its supplier 2; an added kong bills each opponent 1; a concealed kong bills each 2.
  Each consecutive kong after the first doubles its own kong payment. A robbed added kong earns nothing
  and its existing pon remains in play.
- At wall exhaustion, every unready seat pays each ready seat the maximum applicable basic value and each
  winning seat its highest basic value from wins already taken. Include pure suit; exclude concealed quad
  bonuses and event bonuses. Use structural waits, excluding fifth copies across one's hand and melds.
  Points from earlier wins and kongs stand. The result shows each win, the kong totals, and the ready-hand payments.
