# JJ Three-Player · Multipliers

This room follows the [2016 JJ rules](https://www.jj.cn/help/new_72708.shtml). It uses the same
three-seat, 84-tile supply as the [call-points room](JJThreePlayer.md).

- Use characters, dots and the three dragons. The dealer starts with 14 tiles and other seats with 13.
  Pon and kong are allowed, with no chow. A standard hand or seven pairs may win. Added kongs can be robbed.
- At tenpai and with more than three live tiles, choose to reveal the hand or declare **dark ready**.
  Dark ready keeps a maximal set of fixed concealed triplets covered. The engine tests that their exclusion
  as fixed sets preserves the entire predeclaration wait set. A full reveal shows every concealed tile remaining
  after the declared discard. The declaration reveals neither subsequent draws nor the ranks of covered tiles.
- After declaring, discard only the tile drawn, take a win, or kong a covered triplet. A kong is allowed only
  if it preserves the original waits exactly. A full reveal forbids new kongs. A successful covered-triplet kong
  makes that set public. If an undeclared seat has a discard outside other seats' publicly inferable waits,
  it must choose such a discard. The check treats covered triplets as anonymous sets, so it never consults their ranks.
- A passed offered win locks that combination of hand and event patterns until the seat draws. A different
  pattern combination can still win.
- Multiply the supported hand patterns, then add 2. The ordinary multiplier is 1. All triplets, middle-five,
  and kong-event wins double. Pure suit, seven pairs, small three dragons and four exposed sets are ×4;
  big three dragons is ×8. Double the seven-pairs multiplier per concealed quad. Concealed and exposed
  four-in-one values are ×4 and ×2 outside seven pairs, wherever they occur in the hand. Four exposed sets
  replaces the all-triplets multiplier. The full pattern table is in the operator's rules.
- For each payer, double the multiplier before adding 2 if either that payer or the winner has
  declared. Two declarations still double only once. Self-draw bills both opponents; discard or added-kong
  robbery bills the supplier for each winner. Multiple winners use one physical winning tile.
- A direct kong charges its supplier 1. Added and concealed kongs charge each opponent 1 or 2 respectively.
  The second and later consecutive kongs by the same seat double their own payment. At wall exhaustion
  the first revealed seat pays each seat that has not revealed 1. Kong points stand. After a single win the
  winner deals; the supplier deals after multiple wins and the dealer repeats after a draw.
