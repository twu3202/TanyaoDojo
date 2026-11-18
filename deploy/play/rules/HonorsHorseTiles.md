# Regional rooms with honors and horse tiles

All three tables deal 13 tiles, then give the dealer the opening draw. Ordinary draws use one end
of the wall. Kong replacement draws use the other end of the live wall and leave the entire bonus
reserve untouched. The last live discard may be won on where the room allows it. No meld calls are
available on that discard and the last live draw cannot begin a kong.

On a win, each winner's calculation uses the same physical reveal. A win contributes one per-payer
base share multiplied by one plus the winner's hits. Kong and opening-discard transfers are shown
separately. The unflipped reserve does not appear in table observations or AI inputs.

## Shangsi Mahjong / 上思麻将

Rule ID `shangsi-discard-4-10-dojo-v1`. [Operator rules](https://www.hzbianqu.com/game/1091.html).

- Four seats, 136 tiles, no chow. This table uses the discard-win, 4/10 horse-tile room with
  added-kong robbery enabled. Standard hands, seven pairs and thirteen orphans all use the same
  base. Self-draw is 2 per opponent. A discard or robbed added kong costs the responsible player
  1 per winner. Only the agreed optional scoring choices in this table are used.
- Reserve ten from the start. Reveal all ten on self-draw and the first four for a discard
  win or robbery. The dealer matches ranks 1/5/9, East and Red. The next seat matches 2/6, South
  and Green. The opposite seat matches 3/7, West and White. The last seat matches 4/8 and North.
- A declined ron locks the same tile until the next draw unless the next offered win has a
  higher base share. Under this table's fixed score, all offered wins have the same base. Passing
  pon locks pon on that same tile until the next draw. Direct kong remains a separate action.
- Direct and added kongs are paid by all three opponents at 1 each. Concealed kong is 2 each.
  These transfers are refunded on a draw. The dealer pays each opponent 1 if the first four
  uninterrupted discards all match, even on a draw or when the fourth discard wins.
- The room creator starts as dealer. The sole winner, or the provider for multiple winners,
  deals next. The dealer repeats on a draw.

## Hundred-Tile Mahjong / 100 张麻将

Rule ID `hundred-tiles-4-horses-dojo-v1`. [Operator rules](https://www.hzbianqu.com/game/1086.html).

- Four seats, dots and bamboo plus all seven honor kinds. No chow, joker, or ordinary discard win.
  Added kong may be robbed by more than one player. A successful robbery cancels the kong transfer.
  A concealed kong displays one face and three backs; the one exposed face determines the type.
- Table choices use four reserved horse tiles, winner-relative matches as in Shangsi and the
  multiplicative bonus. On an added-kong robbery the provider covers three shares per winner.
  The operator description gives both a passed-pon rule and a special rule that disables it.
  This room uses the latter, with no passing locks.
- Base 2. All triplets 4 can combine with a pure suit 8 or a mixed one suit 4. Seven pairs 8
  can combine with a pure suit. All terminals and honors pays 12 without stacking. All honors
  pays 16 without stacking and includes the dragon tiles in this room. Use the highest eligible
  interpretation for ambiguous shapes.
- Direct kong costs its supplier 3. Added and concealed kong cost each opponent 1 or 2, respectively.
  Keep these points on a draw. The sole winner or the provider after multiple robberies deals
  next. The dealer repeats on a draw.

## Hezhou Mahjong / 贺州麻将

Rule ID `hezhou-136-four-yao-horses-dojo-v1`. [Operator rules](https://www.hzbianqu.com/game/1092.html).

- Four seats, 136 tiles, with chow from the previous seat. Jokers are disabled. Ordinary wins
  on a discard and added-kong robbery can be shared. Passing a win or pon blocks that action until
  the next draw; self-draw and otherwise legal direct kongs remain available. The table allows
  discarding the same tile type after a chow.
- Base 1. All triplets, pure suit, seven pairs, closed self-draw and replacement self-draw
  each double the base share. Dragon seven pairs is ×4 in place of seven pairs, and thirteen
  orphans uses 10 on its own. Robbing an added kong or winning on its replacement discard adds
  1 to the share before multiplying by the horse factor.
- The table reserves four horses. A hit is a 1, 5 or 9, East, or Red irrespective of seat.
  Use the multiplicative schedule in the source's settlement section, with no additional
  flat per-hit transfer.
- The third exposed meld from the same supplier creates mutual liability for those two seats.
  On an opponent's self-draw or when feeding that partner a win, the responsible seat pays
  three shares. For a discard from outside the pair, only its supplier pays one. If several
  partners could cover a self-draw, the last established pair takes responsibility.
- Direct kong costs its supplier 1. Added and concealed kong cost each other seat 1. Kong points
  stand on a draw. The sole winner or the multi-win provider deals next. The dealer repeats on a draw.
