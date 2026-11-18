# Four-seat blood battle rooms

The [Sichuan Xuezhan rules](https://www.hzbianqu.com/game/1100.html) and
[Yunnan Xuezhan rules](https://www.hzbianqu.com/game/1108.html) are operator descriptions.
The two Web room profiles below fix their configurable and underspecified choices.

## Shared play

- 108 suited tiles. The dealer starts with 14; everyone else starts with 13. Each chooses a void suit.
  Discard that suit first. There is no chow and the named suit cannot participate in melds or a win.
  Non-void pons and kongs are allowed before clearing the void suit.
- Only an ordinary discard or an added kong can yield multiple wins. A declared winner stops drawing, calling
  and paying subsequent win and kong bills. Play ends at three winners or after the wall and final calls are exhausted.
  A kong needs a replacement tile. Pon is still permitted after the last draw.
- The first winner deals after a three-winner finish. When the first win event has multiple winners, the discarder or
  robbed kong maker deals. For a wall-exhaustion settlement, the dealer repeats, even if there were earlier wins.
- Each win pays `2 ** min(4, fan)` per share. A discard or a robbery bills the supplier; each still-active
  opponent pays on a self-draw, including a direct-kong replacement win. An open kong bills its supplier 2;
  an added or concealed kong bills each still-active opponent 1 or 2, respectively. A robbed kong earns no kong points.
- Compare uncapped fan for passed wins, including when the previously offered win was a self-draw. A draw or successful
  pon/kong clears the lock. Win legality does not depend on the contents of another seat's concealed hand.
- At wall exhaustion, every unready seat pays every ready, still-active seat its maximum capped **hand-only** value
  and refunds all of its own kong income to the original payers. The wait check tests the hand's shape and rejects a
  fifth copy across one's hand and melds; wall and other players' tiles are irrelevant.

## Sichuan Xuezhan · Exchange Three

- Each seat privately picks three tiles of one suit from its dealt hand. Once all four confirm, the game picks uniformly
  among clockwise, counterclockwise and opposite and swaps all four groups together. Then seats select their void suit.
- The last draw allows pon and ordinary discard wins. The published exception to the pass lock applies to winning on a
  post-kong discard or robbing an added kong. A concealed kong is offered when the matching tile is drawn. If a seat
  chose pon when a direct kong was available, that seat may add the fourth tile on a later draw turn with no kong income.
- A basic hand starts at 0 fan. All triplets add 1, pure suit 2, and seven pairs 2 (3 with at least one quad).
  The regular golden hook adds 1 to triplets, and the pure golden hook is 4 plus the triplets fan. Roots add 1 per quad
  outside seven pairs. Closed hand (including concealed kongs) and all simples each add 1. Simple ranks are checked
  across the entire hand and melds. Terminals and 2/5/8 options are off.
- Self-draw adds 1. An opening dealer self-draw adds 3, and a win on the dealer's first discard or a nondealer's
  first uninterrupted draw adds 2. A kong or completed pon breaks the opening. Kong replacement wins, the next
  discard after a kong, robbery, the last draw, and a win on the discard immediately following the last draw each add 1.
- The trained **Standard** policy belongs to the original Sichuan Xuezhan profile. This room uses the rule-aware
  tile-efficiency policy, which ranks outgoing tiles with its own pre-exchange hand and never observes other selections.

## Yunnan Xuezhan · Flower Pig

- There is no opening exchange. A passed pon locks matching pon offers until a draw or successful pon/kong.
  A discard or robbed-kong winner displays only their winning tile. A self-draw winner retains a concealed hand.
- At a wall-exhaustion settlement, holding any void tile makes a seat a flower pig. Each flower pig pays 16 to
  every non-flower-pig seat, including winners. This room applies ready-hand penalties in addition to the
  flower-pig penalties. A finish with three winners skips both penalties and the kong refunds.
- The operator specifies the patterns, root and kong-event doubles, and a 16-point cap example. This room fixes
  ordinary hand at 0 fan, all triplets at 1, one suit at 2, seven pairs at 2 plus 1 per quad, golden hook
  at 1 in place of all triplets, and four kongs at 6 in place of hook, triplets and roots. Pure suit still adds
  2 to these patterns. Standard hands add 1 for each root. Closed hand, terminals, 2/5/8, opening and last-tile
  extras are off. Self-draw uses the same fan as a discard with the same hand. Robbery, a kong replacement win and
  a win off the immediate post-kong discard each add 1.
