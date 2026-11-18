# Macau Tournament Mahjong

Rule ID `macau-alumni-2011-dojo-v1-20261118`. The room follows the 2011
[University of Macau Alumni Association tournament rules](https://www.aaaum.org.mo/wp-content/uploads/2011/06/mahjong_rules_2011.pdf)
with the digital table conventions below.

## Play and match

Four players use 136 tiles without flowers. Four sets and a pair or Thirteen Orphans can win at any faan value.
Chow comes from the preceding player. Wins beat other claims, and pong and direct kong take priority over chow.
Add a kong only after drawing. Only Thirteen Orphans can rob a concealed kong. An added kong can be robbed by
any legal winning hand. Kongs draw replacements from the opposite end, with 14 tiles left out of ordinary draws.
The last live draw cannot be konged. The resulting last discard can only be claimed to win. A draw pays zero.

The first dealer is random. The dealer advances every hand, including on a win or draw. East, South, West and
North each have four hands. The room plays all 16 without a physical tournament clock. Table and seat winds are
both used in scoring. The highest cumulative score wins, with ties allowed.

The digital table allows multiple wins from the same discard or kong offer, and does not impose an additional
same-tile pass restriction or a post-chow swap restriction. The written rules leave these cases unspecified.
Initial-hand wins require an uninterrupted opening. The engine prevents illegal claims that the physical table
would handle as officiating decisions.

## Faan and payments

The engine uses the highest-scoring valid interpretation. Ordinary scores cap at 6 faan. A zero-faan discard
win costs 5 points. From 1 through 6 faan the discarder pays 20 points per faan, or each opponent pays 10 points
per faan for self-draw. A robbed kong uses the discard column, with the kong maker as payer.

| Faan or tier | Examples | Discard payer | Self-draw share |
| --- | --- | ---: | ---: |
| 1 per pattern | Closed hand, all sequences, self-draw, each value pung, pure straight, replacement win, added-kong robbery | 20 per faan | 10 per faan |
| 3 per pattern | Half flush, all triplets, mixed terminals and honors | 60 at 3 | 30 at 3 |
| 6 per pattern | Little Three Dragons, full flush, three kongs | 120 at cap | 60 at cap |
| Special A | Big Three Dragons, Little Four Winds, Thirteen Orphans, four concealed triplets, all honors | 500 | 250 |
| Special B | Pure terminals, Nine Gates, Big Four Winds, four kongs, one of the three opening wins | 1,000 | 500 |

Nine Gates requires the complete nine-sided wait before receiving the winning tile. Opening wins cover the
dealer's dealt hand, the dealer's first discard, and an opponent's first natural draw before any accepted call
or kong. Ordinary faan add together before the cap. Little Three Dragons includes its matching dragon pungs in
its six-faan value. Matching seat and round winds each score. Special tier B takes precedence when both apply.

The player who provides the decisive exposed meld can owe all three self-draw shares for full flush with four
exposed sets, Big Three Dragons, Big Four Winds, or four kongs. This applies only if the corresponding pattern
actually wins by self-draw. The most recent qualifying provider pays if several liabilities match. Kongs have
no separate transfer. The payment dialog shows each actual payer.
