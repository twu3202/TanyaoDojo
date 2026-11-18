# Playable Mahjong MVP

Human vs one to three AI players, with a responsive browser table and a CPU-only server. The 96 playable rulesets are
Japanese riichi (complete hanchan), Sichuan Xuezhan, Hongzhong Double Ma, Hongzhong One Ma, Fangchenggang
Hongzhong, Wenzhou Fangpao, Changsha Zhuanzhuan, Taishun, Xuanhua, Shaanxi, Lishui Fangpao, Wenxi, Guilin
Zhuanzhuan, Pook Dazhong, Shanxi Datang Tuidaohu, no-flower Hong Kong Old Style, Macau Tournament Mahjong,
Shangsi, Hundred-Tile Mahjong, Hezhou, Liuzhou Exploding Fish, Guangzhou Three-Faan, Golden Island Guangdong,
Guiliu Wake-Tile Mahjong, Sichuan Xuezhan Exchange Three, Yunnan Xuezhan Flower Pig, Kaxinwu Continuous Wins,
Suizhou Ka Wu Xing, Xiangyang Ka Wu Xing, Shiyan Ka Wu Xing, JJ Three-Player Call Points,
JJ Three-Player Multipliers, Xiaogan Ka Wu Xing, Hangzhou Three-Player, Hangzhou Four-Player, Tonglu,
Chun’an, Hangzhou Two-Player Flip Wildcard, Xiaoshan, Songyang, Suichang, Deqing Wukang, Deqing Xinshi,
Anji Fifteen Flowers, Anji Tea House Self-Draw, Dongyang, Bianfeng Chongji Ten Ten Up, Bianfeng Chongji Seven Gaps and
Tiantai Three-Player 300 Hu, Tiantai White Dragon Wild, Taizhou Four-Player Hu, Daxi Buy the Base, Linhai Six Chong Tiles,
Jinyun Classic, Jinyun Cantonese Style, Jiaxing Raise the Head, Pinghu Flowers and Faan, Quzhou Duanzi,
Quzhou Lata, Quzhou Lata Banked Match, Bengbu Classic, Bengbu East Wind Flowers, Lipu Feed Liability, Wuhu Eight Suit Tiles,
Wuhu Red Dragon 515, Anhui Red Dragon, Fuzhou 36 Flowers, Taiwanese Sixteen Tiles, Longyan New Chapter,
Longyan Half Self-Draw, Nanping Honors and Flowers, Nanping Suits and Flowers, Nanping Red Dragon Gold,
Ningde White Gold, Xianyou White Dragon Substitution, Xianyou White Dragon Flowers, Xianyou Self-Draw,
Quanzhou Triplet and Kong Points, Ningde Random Gold and Flowers, Changchun Shared Bao, Harbin Four Bao Slots,
Jilin City Dice Bao, Shuangliao Pung Multipliers, Shenyang Whirlwind Kongs, Fuxin Middle Five, Fuxin Hidden Ready,
Jianchang Final Split, Liaoyang Joker and Treasure, Panjin Joker Kong, Anshan Open Guns,
Dalian Qionghu Private Bao, Tieling Closed Ready, Changbai Piao and Single Wait,
Chaoyang Town Eggs and Opposite End, Nanchang Double Jing, and Shangrao Bao Self-Draw.
The Macau room plays a complete 16-hand match.

## Architecture

- `jiangxi.py` and `jiangxi_shapes.py`: physically tracked double jing, honor chows and separated hands
  in Nanchang, plus Shangrao's self-draw bao. [Nanchang and Shangrao rules](rules/NanchangShangrao.md).

- `changbai.py` and `chaoyang_town.py`: blind piao, public four-in-one charges, egg waits, and opposite-end liability.
  [Changbai and Chaoyang Town rules](rules/ChangbaiChaoyang.md).

- `dalian_tieling.py`: Dalian's choice of ready locks and private bao, and Tieling's closed ready,
  final split and cross-hand longmao. [Dalian and Tieling rules](rules/DalianTieling.md).

- `liaoning_joker.py`: Panjin's protected joker kong, late kong and supplier liability, and Anshan's
  public gun face and seat-specific bills. [Panjin and Anshan rules](rules/PanjinAnshan.md).

- `liaoning_split.py`: Jianchang's eye waits and dealer draw transfers, and Liaoyang's restricted joker,
  final split draws and post-win treasure. [Jianchang and Liaoyang rules](rules/JianchangLiaoyang.md).

- `shenyang.py` and `fuxin.py`: cross-seat whirlwind scoring, final split draws, middle-five transfers, and face-down declarations.
  [Shenyang and Fuxin rules](rules/ShenyangFuxin.md).
- `jilin_shuangliao.py`: capped dice-bao scoring with seven pairs in Jilin City and stand-alone pung multipliers in Shuangliao.
  [Jilin and Shuangliao rules](rules/JilinShuangliao.md).
- `northeast.py` and `changchun.py`: ready declarations, seat-scoped bao, egg kongs, and separate Harbin and
  Changchun settlement. [Northeastern bao rules](rules/ChangchunHarbinBao.md).
- `fujian_travel.py` supports exact triplet and kong weights, immediate third gold travel, and flower-water scoring.
  [Quanzhou and Ningde rules](rules/QuanzhouNingdeGold.md).
- `xianyou.py` shares five-set gold travel between two White Dragon treatments and the 112-tile self-draw game.
  [Xianyou rules and room conventions](rules/Xianyou.md).
- `fujian_sixteen.py`: three Nanping tile sets and payment schedules, plus Ningde's fixed White Dragon
  gold and pairwise loser transfers. [Fujian sixteen-tile rooms](rules/NanpingNingde.md).
- `longyan.py` and `gold_opening.py`: opening gold priority, White Dragon substitutions, traveling gold,
  follow-discard payments and two different win and robbery profiles. [Longyan room rules](rules/Longyan.md).
- `sixteen.py`, `fuzhou.py` and `taiwan.py`: shared 16/17-tile opening and five-set progression, with opening gold
  and flower-win rules. [Rules and room conventions](rules/SixteenTileMahjong.md).
- `wuhu.py` and `hongzhong.py`: eight-suit-tile gates, decomposition-dependent scoring and all-payer or
  third-pon liability, plus Anhui's natural-hand robbery and reserved horses. [Anhui rules](rules/WuhuAnhui.md).
- `bengbu.py` and `lipu.py`: seat-timed flower replacement, dealer surcharges, reciprocal feed liability,
  sealed wins and reversible kong payments. [Rules and room conventions](rules/BengbuLipu.md).
- `quzhou.py`: strict and flexible separated hands, position-sensitive payments, and a 30-point banked match.
  [Rules and room conventions](rules/Quzhou.md).
- `jiaxing.py` and `pinghu.py`: same-face wildcards, with carried faan and three-feed choices in Jiaxing and
  flower-count scoring in Pinghu. [Rules and room conventions](rules/JiaxingPinghu.md).
- `jinyun.py`: opening follow-tile and dealer-weighted settlement in Classic, and self-draw, kong liability and draw refunds
  in Cantonese Style. [Rules and room conventions](rules/Jinyun.md).
- `daxi.py` and `linhai.py`: opening buy-ins and supplier-specific kongs in Daxi, and wildcards, reciprocal feed liability
  and six post-win chong tiles in Linhai. [Rules and room conventions](rules/DaxiLinhai.md).
- `registry.py`: the versioned playable catalog and profile choices. The lobby searches names in both languages
  and filters by stable family IDs.
- `hangzhou.py` and `wildcard_shapes.py`: exact concealed wildcard shapes, Burst, floating and payer-specific
  dealer and three-feed settlement over the honors engine. [Rules and sources](rules/Hangzhou.md).
- `core.py`: bounded per-table worker, revisioned legal choices, cancellation and snapshot isolation.
- `riichi_game.py`: one native `Rollout` per hand, reusing existing Mortal/libriichi legality and settlement.
  Inter-hand continuation uses the first-hand scores/pot, not the truncated game's artificial final payout.
- `sichuan_game.py`: corrected Python reference rules, rule-aware shanten/ukeire baseline and interactive
  robbing-a-kong responses. The training reference auto-accepts those wins; web users can explicitly pass.
- `blood_battle.py`: web rules for simultaneous same-suit exchange, passed calls, multi-win dealer succession
  and Yunnan's flower-pig, ready-hand and kong-refund settlements.
  [Sichuan and Yunnan rules](rules/BloodBattle.md).
- `suited.py`, `suited_game.py`, `suited_regional.py`: shared 108-tile first-win engine, with per-room passed-win
  rules, ready declarations, payout schedules, kong refunds and Guilin wake indicators.
  [Rules and operator sources](rules/SuitedDiscardWin.md).
- `kaxinwu.py`: 72-tile continuous-win room over the suited call engine. Tracks repeated wins, locked draws,
  kong chains and ready-hand payments to both ready and previously winning seats. [Rules](rules/Kaxinwu.md).
- `honors.py`, `hongkong.py`, `macau.py`, `guangzhou.py`, `honors_game.py`: 136-tile engine with chows, scoring over all hand decompositions,
  and named payers. `honors_policy.py` ranks legal waits and routes to a three-faan hand using public tiles and the
  existing suit table. Rules for [Hong Kong](rules/HongKongOldStyle.md), [Macau](rules/MacauTournament.md)
  and [Guangzhou](rules/GuangzhouThreeFaan.md).
- `kawuxing.py`: three-seat, 84-tile rooms using the honors engine and its tile-efficiency AI.
  Includes open ready, supplier liability, first-reveal draw payments and Shiyan's minimum discard-win value.
  [Suizhou and Xiangyang rules](rules/KaWuXing.md), [Shiyan rules](rules/ShiyanKaWuXing.md).
- `jj_three.py`: three-seat chows, pons, private ready and immediate rewards, with a fixed win share.
  [JJ Call Points rules](rules/JJThreePlayer.md).
- `jj_multiplier.py` and `xiaogan.py`: partly revealed ready hands and public-wait discard restrictions,
  or private opening piao choices and rank horses. [JJ Multipliers rules](rules/JJMultipliers.md),
  [Xiaogan rules](rules/XiaoganKaWuXing.md).
- `honors_bonus.py`: regional point schedules with reserved horse tiles, configurable tile sets, immediate kong
  transfers, and shared bonus reveals on multi-win hands.
  [Shangsi, Hundred-Tile, Hezhou and Liuzhou rules](rules/HonorsHorseTiles.md).
- `golden_guangdong.py` and `guiliu.py`: exchange three at the opening, payer-specific horse multipliers,
  adjacent wake reveals and reciprocal settlement. [Golden Island and Guiliu rules](rules/GuangdongGuiliu.md).
- `policies.py`, `sichuan_model.py`: existing tpt 408k and Sichuan P2 adaptation `seed201`, CPU inference only. The Sichuan Torch
  port and NumPy observation encoder are cross-checked against the original JAX network/encoder.
- `server.py`: one HTTP port, session-capability IDs, server-side legal-choice validation, bounded request sizes.
- `static/`: original SVG tiles and a dependency-free responsive client.

The browser never receives wall seeds, the true wall, raw native logs or concealed opponent hands. Policies
receive their own seat observation and legal actions. Sessions are in-memory and browser-tab recovery uses
sessionStorage; server restarts lose games. There is a 32-table cap and inactivity expiry. No accounts, billing,
matchmaking or persistent game storage are included in the MVP.

## Local run

Use an existing Python environment with NumPy and Torch. `libriichi` must match that Python ABI and OS.
Set `TANYAO_NATIVE_DIR` to the directory containing the native `libriichi.so`, and `TANYAO_ASSETS` to the directory
containing `model.py`, the existing riichi `model.pth`, and the exported Sichuan `sichuan.npz`.

```bash
HOST=127.0.0.1 PORT=8765 python -m deploy.play.server
```

Missing model assets remove that model difficulty from the catalog; they never silently masquerade as a model.
The riichi native engine remains required for riichi play. Do not point the source loader at arbitrary user files.

## Checks

```bash
python -m unittest -v deploy.play.tests.test_games
# In the existing JAX environment, then in the Torch serving environment:
python deploy/play/tests/model_parity.py generate
python deploy/play/tests/model_parity.py verify
```

Covered: complete riichi hanchan; pot conservation between hands; red-five chi choices; Sichuan multiple rounds,
manual rob-kong pass/win; stale/illegal actions; hidden-information invariance. Browser checks cover manual discard,
refresh/recovery, autoplay, settlement, next hand, switching variants and a 390 px mobile layout.

## HF Space build

`build_space.py` creates an allowlisted build context in a new directory (normally beneath gitignored `runs/`).
It includes the exact native Rust source and builds that source for the Docker runtime, rather than reusing an
incompatible platform binary. It copies only the model assets and code needed by the game, with license notices
and SHA256 manifest; no game logs, training datasets, credentials or working-tree metadata are packaged.

```bash
python deploy/play/build_space.py OUTPUT --assets ASSETS --native-source MORTAL_SOURCE
```

Publish that context as a Docker Space, port 7860. The runtime is CPU inference; it must not start training.
The catalog exposes only implemented rulesets, not a placeholder list of unimplemented regional names. Future rule
families should bring independent legality/settlement tests before being added to the catalog.


## Language and the first wildcard-family expansion (2026-11-16)

English is the default, regardless of browser locale. English and Simplified Chinese are selectable in the header
and dialogs. The preference is persisted in localStorage. Rules, action prompts, result panels, tile alternatives,
public event messages and error codes are localized; switching language does not submit an action or restart a hand.
Backend events carry stable codes and parameters rather than language-specific UI strings.

`hongzhong.py` implements the three profiles in the 112-tile wildcard family. `hongzhong_game.py` provides the
public-information heuristic and web session. See [the rule specification](rules/Hongzhong.md) for scoring and
room choices, including the Fangchenggang reserve and seven-pairs differences.

The web Sichuan adapter now records an explicit post-pon turn: it requires a discard and suppresses the drawn-tile
feature. A claimed triplet must not be reclassified as a self-draw. The legacy training environment inferred this
from hand length, so parity testing covers ordinary states and tests this intentional public-information correction
separately. The parameter file itself is unchanged. Future training needs the matching turn-context correction.
