---
title: TanyaoDojo Mahjong
emoji: 🀄
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: agpl-3.0
short_description: Play 33 Mahjong rulesets vs AI. English / 中文.
---

# TanyaoDojo · Mahjong table

Choose **Japanese riichi**, **Sichuan Xuezhan**, **Hongzhong Double Ma**, **Hongzhong One Ma**,
**Fangchenggang Hongzhong**, **Wenzhou Fangpao**, **Changsha Zhuanzhuan**, **Taishun**, **Xuanhua**, **Shaanxi**,
**Lishui Fangpao**, **Wenxi**, **Guilin Zhuanzhuan**, **Pook Dazhong**, **Shanxi Datang Tuidaohu**,
**Hong Kong Old Style**, **Macau Tournament Mahjong**, **Shangsi**, **Hundred-Tile Mahjong**, **Hezhou**,
**Liuzhou Exploding Fish**, **Guangzhou Three-Faan**, **Golden Island Guangdong**, **Guiliu Wake-Tile Mahjong**,
**Sichuan Xuezhan Exchange Three**, **Yunnan Xuezhan Flower Pig**, **Kaxinwu Continuous Wins**,
**Suizhou Ka Wu Xing**, **Xiangyang Ka Wu Xing**, **Shiyan Ka Wu Xing**, **JJ Three-Player Call Points**,
**JJ Three-Player Multipliers**, or **Xiaogan Ka Wu Xing**
and play with two or three AI opponents. Search in either language or
filter by family. Riichi plays a complete East-South match, and Macau plays a 16-hand match over four winds.

- **Beginner:** a rule-aware tile-efficiency policy.
- **Standard:** the TanyaoDojo tpt 408k riichi network or the corrected-rules Sichuan P2 adaptation.
- Click a tile, then confirm the discard. Legal calls, wins, passes and ready declarations appear when available.
- You can autoplay the rest of the current hand, continue to the next hand, or return to the lobby.
- Refreshing the page restores your live table in the same browser tab. Tables expire after inactivity and are lost
  on a server restart. This preview has a bounded number of simultaneous tables.

Only the player's own hand and public information are sent to the browser. Other players' concealed tiles and
wall seeds stay on the server. AI policies receive their own seat's observation, not other seats' hidden hands.

## Language

English is the default. Select English or 简体中文 in the header or dialogs; the preference is remembered.
You can switch during a hand without losing your table or pending choice.

## Rule scope

The **Sichuan Xuezhan** table has mandatory dingque, no chi, multiple winners, a four-fan cap, kong,
ready-hand and refund payments. **Sichuan Xuezhan Exchange Three** adds simultaneous same-suit exchange, and
**Yunnan Xuezhan Flower Pig** adds passed-pon restrictions and flower-pig penalties paid to all unaffected seats.
[Rules and sources](deploy/play/rules/BloodBattle.md).

**Kaxinwu Continuous Wins** uses two suits. All seats keep playing after each win, and a won seat
must discard its drawn tile unless it can win again or make a qualifying kong. The wall-exhaustion
settlement includes ready-hand payments to seats that already won. [Rules](deploy/play/rules/Kaxinwu.md).

The 84-tile **Suizhou** and **Xiangyang Ka Wu Xing** rooms have three seats and open-ready declarations.
They use different caps, discard liabilities and passed-win rules. [Rules](deploy/play/rules/KaWuXing.md).

**Shiyan Ka Wu Xing** limits plain hands to self-draw and allows double discard wins.
**JJ Three-Player Call Points** pays for actual chows and pons and private ready declarations,
then settles wins at five points per payer.
[Shiyan rules](deploy/play/rules/ShiyanKaWuXing.md), [JJ Call Points rules](deploy/play/rules/JJThreePlayer.md).

**JJ Three-Player Multipliers** includes dark ready and public-wait discard restrictions.
**Xiaogan Ka Wu Xing** adds an opening piao choice and a rank horse on revealed self-draws.
[JJ Multipliers rules](deploy/play/rules/JJMultipliers.md), [Xiaogan rules](deploy/play/rules/XiaoganKaWuXing.md).

The **112-tile wildcard family** offers two GameTea bonus modes and the Fangchenggang room with seven pairs and
a four-tile reserve. Source rules and the room choices for unspecified boundaries are listed in the
[rule specification](deploy/play/rules/Hongzhong.md) and the in-game rules dialog.

Ten rooms share a **108-tile discard-win engine**. Shaanxi adds nearest-first wins, Lishui and Wenxi have
separate payout schedules and draw refunds, Guilin reveals wake indicators after a win, and Pook Dazhong
and Shanxi Datang Tuidaohu offer distinct scoring for declared-ready play.
[Rules and sources](deploy/play/rules/SuitedDiscardWin.md).

The **Hong Kong Old Style** room adds 136 tiles with honors, chows, three faan to win, and liability payments.
[Rules and sources](deploy/play/rules/HongKongOldStyle.md).

**Macau Tournament Mahjong** shares the 136-tile setup and adds zero-faan discard wins, two fixed special-hand
tiers, a 14-tile reserve and fixed dealer rotation. [Rules and sources](deploy/play/rules/MacauTournament.md).

Several rooms with **honors and bonus tiles** reveal bonuses from a reserve. Shangsi scores by seat,
Hundred-Tile removes the character suit, Hezhou adds three-feeds liability, and Liuzhou reveals one
fish tile worth its rank or five for an honor. [Rules and sources](deploy/play/rules/HonorsHorseTiles.md).

**Guangzhou Three-Faan Mahjong** caps ordinary hands at three faan and tracks who covers a
self-draw after a fourth open meld or qualifying kong. [Rules and sources](deploy/play/rules/GuangzhouThreeFaan.md).

**Golden Island Guangdong** opens with an exchange of three tiles and scores horses per payer.
**Guiliu** reveals adjacent wake tiles after a win and supports repeated reveals.
[Rules and sources](deploy/play/rules/GuangdongGuiliu.md).

The catalog exposes only implemented, tested rulesets. More regional variants will be added
by rule family. No training runs inside this Space; models are fixed CPU inference snapshots.

## Source and attribution

- App, original SVG tile drawings, and Sichuan implementation: [TanyaoDojo](https://github.com/twu3202/TanyaoDojo).
- Native riichi engine and riichi model architecture: [Mortal](https://github.com/Equim-chan/Mortal), AGPL-3.0-or-later.
  The corresponding Rust source used by this Docker build is included in `native_source/`; it is based on upstream
  commit `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` with the project's rollout bridge changes.
- Riichi checkpoint and model code are reused from the owner's [decision Space](https://huggingface.co/spaces/Twu31/TanyaoDojo-decisions).
- Sichuan weights: project P2 adaptation `seed201`, trained on the corrected draw-turn engine. Against frozen
  P2 b2272 it scored +0.188 ± 0.084 per hand in fixed duplicate 1v3 and +0.182 ± 0.085 on a separate wall panel,
  12,000 games per panel with group-level 95% intervals. Exported parameter arrays were checked against JAX
  for 160 legal CPU action choices before deployment.

See `LICENSE`, `LICENSE-MIT`, `ASSET_MANIFEST.json` and `deploy/play/README.md`. The web adapter is not a substitute
for the project's fixed-protocol research evaluations. No Tenhou game logs or derived datasets are included.
