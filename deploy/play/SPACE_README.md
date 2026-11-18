---
title: TanyaoDojo Mahjong
emoji: 🀄
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: agpl-3.0
short_description: Play 17 Mahjong rulesets vs AI. English / 中文.
---

# TanyaoDojo · Mahjong table

Choose **Japanese riichi**, **Sichuan Xuezhan**, **Hongzhong Double Ma**, **Hongzhong One Ma**,
**Fangchenggang Hongzhong**, **Wenzhou Fangpao**, **Changsha Zhuanzhuan**, **Taishun**, **Xuanhua**, **Shaanxi**,
**Lishui Fangpao**, **Wenxi**, **Guilin Zhuanzhuan**, **Pook Dazhong**, **Shanxi Datang Tuidaohu**,
**Hong Kong Old Style**, or **Macau Tournament Mahjong**
and play with three AI opponents. Search in either language or
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

The Sichuan rule version has mandatory dingque, no chi, multiple winners, wind/rain kong payments, a four-fan cap,
ready-hand settlement and refunds. It **does not include** exchanging three tiles, call transfer, flower-pig
penalties or every local room option. Those are separate future rule versions, not silently claimed as supported.

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
