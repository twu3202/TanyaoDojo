---
title: TanyaoDojo Decisions
emoji: 🀄
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: app.py
license: agpl-3.0
short_description: What would a near-Mortal-v4 riichi model discard?
---

# TanyaoDojo decisions

A "what to discard" (nanikiru) board for riichi mahjong, answered by a Q network trained end to end by
[TanyaoDojo](https://github.com/twu3202/TanyaoDojo): an offline base learned from Tenhou Phoenix-room games, then the
project's own online RL. Measured under duplicate 1v3 against Mortal v4 (100k games, same walls, pt
[90, 45, 0, −135]): **−0.64 ± 0.46** placement points per hanchan.

v4's weights and v4's choices were never trained on: this is not a v4 imitation. v4 did play as one of the
self-play opponents (about 2 of every 7 generated games), and it is the yardstick every number is measured against.

- Click tiles to build the hand (the drawn tile last), the dora indicator and the rivers, or type them in Tenhou
  notation. Then ask the model: it ranks every legal option in one forward pass.
- **Q** is the model's value estimate of each option. **Probability** is softmax(Q / temperature) over the options:
  a Boltzmann policy, not a calibrated chance of being right.
- When riichi is legal, the board also shows which tile the model would discard with riichi.
- `POST /v1/decisions` takes any position as an mjai event log and answers in the shape of the decision-model APIs
  (see the Decision API tab).

For study and post-game review. Do not use it in rated games.

License: AGPL-3.0. The Space links [Mortal](https://github.com/Equim-chan/Mortal)'s libriichi and uses Mortal's model
code (both AGPL-3.0). The libriichi build is the one described in [riichi-eval](https://github.com/twu3202/riichi-eval)
(upstream Mortal plus `rust/rollout.patch`); the app and the decision interface are in
[`deploy/hf_space`](https://github.com/twu3202/TanyaoDojo/tree/master/deploy/hf_space) and
[`jax_rl/mjai_bot/decision_api.py`](https://github.com/twu3202/TanyaoDojo/blob/master/jax_rl/mjai_bot/decision_api.py).
