# Playable Mahjong MVP

Human vs three AI players, with a responsive browser table and a CPU-only server. The ten playable rulesets are
Japanese riichi (complete hanchan), Sichuan Xuezhan, Hongzhong Double Ma, Hongzhong One Ma, Fangchenggang
Hongzhong, Wenzhou Fangpao, Changsha Zhuanzhuan, Taishun, Xuanhua and no-flower Hong Kong Old Style. The regional
rooms offer complete hands with settlement and a next-hand flow.

## Architecture

- `registry.py`: the versioned playable catalog and profile choices. The lobby searches names in both languages
  and filters by stable family IDs.
- `core.py`: bounded per-table worker, revisioned legal choices, cancellation and snapshot isolation.
- `riichi_game.py`: one native `Rollout` per hand, reusing existing Mortal/libriichi legality and settlement.
  Inter-hand continuation uses the first-hand scores/pot, not the truncated game's artificial final payout.
- `sichuan_game.py`: corrected Python reference rules, rule-aware shanten/ukeire baseline and interactive
  robbing-a-kong responses. The training reference auto-accepts those wins; web users can explicitly pass.
- `suited.py`, `suited_game.py`: shared 108-tile first-win engine and browser session, with per-room passed-win,
  robbery and settlement rules. [Rules and operator sources](rules/SuitedDiscardWin.md).
- `honors.py`, `hongkong.py`, `honors_game.py`: 136-tile engine with chows, scoring over all hand decompositions,
  and named payers. `honors_policy.py` ranks legal waits and routes to a three-faan hand using public tiles and the
  existing suit table. [Rules and sources](rules/HongKongOldStyle.md).
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
