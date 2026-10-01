# Licensing (dual-licensed by directory)

| Path | License | Reason |
|---|---|---|
| All code in the repository except the directory below<br/>(the environment / observation / network / training / data-bridge code in `jax_rl/`, `scripts/`, `configs/`, the docs) | **MIT** | Original to this project, with no copyleft dependencies |
| `jax_rl/mjai_bot/` | **AGPL-3.0** | Links `libriichi` from upstream [Mortal](https://github.com/Equim-chan/Mortal) (AGPL-3.0) at runtime |

## Notes

- The evaluation bridge (`jax_rl/mjai_bot/`) uses the upstream game engine and legality checks through `import libriichi`;
  under the linking terms of AGPL-3.0, that directory is released under AGPL-3.0.
- The upstream Mortal source is **not distributed with this repository**; clone and build it yourself following [SETUP.md](SETUP.md).
  Its license and copyright notices belong to upstream.
- The training/inference core (the Mahjax environment adaptation, the `obs_lean`/`obs_v2` observations, the `net_lean` network,
  the `bc_stream`/`ppo_*` trainers, and `data_bridge` log replay) does not depend on libriichi and is provided under MIT.
- This project does not distribute Tenhou game logs or any datasets derived from them (see SETUP.md §4).

*This note does not constitute legal advice.*
