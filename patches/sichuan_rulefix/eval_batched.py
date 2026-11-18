"""Batch the JAX shadow and network calls; keep Python rules and bot decisions unchanged."""

import random

import jax
import jax.numpy as jnp
import numpy as np

from sichuan import env_jax, eval_h2h as E


def _public_shadow(state):
    return state.current_player, state.legal_action_mask, state.drawn_tile, state.terminated, state.score


@jax.jit
def _init(walls):
    state = jax.vmap(env_jax._init_from_wall)(walls)
    return state, _public_shadow(state)


@jax.jit
def _step(state, actions):
    state = jax.vmap(env_jax._step_core)(state, actions)
    return state, _public_shadow(state)


class BatchedMatches:
    def __init__(self, candidate, opponent, groups_per_batch):
        self.policies = (candidate, opponent)
        self.batch_size = 4 * groups_per_batch
        self.net_picks = [jax.jit(jax.vmap(fn)) if kind == 'net' else None for kind, fn in self.policies]

    def play(self, seeds, traces=None):
        games = []
        walls = []
        rngs = []
        for seed in seeds:
            wall = [t for t in range(env_jax.NUM_TILES) for _ in range(4)]
            random.Random(seed).shuffle(wall)
            for seat in range(4):
                games.append(E.SichuanGame(seed))
                walls.append(wall)
                rngs.append(random.Random(seed * 4 + seat))
        if not 0 < len(games) <= self.batch_size:
            raise ValueError('invalid evaluation batch size')
        walls.extend([walls[-1]] * (self.batch_size - len(games)))
        state, shadow = _init(jnp.asarray(walls, dtype=jnp.int8))
        active = list(range(len(games)))

        for _ in range(4000):
            current, masks, drawn, terminated, scores = jax.device_get(shadow)
            actions = np.zeros(self.batch_size, dtype=np.int32)
            needing_model = [[], []]
            remaining = []
            for index in active:
                game = games[index]
                if game.phase == 'over':
                    if not terminated[index] or not np.array_equal(scores[index], game.scores()):
                        raise RuntimeError(f'shadow terminal score mismatch on table {index}')
                    if sum(game.scores()) != 0 or game.tile_conservation() != 108:
                        raise RuntimeError(f'non-conserving terminal game on table {index}')
                    continue

                player, legal = game.legal_actions()
                by_id = {E.ref_to_id(action): action for action in legal}
                expected_draw = -1 if game.drawn_tile is None else game.drawn_tile
                if (terminated[index] or player != current[index] or expected_draw != drawn[index]
                        or sorted(by_id) != np.flatnonzero(masks[index]).tolist()):
                    E.STATS['mask_mismatch'] += 1
                    raise RuntimeError(f'shadow state mismatch on table {index}, turn {game.turn_guard}')
                policy_ix = int(player != index % 4)
                kind, fn = self.policies[policy_ix]
                remaining.append(index)
                if kind == 'net':
                    E.STATS['net_decisions'] += 1
                    if len(by_id) > 1:
                        needing_model[policy_ix].append((index, player, by_id))
                        continue
                    action = legal[0]
                else:
                    action = fn(game, player, legal, rngs[index])
                actions[index] = E.ref_to_id(action)
                if traces is not None:
                    traces[index].append((player, action))
                game.step(action)

            if not remaining:
                return np.asarray([game.scores() for game in games]).reshape(-1, 4, 4)

            for pick, pending in zip(self.net_picks, needing_model):
                if not pending:
                    continue
                choices = np.asarray(pick(state))
                for index, player, by_id in pending:
                    aid = int(choices[index])
                    if aid not in by_id:
                        E.STATS['fallback'] += 1
                        raise RuntimeError(f'network chose illegal action {aid} on table {index}')
                    actions[index] = aid
                    if traces is not None:
                        traces[index].append((player, by_id[aid]))
                    games[index].step(by_id[aid])
            state, shadow = _step(state, actions)
            active = remaining
        raise RuntimeError('batched games did not terminate')
