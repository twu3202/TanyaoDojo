"""Web table and shared tile-efficiency policy for suited discard-win rooms."""

from collections import Counter
from dataclasses import asdict
import time

from .core import Closed, GameSession, tile_name
from .errors import ClientError
from .kaxinwu import VARIANT as KAXINWU, KaxinwuGame
from .sichuan_game import heuristic
from .suited import PASS, Profile, SuitedGame, _waits


def policy_view(game, actor, acts):
    visible = Counter(tile for river in game.discards for tile in river)
    for player in game.players:
        for kind, tile in player.melds:
            visible[tile] += 3 if kind == 'peng' else 4
    offer = game.offer
    if offer is not None and offer.kind == 'rob_kong':
        visible[offer.tile] += 1
    player = game.players[actor]
    view = {
        'hand': player.hand.copy(), 'melds': player.melds.copy(), 'void': None,
        'allow_seven_pairs': game.rules.allow_seven_pairs,
        'actions': acts.copy(), 'public': dict(visible),
        'pending_tile': None if offer is None else offer.tile,
        'opponents': [{'void': None, 'finished': False, 'melds': len(other.melds)}
                      for seat, other in enumerate(game.players) if seat != actor],
    }
    if isinstance(game, KaxinwuGame):
        visible.update(game.won_tiles)
        view.update(void=0, own_kong=True, public=dict(visible))
    return view


def suited_policy(view):
    move = heuristic(view)
    ready = ('declare_ready', move[1])
    if move[0] != 'discard' or ready not in view['actions']:
        return move
    hand = view['hand'].copy()
    hand[move[1]] -= 1
    waits = _waits(tuple(hand), tuple(view['melds']))
    if any(view['hand'][tile] + view['public'].get(tile, 0) < 4 for tile in waits):
        return ready
    return move


class SuitedSession(GameSession):
    def __init__(self, policy='heuristic', *, model=None, seed=None, pacing=.12, profile):
        self.profile = None if profile == KAXINWU else Profile(profile)
        variant = KAXINWU if self.profile is None else self.profile.value
        super().__init__(variant, policy, seed=seed, pacing=pacing)
        self.round_no = 0
        self.totals = [0] * 4
        self.next_dealer = None
        self.game = None

    def begin(self):
        if self.thread is not None and self.thread.is_alive() and self.status == 'round_end':
            self.thread.join(timeout=1)
        with self.cv:
            if self.status == 'starting' and self.thread is not None:
                raise ClientError('round_starting')
            if self.status not in ('starting', 'round_end'):
                raise ClientError('round_active')
            self.round_no += 1
            seed = self.seed + self.round_no
            if self.variant == KAXINWU:
                self.game = KaxinwuGame(seed, self.next_dealer)
            else:
                self.game = SuitedGame(seed, self.next_dealer, profile=self.profile)
            self.result = None
            self.options = []
            self.automatic = False
            self.status = 'starting'
            self.version += 1
            self.log_event('hand_start', kind='round')
            self.start_worker(self.play_round)

    def view(self):
        game = self.game
        players = []
        for seat, player in enumerate(game.players):
            revealed = seat in game.revealed
            hand = game.revealed.get(seat, player.hand)
            players.append({
                'id': seat, 'score': self.totals[seat] + player.score, 'delta': player.score,
                'hand_count': sum(hand),
                'hand': [tile_name(tile) for tile, count in enumerate(hand) for _ in range(count)]
                        if seat == 0 or revealed else [],
                'revealed': seat == 0 or revealed, 'won': revealed, 'riichi': False, 'void_suit': None,
                'river': [{'tile': tile_name(tile), **({'ready': True} if index == player.ready_discard else {})}
                          for index, tile in enumerate(game.discards[seat])],
                'melds': [{'kind': kind, 'tiles': [tile_name(tile)] * (3 if kind == 'peng' else 4)}
                          for kind, tile in player.melds],
            })
            if game.rules.declare_ready:
                players[-1]['ready'] = bool(player.ready_waits)
            if isinstance(game, KaxinwuGame):
                players[-1]['win_count'] = game.win_counts[seat]
                players[-1]['won'] = bool(game.win_counts[seat])
        actor, _ = game.legal_actions()
        if actor != 0 and game.offer is not None:
            actor = None
        drawn = tile_name(game.drawn_tile) if actor == 0 and game.drawn_tile is not None else None
        view = {'players': players, 'hand_number': self.round_no, 'dealer': game.dealer,
                'current': actor, 'remaining': len(game.wall), 'dora': [], 'honba': 0,
                'pot': 0, 'phase': game.phase, 'drawn': drawn}
        if isinstance(game, KaxinwuGame):
            view['winning_tiles'] = [tile_name(tile) for tile in game.won_tiles]
        return view

    def play_round(self):
        game = self.game
        while game.phase != 'over':
            if self.closed:
                raise Closed()
            actor, acts = game.legal_actions()
            old_wins = len(game.win_details)
            if acts == [PASS]:
                game.step(PASS)
                self.log_call()
                self.log_wins(game.win_details[old_wins:])
                continue
            preferred = suited_policy(policy_view(game, actor, acts))
            forced_draw_discard = (isinstance(game, KaxinwuGame) and game.win_counts[actor] and
                                   acts == [('discard', game.drawn_tile)])
            if actor == 0 and not forced_draw_discard:
                options = []
                for action in acts:
                    kind, tile = action
                    if kind in ('ron', 'peng', 'zhigang'):
                        tile = game.offer.tile
                    options.append({'action': action, 'kind': kind, 'label': kind,
                                    'tiles': [] if tile is None else [tile_name(tile)]})
                action = self.ask(self.view(), options, acts.index(preferred))
            else:
                self.publish(self.view())
                if self.pacing and not self.automatic:
                    time.sleep(self.pacing)
                action = preferred
            game.step(action)
            kind, tile = action
            if kind in ('discard', 'declare_ready'):
                self.log_event('discard', {'actor': actor, 'tile': tile_name(tile)})
            if kind == 'declare_ready':
                self.log_event('action', {'actor': actor, 'action': kind}, 'call')
            self.log_call()
            self.log_wins(game.win_details[old_wins:])

        view = self.view()
        delta = game.scores()
        self.totals = [total + points for total, points in zip(self.totals, delta)]
        self.next_dealer = game.next_dealer
        winners = list(dict.fromkeys(win.actor for win in game.win_details))
        if isinstance(game, KaxinwuGame):
            self.log_event('repeat_win_end', kind='round')
        elif not winners:
            self.log_event('draw', kind='round')
        with self.cv:
            self.result = {'reason_code': game.reason_code, 'scores': self.totals.copy(),
                           'deltas': delta, 'winners': winners,
                           'win_details': [asdict(win) for win in game.win_details], 'next': True}
            if isinstance(game, KaxinwuGame):
                for win in self.result['win_details']:
                    win['tile'] = tile_name(win['tile'])
                self.result['ready_deltas'] = game.ready_deltas.copy()
                self.result['kong_deltas'] = game.kong_deltas.copy()
            if game.rules.regional_payment:
                refunded = game.reason_code == 'draw' and game.rules.refund_kongs_on_draw
                self.result['kong_deltas'] = [0] * 4 if refunded else game.kong_deltas.copy()
            if game.wake_results:
                self.result['wake_tiles'] = [tile_name(tile) for tile in game.wake_indicators]
                self.result['wake_details'] = [{'actor': actor, 'count': count, 'hits': hits}
                                               for actor, (count, hits) in game.wake_results.items()]
            self.publish(view, 'round_end')

    def log_wins(self, wins):
        for win in wins:
            params = {'actor': win.actor, 'patterns': win.patterns}
            if isinstance(self.game, KaxinwuGame):
                self.log_event('flow_win', {**params, 'tile': tile_name(win.tile), 'fan': win.fan,
                                           'action': win.kind}, 'win')
            elif win.fan is None:
                self.log_event('win_points', {**params, 'points': win.points}, 'win')
            else:
                self.log_event('win_fan', {**params, 'fan': win.fan}, 'win')

    def log_call(self):
        if self.game.last_call is not None:
            actor, kind = self.game.last_call
            self.log_event('action', {'actor': actor, 'action': kind}, 'call')
