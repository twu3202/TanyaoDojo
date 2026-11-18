"""Web table and shared tile-efficiency policy for suited discard-win rooms."""

from collections import Counter
from dataclasses import asdict
import time

from .core import Closed, GameSession, tile_name
from .errors import ClientError
from .sichuan_game import heuristic
from .suited import PASS, Profile, SuitedGame


def policy_view(game, actor, acts):
    visible = Counter(tile for river in game.discards for tile in river)
    for player in game.players:
        for kind, tile in player.melds:
            visible[tile] += 3 if kind == 'peng' else 4
    offer = game.offer
    if offer is not None and offer.kind == 'rob_kong':
        visible[offer.tile] += 1
    player = game.players[actor]
    return {
        'hand': player.hand.copy(), 'melds': player.melds.copy(), 'void': None,
        'allow_seven_pairs': game.rules.allow_seven_pairs,
        'actions': acts.copy(), 'public': dict(visible),
        'pending_tile': None if offer is None else offer.tile,
        'opponents': [{'void': None, 'finished': False, 'melds': len(other.melds)}
                      for seat, other in enumerate(game.players) if seat != actor],
    }


class SuitedSession(GameSession):
    def __init__(self, policy='heuristic', *, model=None, seed=None, pacing=.12, profile):
        self.profile = Profile(profile)
        super().__init__(self.profile.value, policy, seed=seed, pacing=pacing)
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
            self.game = SuitedGame(self.seed + self.round_no, self.next_dealer, profile=self.profile)
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
                'river': [{'tile': tile_name(tile)} for tile in game.discards[seat]],
                'melds': [{'kind': kind, 'tiles': [tile_name(tile)] * (3 if kind == 'peng' else 4)}
                          for kind, tile in player.melds],
            })
        actor, _ = game.legal_actions()
        if actor != 0 and game.offer is not None:
            actor = None
        drawn = tile_name(game.drawn_tile) if actor == 0 and game.drawn_tile is not None else None
        return {'players': players, 'hand_number': self.round_no, 'dealer': game.dealer,
                'current': actor, 'remaining': len(game.wall), 'dora': [], 'honba': 0,
                'pot': 0, 'phase': game.phase, 'drawn': drawn}

    def play_round(self):
        game = self.game
        while game.phase != 'over':
            if self.closed:
                raise Closed()
            actor, acts = game.legal_actions()
            if acts == [PASS]:
                game.step(PASS)
                self.log_call()
                continue
            preferred = heuristic(policy_view(game, actor, acts))
            if actor == 0:
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
            if kind == 'discard':
                self.log_event('discard', {'actor': actor, 'tile': tile_name(tile)})
            self.log_call()

        view = self.view()
        delta = game.scores()
        self.totals = [total + points for total, points in zip(self.totals, delta)]
        self.next_dealer = game.next_dealer
        if not game.winners:
            self.log_event('draw', kind='round')
        for win in game.win_details:
            if win.fan is None:
                self.log_event('win_points', {'actor': win.actor, 'points': win.points, 'patterns': win.patterns}, 'win')
            else:
                self.log_event('win_fan', {'actor': win.actor, 'fan': win.fan, 'patterns': win.patterns}, 'win')
        with self.cv:
            self.result = {'reason_code': game.reason_code, 'scores': self.totals.copy(),
                           'deltas': delta, 'winners': game.winners.copy(),
                           'win_details': [asdict(win) for win in game.win_details], 'next': True}
            if game.rules.regional_payment:
                refunded = game.reason_code == 'draw' and game.rules.refund_kongs_on_draw
                self.result['kong_deltas'] = [0] * 4 if refunded else game.kong_deltas.copy()
            if game.wake_results:
                self.result['wake_tiles'] = [tile_name(tile) for tile in game.wake_indicators]
                self.result['wake_details'] = [{'actor': actor, 'count': count, 'hits': hits}
                                               for actor, (count, hits) in game.wake_results.items()]
            self.publish(view, 'round_end')

    def log_call(self):
        if self.game.last_call is not None:
            actor, kind = self.game.last_call
            self.log_event('action', {'actor': actor, 'action': kind}, 'call')
