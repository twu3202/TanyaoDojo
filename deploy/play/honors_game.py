"""Browser session adapter for rooms using suited tiles and honors."""

from dataclasses import asdict
import random
import time

from . import golden_guangdong, guangzhou, guiliu, hongkong, macau
from .honors_bonus import HEZHOU, HUNDRED, LIUZHOU, SHANGSI
from .core import Closed, GameSession, tile_name
from .errors import ClientError
from .honors import DEFAULT_OPTIONS, PASS, HonorsGame
from .honors_policy import heuristic, policy_view


RULESETS = {rule.VARIANT: rule for rule in (
    hongkong, macau, guangzhou, SHANGSI, HUNDRED, HEZHOU, LIUZHOU, golden_guangdong, guiliu,
)}


class HonorsSession(GameSession):
    def __init__(self, policy='heuristic', *, model=None, seed=None, pacing=.12, variant=hongkong.VARIANT):
        super().__init__(variant, policy, seed=seed, pacing=pacing)
        self.rules = RULESETS[variant]
        self.hand_number = 0
        self.totals = [0] * 4
        options = getattr(self.rules, 'OPTIONS', DEFAULT_OPTIONS)
        self.dealer = 0 if options.host_deals_first else random.Random(self.seed).randrange(4)
        self.dealer_advances = 0
        self.game = None

    def begin(self):
        if self.thread is not None and self.thread.is_alive() and self.status == 'round_end':
            self.thread.join(timeout=1)
        with self.cv:
            if self.status == 'starting' and self.thread is not None:
                raise ClientError('round_starting')
            if self.status not in ('starting', 'round_end'):
                raise ClientError('round_active')
            self.hand_number += 1
            self.game = HonorsGame(self.seed + self.hand_number, self.dealer,
                                   27 + (self.dealer_advances // 4) % 4, rules=self.rules)
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
        for actor, player in enumerate(game.players):
            won = actor in game.revealed
            hand = game.revealed.get(actor, player.hand)
            revealed = actor == 0 or won
            players.append({
                'id': actor, 'score': self.totals[actor] + player.score, 'delta': player.score,
                'hand_count': sum(hand), 'revealed': revealed, 'won': won, 'riichi': False,
                'hand': [tile_name(tile) for tile, count in enumerate(hand) for _ in range(count)] if revealed else [],
                'river': [{'tile': tile_name(tile)} for tile in game.discards[actor]],
                'melds': [self.view_meld(meld) for meld in player.melds],
            })
        current = game.cur if game.phase != 'over' else None
        if game.offer is not None:
            current = 0 if game.responses[0] == 0 else None
        drawn = tile_name(game.drawn_tile) if current == 0 and game.drawn_tile is not None else None
        return {'players': players, 'hand_number': self.hand_number, 'dealer': game.dealer,
                'round_wind': tile_name(game.round_wind), 'show_winds': game.options.show_winds,
                'match_hands': self.rules.MATCH_HANDS,
                'current': current, 'remaining': game.remaining, 'dora': [], 'honba': 0,
                'pot': 0, 'phase': game.phase, 'drawn': drawn}

    def view_meld(self, meld):
        tiles = [tile_name(tile) for tile in meld.tiles]
        if meld.kind == 'gang_an':
            visible = self.game.options.concealed_kong_faces
            tiles = tiles[:visible] + [None] * (len(tiles) - visible)
        return {'kind': meld.kind, 'tiles': tiles, 'provider': meld.provider}

    def option(self, action):
        kind, tile = action
        if kind == 'exchange':
            tiles = [tile_name(index) for index in tile]
        elif kind == 'chi':
            tiles = [tile_name(index) for index in range(tile, tile + 3)]
        elif kind in ('ron', 'peng', 'zhigang'):
            tiles = [tile_name(self.game.offer.tile)]
        else:
            tiles = [] if tile is None else [tile_name(tile)]
        return {'action': action, 'kind': kind, 'label': kind, 'tiles': tiles}

    def play_round(self):
        game = self.game
        while game.phase != 'over':
            if self.closed:
                raise Closed()
            actor, actions = game.legal_actions()
            if actions == [PASS]:
                action = PASS
            else:
                action = heuristic(policy_view(game, actor, actions))
                if actor == 0:
                    action = self.ask(self.view(), [self.option(a) for a in actions], actions.index(action))
                else:
                    self.publish(self.view())
                    if self.pacing and not self.automatic:
                        time.sleep(self.pacing)
            game.step(action)
            kind, tile = action
            if kind == 'discard':
                self.log_event('discard', {'actor': actor, 'tile': tile_name(tile)})
            if game.last_call:
                caller, meld = game.last_call
                self.log_event('action', {'actor': caller, 'action': meld.kind}, 'call')

        table = self.view()
        delta = game.scores()
        self.totals = [total + points for total, points in zip(self.totals, delta)]
        self.dealer_advances += game.next_dealer != self.dealer
        self.dealer = game.next_dealer
        if not game.winners:
            self.log_event('draw', kind='round')
        for win in game.win_details:
            if win.tier:
                self.log_event('win_tier', {'actor': win.actor, 'tier': win.tier, 'patterns': win.patterns}, 'win')
            elif win.fan is None:
                self.log_event('win_points', {'actor': win.actor, 'points': win.points, 'patterns': win.patterns}, 'win')
            else:
                self.log_event('win_fan', {'actor': win.actor, 'fan': win.fan, 'patterns': win.patterns}, 'win')
        match_hands = self.rules.MATCH_HANDS
        finished = self.hand_number == match_hands
        with self.cv:
            self.result = {'reason_code': game.reason_code, 'scores': self.totals.copy(),
                           'deltas': delta, 'winners': game.winners.copy(),
                           'win_details': [asdict(win) for win in game.win_details], 'next': not finished}
            if game.options.kong_points:
                self.result['kong_deltas'] = game.kong_deltas.copy()
            if any(game.echo_deltas):
                self.result['echo_deltas'] = game.echo_deltas.copy()
            if game.horse_tiles:
                self.result['bonus'] = [tile_name(tile) for tile in game.horse_tiles]
                mode = getattr(self.rules, 'BONUS_MODE', None)
                if mode:
                    self.result['bonus_mode'] = mode
                self.result['bonus_details'] = [
                    {'actor': actor, 'count': count, 'hits': hits, 'per_loser': points, 'extra': extra}
                    for actor, count, hits, points, extra in game.horse_details
                ]
                if game.bilateral_horses:
                    self.result['horse_breakdown'] = game.bilateral_horses.copy()
            if game.wake_tiles:
                self.result['wake_tiles'] = [tile_name(tile) for tile in game.wake_tiles]
                self.result['wake_details'] = game.wake_details.copy()
            if finished:
                best = max(self.totals)
                self.result['match_hands'] = match_hands
                self.result['match_winners'] = [seat for seat, points in enumerate(self.totals) if points == best]
            self.publish(table, 'finished' if finished else 'round_end')
