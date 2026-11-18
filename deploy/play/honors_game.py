"""Browser session adapter for rooms using suited tiles and honors."""

from dataclasses import asdict
import random
import time

from . import golden_guangdong, guangzhou, guiliu, hongkong, jj_multiplier, jj_three, macau, tiantai
from .anji import ANJI, ANJI_TEA_HOUSE, AnjiGame, AnjiTeaHouseGame
from .chongji import CHONGJI, CHONGJI_SEVEN, ChongjiGame
from .daxi import DAXI, DaxiGame
from .dongyang import DONGYANG, DongyangGame
from .honors_bonus import HEZHOU, HUNDRED, LIUZHOU, SHANGSI
from .core import Closed, GameSession, tile_name
from .errors import ClientError
from .flower_wall import FLOWER_TILES, FlowerWall
from .hangzhou import FLOAT, HANGZHOU_ROOMS, WHITE, HangzhouGame
from .honors import DEFAULT_OPTIONS, PASS, HonorsGame
from .honors_policy import heuristic, policy_view
from .jinyun import JINYUN, JINYUN_CANTONESE, JinyunCantoneseGame, JinyunClassicGame
from .jiaxing import JIAXING, JiaxingGame
from .kawuxing import SHIYAN, SUIZHOU, XIANGYANG, KaWuXingGame
from .linhai import LINHAI, LinhaiGame
from .pinghu import PINGHU, PinghuGame
from .quzhou import QUZHOU_BANKED, QUZHOU_DUANZI, QUZHOU_LATA, QuzhouGame
from .taizhou import TAIZHOU, TaizhouGame
from .xiaogan import XIAOGAN, XiaoganGame
from .zhejiang_flipped import (DEQING, DEQING_XINSHI, ELEVEN_WINDS, SONGYANG, SUICHANG, TEN_WINDS, XIAOSHAN, DeqingGame,
                               DeqingXinshiGame, SongyangGame, SuichangGame, XiaoshanGame)


RULESETS = {rule.VARIANT: rule for rule in (
    hongkong, macau, guangzhou, SHANGSI, HUNDRED, HEZHOU, LIUZHOU, golden_guangdong, guiliu,
    SUIZHOU, XIANGYANG, SHIYAN, jj_three, jj_multiplier, XIAOGAN, *HANGZHOU_ROOMS, XIAOSHAN, SONGYANG, SUICHANG,
    DEQING, DEQING_XINSHI, ANJI, ANJI_TEA_HOUSE, DONGYANG, CHONGJI, CHONGJI_SEVEN, tiantai, tiantai.TIANTAI_WILD,
    TAIZHOU, DAXI, LINHAI, JINYUN, JINYUN_CANTONESE, JIAXING, PINGHU, QUZHOU_DUANZI, QUZHOU_LATA, QUZHOU_BANKED,
)}
GAME_ENGINES = {
    **{rules.VARIANT: KaWuXingGame for rules in (SUIZHOU, XIANGYANG, SHIYAN)},
    jj_three.VARIANT: jj_three.JJThreeGame,
    jj_multiplier.VARIANT: jj_multiplier.JJMultiplierGame,
    XIAOGAN.VARIANT: XiaoganGame,
    **{rules.VARIANT: HangzhouGame for rules in HANGZHOU_ROOMS},
    XIAOSHAN.VARIANT: XiaoshanGame,
    SONGYANG.VARIANT: SongyangGame,
    SUICHANG.VARIANT: SuichangGame,
    DEQING.VARIANT: DeqingGame,
    DEQING_XINSHI.VARIANT: DeqingXinshiGame,
    ANJI.VARIANT: AnjiGame,
    ANJI_TEA_HOUSE.VARIANT: AnjiTeaHouseGame,
    DONGYANG.VARIANT: DongyangGame,
    CHONGJI.VARIANT: ChongjiGame,
    CHONGJI_SEVEN.VARIANT: ChongjiGame,
    tiantai.VARIANT: tiantai.TiantaiGame,
    tiantai.TIANTAI_WILD.VARIANT: tiantai.TiantaiWildcardGame,
    TAIZHOU.VARIANT: TaizhouGame,
    DAXI.VARIANT: DaxiGame,
    LINHAI.VARIANT: LinhaiGame,
    JINYUN.VARIANT: JinyunClassicGame,
    JINYUN_CANTONESE.VARIANT: JinyunCantoneseGame,
    JIAXING.VARIANT: JiaxingGame,
    PINGHU.VARIANT: PinghuGame,
    **{rules.VARIANT: QuzhouGame for rules in (QUZHOU_BANKED, QUZHOU_DUANZI, QUZHOU_LATA)},
}


class HonorsSession(GameSession):
    def __init__(self, policy='heuristic', *, model=None, seed=None, pacing=.12, variant=hongkong.VARIANT):
        super().__init__(variant, policy, seed=seed, pacing=pacing)
        self.rules = RULESETS[variant]
        self.hand_number = 0
        options = getattr(self.rules, 'OPTIONS', DEFAULT_OPTIONS)
        self.seat_count = options.seat_count
        self.totals = [QUZHOU_BANKED.BANK if variant == QUZHOU_BANKED.VARIANT else 0] * self.seat_count
        self.dealer = 0 if options.host_deals_first else random.Random(self.seed).randrange(self.seat_count)
        self.dealer_advances = 0
        self.dealer_streak = 0
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
            engine = GAME_ENGINES.get(self.variant, HonorsGame)
            dealer_rules = {'dealer_streak': self.dealer_streak} if issubclass(engine, HangzhouGame) else {}
            if self.rules is QUZHOU_BANKED:
                dealer_rules['bank'] = self.totals
            previous = self.game
            self.game = engine(self.seed + self.hand_number, self.dealer,
                               27 + (self.dealer_advances // self.seat_count) % 4, rules=self.rules, **dealer_rules)
            if isinstance(self.game, JiaxingGame):
                self.game.carry_yellow(previous)
            self.result = None
            self.options = []
            self.automatic = False
            self.status = 'starting'
            self.version += 1
            self.log_event('hand_start', kind='round')
            self.start_worker(self.play_round)

    def view(self):
        game = self.game
        before_play = game.phase in ('piao', 'buy_base')
        players = []
        for actor, player in enumerate(game.players):
            won = actor in game.revealed
            hand = game.revealed.get(actor, player.hand)
            ready = player.open_ready_hand is not None
            reveal = hand if actor == 0 or won else player.open_ready_hand
            if before_play:
                reveal = None
            players.append({
                'id': actor, 'score': self.totals[actor] + player.score, 'delta': player.score,
                'hand_count': 0 if before_play else sum(hand),
                'revealed': reveal is not None, 'won': won, 'riichi': False,
                'hand': [self.display_tile(tile) for tile, count in enumerate(reveal) for _ in range(count)]
                        if reveal is not None else [],
                'river': [{'tile': self.display_tile(tile)} for tile in game.discards[actor]],
                'melds': [self.view_meld(meld, actor == 0) for meld in player.melds],
            })
            if ready:
                players[-1]['open_ready'] = True
            if player.closed_ready:
                players[-1]['ready'] = True
            if isinstance(game, jj_multiplier.JJMultiplierGame) and actor in game.dark_seats:
                players[-1]['dark_ready'] = True
                if not won and actor != 0:
                    players[-1]['hidden_ready_tiles'] = game.hidden_ready_tiles(actor)
            if isinstance(game, XiaoganGame) and len(game.piao) == game.seat_count:
                players[-1]['piao'] = game.piao[actor]
            if isinstance(game, DaxiGame) and not before_play:
                players[-1]['bought_base'] = game.buy_base[actor]
            if isinstance(game, JiaxingGame) and actor in game.folded:
                players[-1]['folded'] = True
            if isinstance(game, HangzhouGame) and game.float_streak[actor]:
                players[-1]['wildcard_floats'] = game.float_streak[actor]
            if isinstance(game, XiaoshanGame) and actor == 0 and game.ten_winds[actor] >= 5:
                players[-1]['ten_winds_run'] = game.ten_winds[actor]
            if isinstance(game, FlowerWall):
                players[-1]['flowers'] = [self.display_tile(tile) for tile in game.flowers[actor]]
        current = game.cur if game.phase != 'over' else None
        if game.offer is not None:
            current = 0 if game.responses[0] == 0 else None
        if game.phase == 'liability_choice':
            current = game.liability_choices[0]
        drawn = self.display_tile(game.drawn_tile) if current == 0 and game.drawn_tile is not None else None
        if game.phase == 'liability_choice':
            drawn = None
        table = {'players': players, 'hand_number': self.hand_number, 'dealer': None if game.phase == 'piao' else game.dealer,
                'round_wind': tile_name(game.round_wind), 'show_winds': game.options.show_winds,
                'match_hands': self.rules.MATCH_HANDS,
                'current': current, 'remaining': game.remaining, 'dora': [], 'honba': 0,
                'pot': 0, 'phase': game.phase, 'drawn': drawn}
        if isinstance(game, HangzhouGame):
            table.update(wildcard=self.display_tile(game.wildcard_tile), dealer_multiplier=game.dealer_multiplier,
                         float_cycle_owner=game.float_cycle_owner, feed_liabilities=game.related_seats.copy())
            if game.wildcard_indicator is not None:
                table['wildcard_indicated'] = True
                if game.indicator_tile != game.wildcard_tile:
                    table['wildcard_indicator'] = self.display_tile(game.wildcard_indicator)
                elif game.wildcard_indicator != game.wildcard_tile:
                    table['white_substitute'] = tile_name(game.wildcard_indicator)
                if isinstance(game, DongyangGame) and game.wildcard_tile != DONGYANG.WILDCARD:
                    table['white_optional'] = self.display_tile(game.wildcard_tile)
                if isinstance(game, QuzhouGame) and game.wildcard_tile != WHITE:
                    table['white_optional'] = self.display_tile(game.wildcard_tile)
        if isinstance(game, ChongjiGame):
            table['dealer_stake'] = game.dealer_stake
        if isinstance(game, tiantai.TiantaiWildcardGame):
            table.update(wildcard=tile_name(self.rules.WILDCARD), wildcard_fixed=True)
        if isinstance(game, LinhaiGame):
            table['wildcard_fixed'] = True
        if isinstance(game, TaizhouGame):
            table['fresh_stage'] = game.fresh_stage
        if isinstance(game, JiaxingGame):
            table['yellow_faan'] = len(game.yellow)
        return table

    def display_tile(self, tile):
        if tile in FLOWER_TILES:
            return FLOWER_TILES[tile]
        if isinstance(self.game, HangzhouGame):
            tile = self.game.physical_tile(tile)
        return tile_name(tile)

    def view_meld(self, meld, own):
        tiles = [self.display_tile(tile) for tile in meld.tiles]
        if meld.kind == 'gang_an':
            visible = self.game.options.concealed_kong_faces
            if visible or not own:
                tiles = tiles[:visible] + [None] * (len(tiles) - visible)
        return {'kind': meld.kind, 'tiles': tiles, 'provider': meld.provider}

    def option(self, action):
        kind, tile = action
        if kind == 'exchange':
            tiles = [self.display_tile(index) for index in tile]
        elif kind in ('piao', 'buy_base', 'liability_choice'):
            return {'action': action, 'kind': kind, 'label': kind, 'tiles': [], 'value': tile}
        elif kind == 'chi':
            tiles = [self.display_tile(index) for index in range(tile, tile + 3)]
        elif kind in ('ron', 'peng', 'zhigang'):
            tiles = [self.display_tile(self.game.offer.tile)]
        else:
            tiles = [] if tile is None else [self.display_tile(tile)]
        option = {'action': action, 'kind': kind, 'label': kind, 'tiles': tiles}
        if kind == 'ron' and isinstance(self.game, HangzhouGame):
            option['win_type'] = 'rob_kong' if self.game.offer.kind == 'gang_bu' else 'fangchong_win'
        return option

    def play_round(self):
        game = self.game
        while game.phase != 'over':
            if self.closed:
                raise Closed()
            actor, actions = game.legal_actions()
            if actions == [PASS] or (isinstance(game, JiaxingGame) and actor in game.folded):
                action = actions[0]
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
            if kind in ('discard', 'declare_ready', 'declare_dark', TEN_WINDS, ELEVEN_WINDS):
                self.log_event('discard', {'actor': actor, 'tile': self.display_tile(tile)})
            elif kind == FLOAT[0]:
                self.log_event('wildcard_float', {'actor': actor, 'count': game.float_streak[actor]}, 'call')
            if kind == 'declare_ready':
                code = 'paid_ready' if game.players[actor].closed_ready else 'open_ready'
                self.log_event(code, {'actor': actor}, 'call')
            elif kind == 'declare_dark':
                self.log_event('dark_ready', {'actor': actor}, 'call')
            elif kind == 'piao' and game.phase != 'piao':
                self.log_event('piao_complete', kind='round')
            elif kind == 'buy_base' and game.phase != 'buy_base':
                self.log_event('buy_base_complete', kind='round')
            if game.last_call:
                caller, meld = game.last_call
                self.log_event('action', {'actor': caller, 'action': meld.kind}, 'call')

        table = self.view()
        delta = game.scores()
        self.totals = [total + points for total, points in zip(self.totals, delta)]
        self.dealer_streak = self.dealer_streak + 1 if game.next_dealer == self.dealer else 0
        self.dealer_advances += game.next_dealer != self.dealer
        self.dealer = game.next_dealer
        if not game.winners:
            self.log_event('draw', kind='round')
        for win in game.win_details:
            if win.tier:
                self.log_event('win_tier', {'actor': win.actor, 'tier': win.tier, 'patterns': win.patterns}, 'win')
            elif self.variant == jj_multiplier.VARIANT or isinstance(game, HangzhouGame):
                self.log_event('win_total', {'actor': win.actor, 'points': sum(win.payments),
                                             'patterns': win.patterns}, 'win')
            elif win.fan is None:
                self.log_event('win_points', {'actor': win.actor, 'points': win.points, 'patterns': win.patterns}, 'win')
            else:
                self.log_event('win_fan', {'actor': win.actor, 'fan': win.fan, 'patterns': win.patterns}, 'win')
        match_hands = self.rules.MATCH_HANDS
        busted = isinstance(game, QuzhouGame) and game.busted
        finished = self.hand_number == match_hands or bool(busted)
        with self.cv:
            self.result = {'reason_code': game.reason_code, 'scores': self.totals.copy(),
                           'deltas': delta, 'winners': game.winners.copy(),
                           'win_details': [asdict(win) for win in game.win_details], 'next': not finished}
            if isinstance(game, TaizhouGame) and game.taizhou_settlement is not None:
                self.result['taizhou_settlement'] = game.taizhou_settlement
            if isinstance(game, DaxiGame):
                self.result['buy_base_deltas'] = game.buy_base_deltas.copy()
            if isinstance(game, JinyunClassicGame):
                self.result['follow_deltas'] = game.follow_deltas.copy()
            if isinstance(game, JinyunCantoneseGame) and any(game.kong_refunds):
                self.result['kong_refund_deltas'] = game.kong_refunds.copy()
            if game.options.kong_points:
                self.result['kong_deltas'] = game.kong_deltas.copy()
            if any(game.echo_deltas):
                self.result['echo_deltas'] = game.echo_deltas.copy()
            if self.variant in (SUIZHOU.VARIANT, XIANGYANG.VARIANT, jj_multiplier.VARIANT):
                self.result['open_ready_deltas'] = game.ready_deltas.copy()
            elif self.variant == jj_three.VARIANT:
                self.result.update({key: values.copy() for key, values in game.side_deltas.items()})
            elif self.variant == XIAOGAN.VARIANT:
                self.result['piao_deltas'] = game.piao_deltas.copy()
                self.result['horse_deltas'] = game.horse_deltas.copy()
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
                self.result['match_hands'] = self.hand_number
                if busted:
                    self.result['busted'] = game.busted.copy()
                self.result['match_winners'] = [seat for seat, points in enumerate(self.totals) if points == best]
            self.publish(table, 'finished' if finished else 'round_end')
