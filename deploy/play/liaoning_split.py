"""Liaoning split-draw rooms with distinct winning shapes and settlement."""

from collections import Counter
from dataclasses import dataclass, replace

from .hangzhou import split_wild
from .honors import DRAGONS, ORPHANS, PASS, TERMINALS, WINDS, ZIMO, HonorsGame, KongPoints, TableOptions, seven_pairs
from .northeast import NortheastRules
from .wildcard_shapes import can_form_pairs, can_form_standard, wildcard_decompositions


@dataclass(frozen=True)
class SplitScore:
    fan: int
    patterns: tuple[str, ...]
    tier = None

    @property
    def share(self):
        return 2 ** self.fan


def rack_faces(hand, melds):
    return {tile for tile, count in enumerate(hand) if count} | {tile for meld in melds for tile in meld.tiles}


def three_suits(faces):
    return len({tile // 9 for tile in faces if tile < 27}) == 3


class SplitDrawGame(HonorsGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        self.split_draw = False
        self.split_passes = []
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _draw(self, actor, kong_chain=0, provider=None):
        self.split_draw = 0 < self.remaining <= self.options.closed_live_tail
        super()._draw(actor, kong_chain, provider)

    def physical_counts(self):
        counts = super().physical_counts()
        counts.update(self.split_passes)
        return counts

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.split_draw and self.phase == 'action' and self.offer is None:
            return actor, [PASS, ZIMO] if ZIMO in actions else [PASS]
        return actor, actions

    def step(self, action):
        if self.split_draw and self.phase == 'action' and self.offer is None and action == PASS:
            self.players[self.cur].hand[self.drawn_tile] -= 1
            self.split_passes.append(self.drawn_tile)
            self.last_call = None
            self._draw((self.cur + 1) % self.seat_count)
            return
        super().step(action)


class JianchangRules(NortheastRules):
    WALL_RESERVE = 16
    OPEN_FOR_STANDARD = True
    OPTIONS = TableOptions(orphans=False, seven_pairs=True, closed_live_tail=4,
                           dealer_echo=True, kong_points=KongPoints(1, 1, 2, single_direct_payer=True),
                           concealed_kong_faces=0)

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return (dealer + 1) % 4 if winners and dealer not in winners else dealer

    @staticmethod
    def rack_allowed(hand, melds):
        faces = rack_faces(hand, melds)
        return bool(faces & ORPHANS) and three_suits(faces)

    @staticmethod
    def shape_allowed(pair, sets):
        return pair in DRAGONS or any(m.kind != 'chi' for m in sets)

    def faan(self, hand, melds, ctx):
        if ctx.kind == 'rob_concealed':
            return None
        hand = tuple(hand)
        if seven_pairs(hand, melds) and self.rack_allowed(hand, melds):
            labels = ['七小对']
            fan = 2
        elif self.closed(melds):
            return None
        else:
            candidates = []
            for pair, sets in self.shapes(hand, melds):
                full = (*sets, *melds)
                pung_only = all(m.kind != 'chi' for m in full)
                if pung_only:
                    candidates.append((2, '飘胡'))
                elif len(melds) < 4 and not any(m.kind != 'chi' and m.tile in DRAGONS for m in full):
                    eye = (ctx.tile == pair or any(m.kind == 'chi' and m.tile + 1 == ctx.tile for m in sets)
                           or (ctx.tile in TERMINALS and hand[ctx.tile] == 1
                               and not (rack_faces(hand, melds) - {ctx.tile}) & ORPHANS
                               and not any(ctx.tile in m.tiles for m in melds)))
                    candidates.append((1, '眼胡') if eye else (0, '屁胡'))
            if not candidates:
                return None
            fan, label = max(candidates)
            labels = [label]
        bonuses = (('自摸', ctx.kind == 'zimo'),
                   ('海捞', ctx.kind == 'zimo' and ctx.late_wall),
                   ('杠开', ctx.kind == 'zimo' and ctx.kong_chain > 0),
                   ('杠次', ctx.kind == 'ron' and ctx.kong_chain > 0),
                   ('抢杠胡', ctx.kind == 'rob_kong'))
        labels.extend(label for label, active in bonuses if active)
        return SplitScore(fan + len(labels) - 1, tuple(labels))

    @staticmethod
    def settle_hand(game, winners, offer):
        return [(win.score, tuple(
            win.score.share * 2 ** (int(game.dealer in (win.actor, seat)) + int(offer is not None and seat == offer.actor)
                                    + int(JIANCHANG.closed(game.players[seat].melds)))
            if seat != win.actor else 0 for seat in range(game.seat_count)
        )) for win in winners]


class JianchangGame(SplitDrawGame):
    def __init__(self, seed, dealer, round_wind, *, rules):
        self.draw_deltas = [0] * rules.OPTIONS.seat_count
        super().__init__(seed, dealer, round_wind, rules=rules)

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, replace(ctx, late_wall=self.split_draw))

    def _kong_factor(self, meld):
        return 4 if meld.tile in DRAGONS else 1

    def _end(self, reason):
        if reason == 'draw':
            for seat, player in enumerate(self.players):
                delta = 1 - self.seat_count if seat == self.dealer else 1
                self.draw_deltas[seat] = delta
                player.score += delta
        super()._end(reason)


class LiaoyangRules(NortheastRules):
    WALL_RESERVE = 17
    OPTIONS = TableOptions(orphans=False, seven_pairs=True, closed_live_tail=4, concealed_kong_faces=0)
    ASSIGNABLE = tuple(tile for tile in OPTIONS.tile_kinds if tile not in TERMINALS)
    CAP = 64
    require_burst = False

    next_dealer = staticmethod(JianchangRules.next_dealer)

    def faan(self, hand, melds, ctx, *, wildcard):
        if ctx.kind == 'rob_concealed' or (ctx.kind != 'zimo' and ctx.tile == wildcard):
            return None
        natural, wild = split_wild(hand, wildcard)
        faces = rack_faces(natural, melds)
        if not faces & ORPHANS:
            return None
        candidates = []
        if not melds and sum(hand) == 14 and can_form_pairs(natural, wild):
            candidates.append((1, ('七小对',)))
        if can_form_standard(natural, wild):
            suit_gate = three_suits(faces)
            for _, _, shapes in wildcard_decompositions(natural, wild, len(melds), self.ASSIGNABLE):
                for pair, sets in shapes:
                    full = (*melds, *sets)
                    pungs = all(m.kind != 'chi' for m in full)
                    needed_pung = pair in DRAGONS or any(m.kind != 'chi' for m in full)
                    if not needed_pung or not (pungs or suit_gate and len(melds) < 4):
                        continue
                    labels = ['飘胡' if pungs else '平胡']
                    fan = int(pungs)
                    if pair < 27 and pair % 9 in (1, 7):
                        labels.append('二八将')
                        fan += 1
                    concealed = [m for m in (*sets, *melds) if m.kind in ('triplet', 'gang_an')]
                    if ctx.kind != 'zimo' and pair != ctx.tile and not any(
                            m.kind == 'chi' and ctx.tile in m.tiles for m in sets):
                        concealed = [m for m in concealed if m.kind == 'gang_an' or m.tile != ctx.tile]
                    if len(concealed) >= 3:
                        labels.append('四叉' if len(concealed) == 4 else '三叉')
                        fan += len(concealed) - 2
                    if any(m.tile in DRAGONS for m in concealed):
                        labels.append('大哥大')
                        fan += 1
                    if len(melds) == 4:
                        labels.append('手把一')
                        fan += 1
                    candidates.append((fan, tuple(labels)))
        if not candidates:
            return None
        if ctx.opening == '天胡' or ctx.first_nondealer_draw:
            return SplitScore(6, ('天胡' if ctx.opening == '天胡' else '地胡',))
        fan, labels = max(candidates)
        counts = Counter({tile: count for tile, count in enumerate(natural) if count})
        counts.update(tile for meld in melds if meld.kind in ('chi', 'peng') for tile in meld.tiles)
        four_in_one = sum(count == 4 for count in counts.values())
        bonuses = (('素胡', not wild), ('自摸', ctx.kind == 'zimo'),
                   ('闷胡', ctx.kind == 'zimo' and self.closed(melds)),
                   ('海捞', ctx.kind == 'zimo' and ctx.late_wall),
                   ('杠开', ctx.kind == 'zimo' and ctx.kong_chain > 0),
                   ('杠炮', ctx.kind == 'ron' and ctx.kong_chain > 0), ('抢杠胡', ctx.kind == 'rob_kong'))
        active = tuple(label for label, applies in bonuses if applies)
        return SplitScore(fan + len(active) + four_in_one, (*labels, *active, *('四归一',) * four_in_one))

    def settle_hand(self, game, winners, offer):
        win, = winners
        game.treasure_tile = game.wall.pop()
        matches = win.hand[game.treasure_tile] + sum(
            meld.tiles.count(game.treasure_tile) for meld in game.players[win.actor].melds
        )
        game.treasure_points = 4 if matches >= (2 if game.treasure_tile == game.wildcard_tile else 3) else matches
        provider = offer.actor if offer is not None else None
        payments = tuple(
            min(self.CAP, win.score.share * 2 ** (int(game.dealer in (win.actor, seat))
                + int(seat == provider) + int(self.closed(game.players[seat].melds)))) + game.treasure_points
            if seat != win.actor else 0 for seat in range(game.seat_count)
        )
        return [(win.score, payments)]


class LiaoyangGame(SplitDrawGame):
    def _prepare_wall(self):
        # This room uses a copy of the revealed face as its joker.
        self.indicator_tile = self.wildcard_indicator = self.wildcard_tile = self.wall.pop(0)
        self.treasure_tile = None
        self.treasure_points = 0

    @property
    def remaining(self):
        return super().remaining + int(self.treasure_tile is not None)

    def physical_counts(self):
        counts = super().physical_counts()
        counts[self.indicator_tile] += 1
        if self.treasure_tile is not None:
            counts[self.treasure_tile] += 1
        return counts

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.offer is None and self.phase == 'action':
            player = self.players[actor]
            actions = [(kind, tile) for kind, tile in actions
                       if not (tile == self.wildcard_tile and kind in ('discard', 'ankan'))
                       and (kind != 'ankan' or self._last_pung_allowed(player.melds, tile))]
        return actor, actions

    @staticmethod
    def _last_pung_allowed(melds, tile):
        # After four pungs, two jokers must always be usable as eyes.
        return len(melds) != 3 or (
            all(meld.kind != 'chi' for meld in melds)
            and (tile in ORPHANS or any(meld.tile in ORPHANS for meld in melds))
        )

    def _response_actions(self):
        actor, actions = super()._response_actions()
        if self.offer.kind != 'discard':
            return actor, actions
        player = self.players[actor]
        tile = self.offer.tile
        if not self._last_pung_allowed(player.melds, tile):
            actions = [a for a in actions if a[0] not in ('peng', 'zhigang')]
        if sum(player.hand) - player.hand[self.wildcard_tile] == 2:
            actions = [a for a in actions if a[0] != 'peng']
        return actor, actions

    def _chows(self, player, tile, provider):
        if len(player.melds) == 3:
            return ()
        return ((start, used, block) for start, used, block in super()._chows(player, tile, provider)
                if self.wildcard_tile not in used and any(
                    count > used.count(t) for t, count in enumerate(player.hand)
                    if t != self.wildcard_tile and t not in block))

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, replace(ctx, late_wall=self.split_draw), wildcard=self.wildcard_tile)


JIANCHANG = JianchangRules('jianchang-final-split')
LIAOYANG = LiaoyangRules('liaoyang-treasure-split')
LIAONING_SPLIT_ROOMS = (JIANCHANG, LIAOYANG)
