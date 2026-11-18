"""Bengbu no-flower and East Wind Flower tables."""

from dataclasses import dataclass

from .flower_wall import FlowerWall
from .honors import DRAGONS, PASS, HonorsGame, TableOptions, standard_shapes
from .honors_bonus import Score


FLOWER_FACES = DRAGONS | {27}


@dataclass(frozen=True)
class BengbuRules:
    VARIANT: str
    flowers: bool = False

    OPTIONS = TableOptions(chow=False, orphans=False, concealed_kong_faces=1, host_deals_first=True,
                           pass_win_all=True, pass_pon_all=True, pass_win_clear_by_call=True,
                           kong_requires_live_tile=True, show_winds=False)
    MATCH_HANDS = 8
    MIN_FAAN = 0
    PASS_LOCK = False
    SWAP_LOCK = False

    @property
    def WALL_RESERVE(self):
        return 0 if self.flowers else 14

    def faan(self, hand, melds, ctx, *, flowers=0):
        if ctx.kind == 'rob_concealed' or not any(standard_shapes(hand, len(melds))):
            return None
        if self.flowers and any(hand[tile] for tile in FLOWER_FACES):
            return None
        self_draw = ctx.kind == 'zimo' or (ctx.kind == 'rob_kong' and not self.flowers)
        points = 2 if self_draw else 1
        patterns = []
        concealed = sum(meld.kind == 'gang_an' for meld in melds)
        exposed = sum(meld.kind in ('gang_ming', 'gang_bu') for meld in melds)
        if exposed:
            points += exposed * (2 if self.flowers and self_draw else 1)
            patterns.append('明杠')
        if concealed:
            points += concealed * 2
            patterns.append('暗杠')
        if flowers:
            points += flowers
            patterns.append(f'花牌x{flowers}')
        if ctx.kind == 'zimo' and ctx.kong_chain and not self.flowers:
            points *= 2
            patterns.append('杠上开花')
        return Score(points, tuple(patterns) or ('平胡',))

    @staticmethod
    def liability(melds, last):
        return ()

    @staticmethod
    def next_dealer(dealer, winners, provider):
        return (dealer + 1) % 4 if winners and dealer not in winners else dealer

    def settle_hand(self, game, winners, offer):
        winner, = winners
        actor, score = winner.actor, winner.score
        shares = tuple(score.share + (game.dealer in (actor, seat)) if seat != actor else 0
                       for seat in range(game.seat_count))
        if offer is None:
            return [(score, shares)]
        amount = sum(shares) if offer.kind == 'gang_bu' and not self.flowers else shares[offer.actor]
        return [(score, tuple(amount if seat == offer.actor else 0 for seat in range(game.seat_count)))]


BENGBU = BengbuRules('bengbu-classic')
BENGBU_FLOWER = BengbuRules('bengbu-east-flower', flowers=True)


class BengbuGame(HonorsGame):
    def _replacement_index(self):
        return self.rules.WALL_RESERVE

    def _response_actions(self):
        if self.winners:
            return self.responses[0], [PASS]
        return super()._response_actions()


class BengbuFlowerGame(FlowerWall, BengbuGame):
    EXTRA_FLOWERS = ()
    excluded_tiles = FLOWER_FACES

    def _prepare_wall(self):
        super()._prepare_wall()
        self.flower_kinds = self.excluded_tiles
        self.flower_started = set()

    def _deal_tile(self, actor):
        return self.wall.pop()

    def _take_tile(self, actor, replacement):
        if actor in self.flower_started:
            return super()._take_tile(actor, replacement)
        if replacement:
            # Keep even a newly drawn flower private until this seat's first ordinary draw.
            return HonorsGame._take_tile(self, actor, replacement)
        tile = super()._take_tile(actor, False)
        if tile is None:
            return None
        self.flower_started.add(actor)
        hand = self.players[actor].hand
        for face in sorted(self.flower_kinds):
            while hand[face]:
                hand[face] -= 1
                self.flowers[actor].append(face)
                self.draw_flowers += 1
                substitute = super()._take_tile(actor, True)
                if substitute is None:
                    hand[tile] += 1
                    return None
                hand[substitute] += 1
        return tile

    def legal_actions(self):
        actor, actions = super().legal_actions()
        if self.phase == 'action':
            actions = [action for action in actions if action[1] not in self.flower_kinds]
        return actor, actions

    def _score(self, actor, hand, melds, ctx):
        return self.rules.faan(hand, melds, ctx, flowers=len(self.flowers[actor]))
