"""Public replacement flowers for games with ordinary 34-slot hands."""

from .honors import TILE_COUNT


FLOWER_TILES = {tile: f'F{tile - TILE_COUNT + 1}' for tile in range(TILE_COUNT, TILE_COUNT + 8)}


class FlowerWall:
    EXTRA_FLOWERS = FLOWER_TILES.keys()

    def _tile_supply(self):
        supply = super()._tile_supply()
        supply.update(self.EXTRA_FLOWERS)
        return supply

    def _prepare_wall(self):
        super()._prepare_wall()
        self.flowers = [[] for _ in self.players]
        self.draw_flowers = 0
        self.flower_kinds = frozenset(FLOWER_TILES)

    def physical_counts(self):
        counts = super().physical_counts()
        for flowers in self.flowers:
            counts.update(flowers)
        return counts

    def _deal_tile(self, actor):
        return self._take_tile(actor, False)

    def _reveal_flower(self, actor, tile):
        self.flowers[actor].append(tile)

    def _deal_flower_racks(self, waiting_tiles):
        order = tuple((self.dealer + i) % self.seat_count for i in range(self.seat_count))
        racks = [[] for _ in self.players]
        packets, tail = divmod(waiting_tiles, 4)
        for size in (4,) * packets + ((tail,) if tail else ()):
            for actor in order:
                racks[actor].extend(self.wall.pop() for _ in range(size))
        racks[self.dealer].append(self.wall.pop())
        while any(tile in self.flower_kinds for rack in racks for tile in rack):
            for actor in order:
                for index, tile in enumerate(racks[actor]):
                    if tile in self.flower_kinds:
                        self._reveal_flower(actor, tile)
                        racks[actor][index] = self.wall.pop(self._replacement_index())
        for actor, rack in enumerate(racks):
            for tile in rack:
                self.players[actor].hand[tile] += 1
        self.players[self.dealer].draws = 1
        self.drawn_tile = racks[self.dealer][-1]

    def _take_tile(self, actor, replacement):
        while (tile := super()._take_tile(actor, replacement)) is not None:
            if tile not in self.flower_kinds:
                return tile
            self._reveal_flower(actor, tile)
            self.draw_flowers += 1
            replacement = True
        return None

    def _draw(self, actor, kong_chain=0, provider=None):
        self.draw_flowers = 0
        super()._draw(actor, kong_chain, provider)
