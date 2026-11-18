"""Public replacement flowers for games with ordinary 34-slot hands."""

from .honors import TILE_COUNT


FLOWER_TILES = {tile: f'F{tile - TILE_COUNT + 1}' for tile in range(TILE_COUNT, TILE_COUNT + 8)}


class FlowerWall:
    def _tile_supply(self):
        supply = super()._tile_supply()
        supply.update(FLOWER_TILES.keys())
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

    def _take_tile(self, actor, replacement):
        while (tile := super()._take_tile(actor, replacement)) is not None:
            if tile not in self.flower_kinds:
                return tile
            self.flowers[actor].append(tile)
            self.draw_flowers += 1
            replacement = True
        return None

    def _draw(self, actor, kong_chain=0, provider=None):
        self.draw_flowers = 0
        super()._draw(actor, kong_chain, provider)
