"""Point schedules, payers and wake tiles for regional 108-tile rooms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WinContext:
    actor: int
    provider: int | None
    dealer: int
    kind: str
    opening: str | None
    after_kong: bool
    kong_provider: int | None
    tiles_left: int
    winner_count: int
    wake_hits: int = 0


def seven_pairs(hand, melds):
    return not melds and sum(hand) == 14 and all(count % 2 == 0 for count in hand)


def all_triplets(hand, melds):
    return sorted(count for count in hand if count) == [2] + [3] * (4 - len(melds))


def _shares(actor, provider, points, *, covered=False):
    if provider is not None:
        return tuple(points * (3 if covered else 1) if seat == provider else 0 for seat in range(4))
    return tuple(0 if seat == actor else points for seat in range(4))


def shaanxi_payment(hand, melds, ctx):
    base = 2 if ctx.kind == 'zimo' else 1
    shares = _shares(ctx.actor, ctx.provider, base)
    payments = tuple(points * (2 if ctx.dealer in (seat, ctx.actor) else 1)
                     for seat, points in enumerate(shares))
    pattern = {'zimo': '自摸', 'ron': '平胡', 'rob_kong': '抢杠'}[ctx.kind]
    return (pattern,), base, payments


# Every row gives the self-draw and ordinary discard payment per share.
LISHUI_POINTS = {'平胡': (2, 1), '对对胡': (4, 6), '豪华大对': (12, 18), '清一色': (8, 12),
                 '七对': (6, 9), '清七对': (12, 18), '天胡': (16, 0), '地胡': (0, 20), '单钓': (8, 12)}
WENXI_POINTS = {'平胡': (2, 1), '对对胡': (4, 2), '豪华大对': (6, 3), '清一色': (6, 3),
                '七对': (6, 3), '清七对': (8, 4), '天胡': (8, 0), '地胡': (0, 4), '单钓': (4, 2)}


def _qingtian_pattern(hand, melds, ctx, schedule, column):
    if ctx.opening:
        return ctx.opening
    candidates = ['平胡']
    suits = {tile // 9 for _, tile in melds} | {tile // 9 for tile, count in enumerate(hand) if count}
    if len(suits) == 1:
        candidates.append('清一色')
    if seven_pairs(hand, melds):
        candidates.append('清七对' if len(suits) == 1 else '七对')
    if len(melds) == 4:
        candidates.append('单钓')
    if all_triplets(hand, melds):
        candidates.append('对对胡')
        if all(kind == 'gang_an' for kind, _ in melds):
            candidates.append('豪华大对')
    return max(candidates, key=lambda pattern: schedule[pattern][column])


def _qingtian_payment(hand, melds, ctx, *, lishui):
    schedule = LISHUI_POINTS if lishui else WENXI_POINTS
    last_four = lishui and ctx.kind == 'ron' and ctx.tiles_left < 4
    column = int(ctx.kind == 'ron' and not last_four)
    # An earthly win remains tied to the discard column even in a fixture with a short wall.
    if ctx.opening == '地胡':
        column = 1
        last_four = False
    pattern = _qingtian_pattern(hand, melds, ctx, schedule, column)
    points = schedule[pattern][column]
    patterns = [pattern]
    provider = ctx.provider
    covered = last_four or ctx.kind == 'rob_kong'
    if last_four:
        patterns.append('尾牌包三')
    if ctx.kind == 'rob_kong':
        patterns.append('抢杠')
    if ctx.kind == 'zimo' and ctx.after_kong:
        patterns.append('杠上花')
        points *= 2 if lishui else 1
        if ctx.kong_provider is not None:
            provider = ctx.kong_provider
            covered = True
    if lishui and ctx.kind == 'zimo' and not ctx.tiles_left:
        patterns.append('海底')
        points *= 2
    if lishui and ctx.kind == 'ron' and ctx.winner_count == 3:
        patterns.append('一炮三响')
        points *= 2
    return tuple(patterns), points, _shares(ctx.actor, provider, points, covered=covered)


def lishui_payment(hand, melds, ctx):
    return _qingtian_payment(hand, melds, ctx, lishui=True)


def wenxi_payment(hand, melds, ctx):
    return _qingtian_payment(hand, melds, ctx, lishui=False)


def wake_award(hand, melds, wall, indicators):
    """Count one winner's shared indicator prefix and extend it when their kong hits."""
    counts = hand.copy()
    kongs = {tile for kind, tile in melds if kind != 'peng'}
    for kind, tile in melds:
        counts[tile] += 3 if kind == 'peng' else 4
    hits = used = 0
    while used < len(indicators) or wall:
        if used == len(indicators):
            indicators.append(wall.pop())
        suit, rank = divmod(indicators[used], 9)
        woke = (suit * 9 + (rank - 1) % 9, suit * 9 + (rank + 1) % 9)
        hits += sum(counts[tile] for tile in woke)
        used += 1
        if kongs.isdisjoint(woke):
            break
    return used, hits


def guilin_payment(hand, melds, ctx):
    points = (2 if ctx.kind == 'zimo' else 1) + ctx.wake_hits
    pattern = {'zimo': '自摸', 'ron': '平胡', 'rob_kong': '抢杠'}[ctx.kind]
    patterns = (pattern, '翻醒') if ctx.wake_hits else (pattern,)
    return patterns, points, _shares(ctx.actor, ctx.provider, points)
