"""Versioned playable rules; English is the API default."""
from .hongzhong import FANGCHENGGANG, ONE_MA
from .hongkong import VARIANT as HONG_KONG
from .suited import Profile

VARIANTS = {'riichi': {'id': 'riichi',
            'family_id': 'riichi',
            'name': 'Riichi Mahjong',
            'family': 'Japanese riichi',
            'rule_version': 'mortal-hanchan-v1',
            'subtitle': 'East–South match · Red fives · You vs three AI players',
            'badge': 'Classic hanchan',
            'description': 'Play a complete East–South match. The native engine handles calls, riichi, '
                           'furiten, wins, draws and dealer repeats.',
            'rules': ['Four-player East–South match, starting at 25,000 points each.',
                      'Red fives are included. A winning hand needs a yaku; dora alone is not a yaku.',
                      'Riichi costs 1,000 points. After declaring it, choose a legal ready-hand discard.',
                      'Uses the existing Mortal rules, including dealer repeats and any required west-round '
                      'extension.'],
            'default_policy': 'model',
            'policies': [{'id': 'model', 'name': 'Standard · Dojo model'},
                         {'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
            'translations': {'zh-CN': {'name': '立直麻将',
                                       'family': '日式立直',
                                       'subtitle': '东南半庄 · 赤宝牌 · 一人对战三位 AI',
                                       'badge': '经典半庄',
                                       'description': '从东一局开始打到半庄结束。支持吃碰杠、立直、荣和与自摸，完整处理振听、流局和连庄。',
                                       'rules': ['四人东南半庄，每人 25,000 点起始。',
                                                 '带赤五，胡牌需要有役；宝牌不单独构成役。',
                                                 '立直需支付 1,000 点，宣告后请选择听牌的弃牌。',
                                                 '采用现有 Mortal 引擎规则，包含连庄及必要的西场延长。']}}},
 'sichuan': {'id': 'sichuan',
             'family_id': 'sichuan',
             'name': 'Sichuan Xuezhan',
             'family': 'Sichuan blood battle',
             'rule_version': 'xuezhan-web-v0-20261116',
             'subtitle': 'Choose a void suit · Multiple winners · Hand-by-hand scoring',
             'badge': 'Blood battle',
             'description': 'Characters, dots and bamboo only. Choose a void suit, then keep playing until '
                            'three players have won or the wall is empty.',
             'rules': ['108 suited tiles. Choose a void suit; discard that suit first while you still hold '
                       'it.',
                       'No chi. Pon and kan are allowed; wins take priority and one discard can have '
                       'multiple winners.',
                       'Four-fan cap, immediate kong payments, exhaustive-draw ready-hand payments and '
                       'refunds.',
                       'Basic Xuezhan profile: no three-tile exchange, call transfer or flower-pig '
                       'penalties. Local room rules may differ.'],
             'default_policy': 'model',
             'policies': [{'id': 'model', 'name': 'Standard · Dojo model'},
                          {'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
             'translations': {'zh-CN': {'name': '四川血战到底',
                                        'family': '四川血战',
                                        'subtitle': '定缺血战 · 多家胡牌 · 一局一结算',
                                        'badge': '轻松开局',
                                        'description': '只用万、筒、条。先选定缺花色，胡牌后离场，其余玩家继续，直到三家胡牌或牌墙耗尽。',
                                        'rules': ['108 张数牌；开局定缺，有缺门牌时必须先打缺门牌。',
                                                  '不吃，可碰和杠；胡牌优先于碰杠，允许一炮多响。',
                                                  '四番封顶，含刮风下雨、流局查大叫和退税。',
                                                  '当前为基础血战规则：不含换三张、呼叫转移、查花猪；不是所有地方房间的统一规则。']}}},
 'hongzhong': {'id': 'hongzhong',
               'family_id': 'hongzhong',
               'name': 'Red Dragon Mahjong',
               'family': 'Red-dragon wildcard family',
               'rule_version': 'hongzhong-dojo-2ma-v1-20261116',
               'subtitle': 'Hongzhong · Wild red dragons · Two bonus tiles',
               'badge': '112 tiles · Dojo room profile',
               'description': 'A four-player wildcard game: win by self-draw or by robbing an added kong. '
                              'This room uses a fixed two-bonus-tile profile with the choices below.',
               'rules': ['112 tiles: the three suits plus four red dragons. Reds are wild in a winning hand, '
                         'but cannot replace another tile in a pon or kan. Reds themselves may be called.',
                         'No chi and no ordinary discard wins. Pon, kan, self-draw wins and multiple wins on '
                         'a robbed added kong are allowed.',
                         'A win costs each loser 2 points, plus 2 per bonus hit. Flip 2 bonus tiles; 1/5/9 '
                         'and red dragons hit. A winner with no red anywhere in the hand, including melds, '
                         'uses one extra bonus tile. Use only the tiles left in the wall.',
                         'Open kong: the discarder pays 3. Added kong: each other player pays 1. Closed '
                         'kong: each pays 2.',
                         'The winner deals next. After a draw, the last player to draw deals next.',
                         'Dojo choices: standard four sets and a pair; seven pairs is off. Four concealed '
                         'red dragons may win on your own draw turn, including the dealer’s initial turn, '
                         'with normal win and bonus scoring.',
                         'Dojo choices: wins may be declined; no passed-win lock. No kong without a '
                         'replacement tile. After an unwon last draw and discard, the hand ends.',
                         'Dojo choices: kong payments stand after a draw; there is no ready-hand payment. '
                         'Multiple robbed-kong winners share the bonus sequence, each receives three shares '
                         'from the kong maker, and the nearest winner deals next.'],
               'default_policy': 'heuristic',
               'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
               'sources': [{'title': 'GameTea rule article (2018)',
                            'url': 'https://gametea.net/news/201809/7405.html'}],
               'translations': {'zh-CN': {'name': '红中麻将 · 双码',
                                          'family': '红中百搭家族',
                                          'subtitle': '红中百搭 · 自摸与抢杠 · 扎双码',
                                          'badge': '112 张 · 道场房间规则',
                                          'description': '四人红中百搭玩法，支持自摸、抢补杠和扎码。本桌采用固定双码规则；原始来源未明确的边界按下列道场设置执行。',
                                          'rules': ['112 张：三门数牌加四张红中。红中胡牌时作百搭，不能代替其他牌碰杠；允许碰杠红中本身。',
                                                    '不吃，普通弃牌不能胡；允许碰、杠、自摸和多人抢补杠。',
                                                    '胡牌每家付 2 分，每中一码再付 2 分。固定扎 2 码，1/5/9 '
                                                    '与红中命中；完整胡牌手牌（含副露）无红中多扎一码。牌墙不足则只翻剩余牌。',
                                                    '接杠由出牌者付 3 分；补杠其余每家付 1 分；暗杠其余每家付 2 分。',
                                                    '胡者接庄；流局由最后摸牌者接庄。',
                                                    '道场设置：仅标准四面子一对，暂不开七对；四张暗手红中在自己的摸牌行动窗口可胡，含庄家起手，按普通自摸加码结算。',
                                                    '道场设置：允许过胡，不设过胡锁；没有补牌时不能杠；最后摸牌无人胡并打出后即流局。',
                                                    '道场设置：流局保留已结算杠分，不查叫；多人抢杠共享码牌序列，杠家向每位赢家包三份，按响应顺序最近的赢家接庄。']}}}}

_DOUBLE = VARIANTS['hongzhong']
_DOUBLE['aliases'] = ['Hongzhong', 'Hong Zhong', '红中麻将', '雙碼']
_ZH_DOUBLE = _DOUBLE['translations']['zh-CN']

VARIANTS['hongzhong-one-ma'] = {
    **_DOUBLE,
    'id': 'hongzhong-one-ma',
    'name': 'Red Dragon Mahjong · One Ma',
    'rule_version': 'hongzhong-dojo-one-ma-v1-20261118',
    'subtitle': 'Hongzhong · Wild red dragons · Every bonus rank pays',
    'description': 'A Hongzhong room with single-tile rank scoring after a self-draw or robbed kong.',
    'rules': [*_DOUBLE['rules'][:2],
              'Each loser pays 2 base points. Flip one bonus tile if the wall has any left. '
              'A suited tile adds its rank plus 1 per share, and a red dragon adds 11. '
              'The Dojo one-ma room draws at most one tile, including for hands without a red dragon.',
              *_DOUBLE['rules'][3:]],
    'translations': {'zh-CN': {**_ZH_DOUBLE,
                               'name': '红中麻将 · 一码全中',
                               'subtitle': '红中百搭 · 自摸与抢杠 · 一码按点数加分',
                               'description': '四人红中百搭玩法。自摸或抢补杠后翻一张码牌，按牌面点数加分。',
                               'rules': [*_ZH_DOUBLE['rules'][:2],
                                         '每份基本胡牌分为 2 分。牌墙有剩余时只翻一张码；数牌按点数加 1 计加分，红中加 11 分。'
                                         '本桌一码全中固定最多翻一张，无红中也不加翻。',
                                         *_ZH_DOUBLE['rules'][3:]]}},
}

VARIANTS['fangchenggang-hongzhong'] = {
    **_DOUBLE,
    'id': 'fangchenggang-hongzhong',
    'name': 'Fangchenggang Red Dragon Mahjong',
    'aliases': ['Fangchenggang', 'Guangxi', '广西', '廣西', '防城港紅中麻將'],
    'rule_version': 'fangchenggang-hongzhong-four-ma-v1-20261118',
    'subtitle': 'Guangxi · Seven pairs · Four reserved bonus tiles',
    'badge': 'Fangchenggang · 112 tiles',
    'description': 'A four-player Fangchenggang room with wild red dragons, seven pairs and a four-tile bonus reserve.',
    'rules': [
        _DOUBLE['rules'][0],
        'No chi or ordinary discard wins. Win by self-draw or by robbing another player’s added kong. '
        'Several players can rob the same kong. A red dragon cannot be robbed.',
        'Standard hands and concealed seven pairs can win. Seven pairs pays the same as a standard win. '
        'Four concealed reds on your draw turn can win and use normal bonus scoring.',
        'A self-draw costs each opponent 2 points. Flip four tiles from the far end of the wall; '
        '1, 5, 9 and red dragons add 2 points each per share. The kong maker pays all three shares to each robber.',
        'Open kong: the discarder pays 3. Added kong: each opponent pays 1. Closed kong: each pays 2. '
        'An added kong must use the tile just drawn; all kong replacements come from the far end.',
        'The last four tiles are held for bonuses. Dojo choices: no calls after the last playable tile '
        'is discarded, no kong at the reserve boundary, no extra bonus for a red-free hand, and kong payments stand on drawn hands.',
        'The winner deals next, or the last drawer after a draw. If a kong is robbed its maker deals next.',
    ],
    'sources': [{'title': 'Fangchenggang rules (Zhejiang Game Hall)', 'url': 'https://www.hzbianqu.com/game/1123.html'}],
    'translations': {'zh-CN': {
        'name': '防城港红中麻将', 'family': _ZH_DOUBLE['family'],
        'subtitle': '广西 · 可胡七对 · 留四张码牌', 'badge': '防城港 · 112 张',
        'description': '采用红中百搭、可胡七对和留四码的防城港四人房间规则。',
        'rules': [
            _ZH_DOUBLE['rules'][0],
            '不吃，普通弃牌不胡；可自摸和抢补杠，可多人抢杠，红中不能被抢杠。',
            '可胡标准手牌或门清七对，七对不加分。自己摸牌回合暗手四红中可胡，按普通扎码结算。',
            '自摸每家付 2 分。由牌墙另一端翻四张码，数牌 1、5、9 与红中每中一张每份加 2 分。'
            '抢杠由杠家向每位赢家包三份。',
            '接杠由出牌者付 3 分；补杠其余每家付 1 分；暗杠其余每家付 2 分。补杠须立即用刚摸的同张牌，杠后从墙尾补牌。',
            '最后四张留作码牌。道场设置：最后可摸牌打出后结束，剩四张时禁止杠，无红中不额外翻码，流局保留既有杠分。',
            '自摸者接庄，流局由最后摸牌者接庄，被抢杠者接庄。',
        ],
    }},
}

_SUITED = {
    'family_id': 'suited-discard-win',
    'family': 'Suited discard-win',
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
}
_WENZHOU = Profile.WENZHOU.value
_CHANGSHA = Profile.CHANGSHA.value
_TAISHUN = Profile.TAISHUN.value
_XUANHUA = Profile.XUANHUA.value
VARIANTS[_WENZHOU] = {
    **_SUITED,
    'id': _WENZHOU,
    'name': 'Wenzhou Fangpao Mahjong',
    'aliases': ['Wenzhou', 'Zhejiang', '温州', '浙江', '温州放炮', '溫州麻將'],
    'rule_version': 'wenzhou-fangpao-dojo-v1-20261118',
    'subtitle': 'Zhejiang · Multiple discard wins · Fan scoring',
    'badge': 'Wenzhou · 108 tiles',
    'description': 'A suited Wenzhou room with discard wins, same-suit declined-win restrictions and fan scoring.',
    'rules': [
        '108 suited tiles. Pon and kong are allowed; no chi, void selection or wildcards.',
        'A self-draw or the first winning discard ends the hand. One discard may have several winners. '
        'If you decline an ordinary discard win, discard wins in that suit are locked until your next draw.',
        'Take the highest base fan: ordinary 1, all triplets 2, one suit / seven pairs / golden hook 3, '
        'one-suit triplets 4, dragon seven pairs / one-suit seven pairs / one-suit hook 5, one-suit dragon seven pairs 6.',
        'Self-draw, winning on or immediately after a kong replacement, and robbing an added kong each '
        'add 1 when applicable without stacking. Heavenly and earthly wins are fixed at 6 fan in this room.',
        'The payer owes 2 to the power of fan minus 1. All opponents pay for self-draw; the provider pays for '
        'each discard or robbed-kong winner. Kong transfers are immediate: open 1, added 1 each, concealed 2 each.',
        'First dealer is random. A sole winner deals next, or the provider after multiple wins. '
        'The dealer repeats on a draw. The final discard allows wins, with no further melds.',
    ],
    'sources': [{'title': 'GameTea Wenzhou Fangpao rules', 'url': 'https://www.gametea.com/games/fangpaomj.html'}],
    'translations': {'zh-CN': {
        'name': '温州放炮麻将', 'family': '数牌点炮家族',
        'subtitle': '浙江 · 一炮多响 · 按番算分', 'badge': '温州 · 108 张',
        'description': '温州数牌房间，支持点炮胡、按花色过水和番型结算。',
        'rules': [
            '108 张数牌。不吃、可碰杠，不定缺、无财神。',
            '自摸或首次有人点炮胡后结束，允许一炮多响。过普通点炮胡后，同花色的点炮胡锁至自己再次摸牌。',
            '基础番只取最高：平胡 1、对对胡 2、清一色／七对／金钩钩 3、清对 4、龙七对／清七对／清金钩钩 5、清龙七对 6。',
            '自摸、杠上花、杠上炮、抢杠符合条件各加一番且不叠加。本桌天胡、地胡固定六番。',
            '每份胡牌分为 2 的（番数减 1）次方，自摸三家付，点炮或抢杠由供牌者分别支付各赢家。'
            '直杠供牌者付 1，补杠其他每家付 1，暗杠其他每家付 2。',
            '首庄随机，单人胡由赢家接庄，多响由供牌者接庄；流局连庄。最后一张弃牌可胡，不再碰杠。',
        ],
    }},
}
VARIANTS[_CHANGSHA] = {
    **_SUITED,
    'id': _CHANGSHA,
    'name': 'Changsha Zhuanzhuan Mahjong',
    'aliases': ['Changsha', 'Hunan', '长沙', '湖南', '转转麻将', '轉轉麻將'],
    'rule_version': 'changsha-zhuanzhuan-91sui-dojo-v1-20261118',
    'subtitle': 'Hunan · Seven pairs · Flat scoring plus kongs',
    'badge': 'Changsha · 108 tiles',
    'description': 'The 91sui-style Zhuanzhuan room with simple points, seven pairs and multiple wins on a discard.',
    'rules': [
        '108 suited tiles. No chi, wildcards, void suit, or restriction on the pair.',
        'Standard hands and concealed seven pairs may win. This room allows discard wins, self-draw, '
        'robbing added kongs and multiple winners. The first win ends the hand after all replies.',
        'One point plus the winner’s completed kong count per liable player. All opponents pay on '
        'self-draw; the discarder or kong maker pays on a shared tile. No separate kong payments or bonus birds.',
        'Kongs require a replacement draw and an added kong uses the tile just drawn. '
        'The final discard allows wins, with no further melds. The next dealer is randomized among non-winners.',
    ],
    'sources': [{'title': '91sui Changsha Zhuanzhuan rules', 'url': 'https://www.91sui.com/Home/GameDetail?id=11'}],
    'translations': {'zh-CN': {
        'name': '长沙转转麻将', 'family': '数牌点炮家族',
        'subtitle': '湖南 · 可胡七对 · 固定底分加杠', 'badge': '长沙 · 108 张',
        'description': '依据 91 随便玩版本的转转房间，支持七对、一炮多响和带杠胡结算。',
        'rules': [
            '108 张数牌，不吃、无百搭、不定缺，任意对子可作将。',
            '可胡标准型与门清七对。可自摸、点炮和抢补杠，允许多响，询问所有响应者后结束。',
            '每份胡牌分为 1 加赢家的成杠数。自摸三家付，点炮或被抢杠由供牌者支付。此桌不另结杠分或抓鸟。',
            '杠须有补牌，补杠须用刚摸的同张牌。最后一张弃牌可胡，不再碰杠。下一局在未胡玩家中随机选庄。',
        ],
    }},
}

VARIANTS[_TAISHUN] = {
    **_SUITED,
    'id': _TAISHUN,
    'name': 'Taishun Mahjong',
    'aliases': ['Taishun', 'Zhejiang', '泰顺', '泰顺麻將', '泰順', '浙江'],
    'rule_version': 'taishun-dojo-v1-20261118',
    'subtitle': 'Zhejiang · Multiple discard wins · Point scoring',
    'badge': 'Taishun · 108 tiles',
    'description': 'A Zhejiang room with distinct discard and self-draw payments, and an all-suit passed-win lock.',
    'rules': [
        '108 suited tiles. Pon and kong are allowed. A discard can have multiple winners.',
        'Passing a discard win locks all discard wins until you draw or successfully take a pon or kong.',
        'Highest base per payer, discard / self-draw: ordinary 1 / 2, all triplets 2 / 4, '
        'one suit or seven pairs 3 / 6, dragon seven pairs 6 / 12.',
        'The initial dealer win and a win on the dealer’s first discard each score 6 per liable player. '
        'A win on the discard after a kong scores 3. A replacement self-draw doubles the self-draw share. '
        'On robbery the kong maker pays three doubled self-draw shares.',
        'Kong shares are 1 from the provider for a direct kong, 1 from each opponent for an added kong, '
        'and 2 each for a concealed kong. Upgrading an immediate pon with a held fourth tile earns no kong points. '
        'Kong points stand on a draw.',
        'You deal the first hand. A sole winner or the provider after multiple wins deals next. '
        'The dealer repeats on a draw.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Taishun rules', 'url': 'https://www.hzbianqu.com/game/63.html'}],
    'translations': {'zh-CN': {
        'name': '泰顺麻将', 'family': '数牌点炮家族',
        'subtitle': '浙江 · 一炮多响 · 按牌型算分', 'badge': '泰顺 · 108 张',
        'description': '支持一炮多响，胡牌按点炮、自摸分别算分，过胡覆盖所有花色。',
        'rules': [
            '108 张数牌，可碰杠，不吃，允许一炮多响。',
            '过普通点炮胡后，所有花色不能接炮，直到自己摸牌或成功碰、杠。',
            '点炮／自摸每份取最高：平胡 1／2，大对 2／4，清一色或七对 3／6，豪七对 6／12。',
            '天胡、庄家首打的地胡每份 6 分。杠上炮 3 分，杠上开花按自摸分翻倍。'
            '抢补杠按自摸分翻倍，由杠家包付三份。',
            '直杠供牌者付 1，补杠其他每家付 1，暗杠其他每家付 2。选四张中的三张碰后，'
            '立即用手里第四张补杠不计杠分。流局保留杠分。',
            '首局你坐庄；单胡赢家接庄，多响供牌者接庄，流局连庄。',
        ],
    }},
}
VARIANTS[_XUANHUA] = {
    **_SUITED,
    'id': _XUANHUA,
    'name': 'Xuanhua Mahjong',
    'aliases': ['Xuanhua', 'Zhangjiakou', 'Hebei', '宣化', '张家口', '河北'],
    'rule_version': 'xuanhua-dojo-v1-20261118',
    'subtitle': 'Hebei · Closed-hand multiplier · Single kong robbery',
    'badge': 'Xuanhua · 108 tiles',
    'description': 'A Hebei room with multiple ordinary discard wins, nearest-first kong robbery, and closed-hand scoring.',
    'rules': [
        '108 suited tiles. Pon and kong are allowed. Regular hands and seven pairs can win without declaring ready.',
        'Passing a discard win blocks all ordinary discard wins until your next discard.',
        'The base is 1 per payer. A closed hand and self-draw each double it; concealed kongs keep the hand closed. '
        'An initial dealer win pays 6 from each opponent. There is no earthly win bonus.',
        'An ordinary discard may have multiple winners: the discarder pays each. Only the nearest accepting player '
        'can rob an added kong; all three opponents pay as on a self-draw.',
        'All three opponents pay each kong: 1 each for direct or added kongs, 2 each for concealed kongs. '
        'Kong points stand on a draw.',
        'You deal the first hand. A sole winner or the discarder after multiple wins deals next. '
        'The dealer repeats on a draw.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Xuanhua rules', 'url': 'https://www.hzbianqu.com/game/309.html'}],
    'translations': {'zh-CN': {
        'name': '宣化麻将', 'family': '数牌点炮家族',
        'subtitle': '河北 · 门清翻倍 · 抢杠截胡', 'badge': '宣化 · 108 张',
        'description': '支持普通点炮多响，抢补杠截胡按自摸计分，门清翻倍。',
        'rules': [
            '108 张数牌，可碰杠，不吃。不报听即可胡标准型或七对。',
            '过普通点炮胡后，在自己下次打牌前不能接炮。',
            '每份底分 1，门清和自摸各翻倍。暗杠保持门清。天胡每家固定付 6，无地胡。',
            '普通点炮允许多响，供牌者分别支付。抢补杠只由顺位最近且选择胡牌者胡，其他三家按自摸付分。',
            '关闭「点杠包一家」，直杠和补杠由其他每家付 1，暗杠其他每家付 2。流局保留杠分。',
            '首局你坐庄；单胡赢家接庄，多响点炮者接庄，流局连庄。',
        ],
    }},
}

VARIANTS[HONG_KONG] = {
    'id': HONG_KONG,
    'family_id': 'honors-and-chows',
    'family': 'Honors and chows',
    'name': 'Hong Kong Old Style',
    'aliases': ['Hong Kong', 'HKOS', 'Cantonese', '香港', '港式', '港麻', '清章', '廣東牌', '广东牌'],
    'rule_version': 'hong-kong-old-style-no-flower-dojo-v1-20261118',
    'subtitle': 'Hong Kong · No flowers · Three faan to win',
    'badge': 'Hong Kong · 136 tiles',
    'description': 'No-flower Hong Kong Old Style. Chow from upstream, build a three-faan hand, and settle up to eight faan.',
    'rules': [
        '136 suited and honor tiles. Chow from the preceding seat only. Wins beat pong and kong, which beat chow. '
        'A chow blocks discarding its called tile and the opposite extension immediately afterward.',
        'A winning hand needs at least three faan and pays at most eight. Four sets and a pair or Thirteen Orphans '
        'can qualify. Common patterns include all sequences (1), all triplets (3), half flush (3) and full flush (7).',
        'Ordinary discard wins can be shared; the discarder pays each winner. Self-draw pays from all opponents. '
        'The supplier of a fourth declared meld or third exposed dragon set may owe all three self-draw shares.',
        'Added-kong robbery pays three self-draw shares from the kong maker. Only Thirteen Orphans can rob a '
        'concealed kong. Declining a win or callable tile locks that same tile until your own draw.',
        'The first dealer is random, retains the seat on a win, and passes the seat otherwise. A drawn hand scores zero. '
        'Seat and prevailing wind triplets count separately in this room.',
    ],
    'sources': [{'title': 'CUHK alumni tournament, no-flower rules (PDF)',
                 'url': 'https://home.culhk.org/files/ICUD/2016ICUD-Mahjong_Comp-Regulation.pdf'},
                {'title': 'Hong Kong Mahjong Association rule index', 'url': 'https://www.hkmahjong.org/rules?lang=en'}],
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'translations': {'zh-CN': {
        'name': '港式清章（无花）', 'family': '字牌与吃牌家族',
        'subtitle': '香港 · 无花牌 · 三番起胡', 'badge': '香港 · 136 张',
        'description': '港式无花桌，可吃上家的牌，三番起胡，八番封顶。',
        'rules': [
            '136 张数牌和字牌，只能吃上家。胡优先于碰杠，碰杠优先于吃。吃后当手不能打出所吃同张或另一端可换的牌。',
            '三番起胡，八番封顶。四组一对或十三幺可胡。常见番型包括平胡 1、对对胡 3、混一色 3、清一色 7。',
            '普通点炮可多响，由点炮者分别支付。自摸时三家付分；供第四组副露或第三组明三元者可能包自摸三份。',
            '抢补杠由开杠者包付三份。只有十三幺可抢暗杠。放弃可胡或可吃碰杠的牌后，同张锁至自己摸牌。',
            '首庄随机，庄家胡则连庄，否则轮庄。流局零分。门风刻与圈风刻分别计番。',
        ],
    }},
}

SESSION_OPTIONS = {
    'hongzhong-one-ma': {'variant': 'hongzhong-one-ma', 'profile': ONE_MA},
    'fangchenggang-hongzhong': {'variant': 'fangchenggang-hongzhong', 'profile': FANGCHENGGANG},
    _WENZHOU: {'profile': _WENZHOU},
    _CHANGSHA: {'profile': _CHANGSHA},
    _TAISHUN: {'profile': _TAISHUN},
    _XUANHUA: {'profile': _XUANHUA},
}
