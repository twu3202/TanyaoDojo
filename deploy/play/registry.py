"""Versioned playable rules; English is the API default."""
from .blood_battle import SICHUAN_EXCHANGE, YUNNAN
from .guangzhou import VARIANT as GUANGZHOU
from .golden_guangdong import VARIANT as GOLDEN_GUANGDONG
from .guiliu import VARIANT as GUILIU
from .hangzhou import FOUR_HANGZHOU, HANGZHOU_ROOMS, THREE_HANGZHOU
from .honors_bonus import HEZHOU, HUNDRED, LIUZHOU, SHANGSI
from .hongzhong import FANGCHENGGANG, ONE_MA
from .hongkong import VARIANT as HONG_KONG
from .jj_multiplier import VARIANT as JJ_MULTIPLIER
from .jj_three import VARIANT as JJ_THREE
from .kaxinwu import VARIANT as KAXINWU
from .kawuxing import SHIYAN, SUIZHOU, XIANGYANG
from .macau import VARIANT as MACAU
from .suited import Profile
from .xiaogan import XIAOGAN

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

_SHAANXI = Profile.SHAANXI.value
_LISHUI = Profile.LISHUI.value
_WENXI = Profile.WENXI.value
_GUILIN = Profile.GUILIN.value

VARIANTS[_SHAANXI] = {
    **_SUITED,
    'id': _SHAANXI, 'name': 'Shaanxi Mahjong',
    'aliases': ['Shaanxi', '陕西', '陝西', '边锋陕西'],
    'rule_version': 'shaanxi-bianfeng-dojo-v1',
    'subtitle': 'Shaanxi · Nearest winner · Dealer doubles', 'badge': 'Shaanxi · 108 tiles',
    'description': 'The 108-tile Bianfeng room with nearest-first wins and dealer-relative payments.',
    'rules': [
        '108 suited tiles. Pon and kong are allowed. Four sets and one pair can win; the seven-pairs option is off.',
        'A discard or added kong has at most one winner, the nearest seat in play order that accepts.',
        'Passing a discard win prevents winning on that same tile until you next draw or successfully call pon or kong.',
        'A discard or robbed kong pays 1; self-draw pays 2 from each opponent. Double each win payment involving '
        'the dealer. Paozi is fixed at zero. Direct kong pays 1 from the provider; added and concealed kongs pay 1 and 2 '
        'from each opponent, respectively. Keep kong points on a draw.',
        'First dealer is random. The winner deals next; on a draw the dealer stays.',
    ],
    'sources': [{'title': 'Bianfeng Shaanxi rules', 'url': 'https://www.hzbianqu.com/game/360.html'}],
    'translations': {'zh-CN': {
        'name': '陕西麻将', 'family': '数牌点炮家族', 'subtitle': '陕西 · 截胡 · 庄家翻倍',
        'badge': '陕西 · 108 张', 'description': '边锋陕西数牌房，支持截胡、同张过胡和庄家翻倍。',
        'rules': ['108 张，只碰杠、不吃。可胡四组一对，本桌关闭七对。',
                  '点炮和抢补杠均截胡，由顺位最近且接受胡牌的一家胡。',
                  '过胡同张锁至自己摸牌或成功碰杠，其他胡张不受影响。',
                  '点炮或抢杠供牌者付 1，自摸其他每家付 2。胡分涉及庄家时加倍。本桌炮子固定 0。'
                  '直杠供牌者付 1，补杠其他每家付 1，暗杠其他每家付 2。流局保留杠分。',
                  '首庄随机，赢家接庄，流局连庄。'],
    }},
}
VARIANTS[_LISHUI] = {
    **_SUITED,
    'id': _LISHUI, 'name': 'Lishui Fangpao Mahjong',
    'aliases': ['Lishui', 'Qingtian Fangpao', '丽水', '麗水', '青田放炮'],
    'rule_version': 'lishui-fangpao-dojo-v1',
    'subtitle': 'Zhejiang · Last four tiles · Three-winner bonus', 'badge': 'Lishui · 108 tiles',
    'description': 'A Qingtian-area room with separate discard and self-draw schedules and end-of-wall payments.',
    'rules': [
        '108 suited tiles. Pon, kong, seven pairs and multiple discard winners are allowed.',
        'Base share, self-draw / discard: ordinary 2 / 1, all triplets 4 / 6, four concealed sets 12 / 18, '
        'one suit 8 / 12, seven pairs 6 / 9, one-suit seven pairs 12 / 18, four exposed sets 8 / 12. '
        'Opening dealer self-draw is 16; a win on the first discard is 20.',
        'A kong replacement and a final-tile self-draw each double the self-draw share. The provider of a direct '
        'kong covers three shares on its replacement win. The provider also covers three self-draw shares on '
        'a robbed kong or a discard from any of the last four draws. Three simultaneous discard wins double the bills.',
        'Direct kong pays 3 from the provider, added kong pays 1 each and concealed kong pays 2 each. '
        'Kong transfers are reversed on a draw.',
    ],
    'sources': [{'title': 'Lishui Fangpao room rules', 'url': 'https://www.hzbianqu.com/game/1066.html'},
                {'title': 'Qingtian scoring table', 'url': 'https://www.hzbianqu.com/article/1519.html'}],
    'translations': {'zh-CN': {
        'name': '丽水放炮麻将', 'family': '数牌点炮家族', 'subtitle': '浙江 · 尾四牌包三 · 三响翻倍',
        'badge': '丽水 · 108 张', 'description': '青田一带的放炮玩法，点炮与自摸有不同分表。',
        'rules': ['108 张数牌，可碰杠、七对和一炮多响。',
                  '每份自摸／点炮分别为平胡 2／1、对对胡 4／6、豪华大对 12／18、清一色 8／12、七对 6／9、'
                  '清七对 12／18、单钓 8／12。天胡每家 16，庄家首打的地胡付 20。',
                  '杠开和海底自摸各翻倍，直杠补牌胡由点杠者包三。抢杠或最后四次摸牌后点炮，'
                  '供牌者按自摸分包三份。一炮三响再翻倍。',
                  '直杠供牌者付 3，补杠其他每家付 1，暗杠其他每家付 2。流局返还所有杠分。'],
    }},
}
VARIANTS[_WENXI] = {
    **_SUITED,
    'id': _WENXI, 'name': 'Wenxi Mahjong',
    'aliases': ['Wenxi', 'Qingtian Wenxi', '温溪', '溫溪'],
    'rule_version': 'wenxi-dojo-v1',
    'subtitle': 'Zhejiang · Direct-kong liability · Draw refunds', 'badge': 'Wenxi · 108 tiles',
    'description': 'A separate Qingtian ruleset with its own base scores and direct-kong replacement payments.',
    'rules': [
        '108 suited tiles. Pon and kong are allowed. Standard hands and seven pairs can win.',
        'Base share, self-draw / discard: ordinary 2 / 1, all triplets 4 / 2, four concealed sets 6 / 3, '
        'one suit 6 / 3, seven pairs 6 / 3, one-suit seven pairs 8 / 4, four exposed sets 4 / 2. '
        'Opening dealer self-draw is 8; a win on the first discard is 4.',
        'Replacement wins use the self-draw share. A direct kong provider covers all three shares on the '
        'replacement win. This room also allows multiple wins and treats an added-kong robbery as a covered self-draw.',
        'Direct kong pays 1 from the provider, added kong pays 1 each and concealed kong pays 2 each. '
        'Kong transfers are reversed on a draw.',
    ],
    'sources': [{'title': 'Wenxi room rules', 'url': 'https://www.hzbianqu.com/game/60.html'}],
    'translations': {'zh-CN': {
        'name': '温溪麻将', 'family': '数牌点炮家族', 'subtitle': '浙江 · 直杠杠开包三 · 流局退杠',
        'badge': '温溪 · 108 张', 'description': '青田温溪数牌玩法，按本地分表结算，直杠杠开由供牌者承包。',
        'rules': ['108 张数牌，可碰杠，能胡标准型或七对。',
                  '每份自摸／点炮分别为平胡 2／1、对对胡 4／2、豪华大对 6／3、清一色 6／3、七对 6／3、'
                  '清七对 8／4、单钓 4／2。天胡每家 8，庄家首打的地胡付 4。',
                  '杠开按自摸分。直杠补牌胡由点杠者包三。本桌采用一炮多响，抢补杠按自摸包三。',
                  '直杠供牌者付 1，补杠其他每家付 1，暗杠其他每家付 2。流局返还所有杠分。'],
    }},
}
VARIANTS[_GUILIN] = {
    **_SUITED,
    'id': _GUILIN, 'name': 'Guilin Zhuanzhuan Mahjong',
    'aliases': ['Guilin', 'Guangxi', '桂林', '桂林转转', '桂林轉轉', '广西'],
    'rule_version': 'guilin-zhuanzhuan-classic-dojo-v1',
    'subtitle': 'Guangxi · Classic room · Wake tiles', 'badge': 'Guilin · 108 tiles',
    'description': 'The classic suited room where winners reveal wake indicators after a win.',
    'rules': [
        '108 suited tiles. Pon, kong and seven pairs are allowed. Passing an ordinary discard win blocks '
        'further ordinary discard wins until your next draw. Multiple wins are allowed.',
        'Discard win or added-kong robbery pays 1 from the provider; self-draw pays 2 from each opponent. '
        'Direct kong pays 3 from the provider, added kong pays 1 each and concealed kong pays 2 each. Keep kong points on a draw.',
        'Reveal a wake indicator after winning. Its same-suit neighbors, wrapping between 1 and 9, each add '
        '1 per matching tile in the winning hand and melds. A matching kong extends the shared sequence for that winner.',
        'You deal first. The sole winner, or the provider after a multiple win, deals next. The dealer repeats on a draw.',
    ],
    'sources': [{'title': 'Guilin Zhuanzhuan classic rules', 'url': 'https://www.hzbianqu.com/game/1042.html'}],
    'translations': {'zh-CN': {
        'name': '桂林转转（经典）', 'family': '数牌点炮家族', 'subtitle': '广西 · 经典场 · 上下醒',
        'badge': '桂林 · 108 张', 'description': '经典数牌房，胡后翻醒牌，醒牌命中杠可续翻。',
        'rules': ['108 张数牌，可碰杠，可胡七对，允许一炮多响。过普通点炮胡后，自己再摸牌前不能接炮。',
                  '点炮或抢杠供牌者付 1，自摸其他每家付 2。直杠供牌者付 3，补杠其他每家付 1，'
                  '暗杠其他每家付 2。流局保留杠分。',
                  '胡后翻醒。同花色前后各一张为醒牌，1 与 9 循环。手牌和副露每命中一张每份加 1。'
                  '某次翻醒命中赢家的杠，该赢家继续看下一张醒牌，多响共用翻出的序列。',
                  '首局你坐庄，单胡赢家或多响供牌者接庄，流局连庄。'],
    }},
}

VARIANTS[Profile.POOK.value] = {
    **_SUITED,
    'id': Profile.POOK.value,
    'name': 'Pook Dazhong Mahjong',
    'aliases': ['Pook', 'Dazhong', '波克', '大众麻将', '波克大众', '波克大眾'],
    'rule_version': 'pook-dazhong-dojo-v1',
    'subtitle': '108 tiles · Optional ready · Higher-value wins after a pass',
    'badge': 'Dazhong · 108 tiles',
    'description': 'An optional ready declaration adds a point to each win payment. Passing a win sets a points threshold.',
    'rules': [
        '108 suited tiles. Pon, kong and seven pairs are allowed. A discard or added kong may have several winners.',
        'Choose a discard that leaves a ready hand to declare. Later discards must be the drawn tile. '
        'A drawn-tile added or concealed kong is allowed if it preserves all your winning tiles.',
        'After passing a win, only a larger per-payer amount can win until you draw or take a call. '
        'Passing a pon locks that tile for pon until then; an otherwise legal direct kong is still available.',
        'Discard pays 1, self-draw 2 per opponent, dealer opening and replacement self-draw 4 per opponent, '
        'and robbing a kong 6 from the kong maker. A ready declaration adds 1 to each of those shares.',
        'Add 1 for seven pairs or 2 for dragon seven pairs, 1 for all triplets or 2 for four melds and a single wait, '
        'and 4 for a pure suit. Direct kong pays 3 from the supplier, added kong 1 each and concealed kong 2 each.',
        'Keep kong transfers on a draw. A single winner or the multi-win provider deals next; the dealer repeats on a draw.',
    ],
    'sources': [{'title': 'Pook Dazhong official rules', 'url': 'https://www.pook.com/games/bkdzmj.html'}],
    'translations': {'zh-CN': {
        'name': '波克大众麻将', 'family': '数牌点炮家族', 'subtitle': '108 张 · 可报听 · 加分可越过胡',
        'badge': '大众 · 108 张', 'description': '可以报听，胡牌时每份加 1。过胡后本轮只可胡更高分的牌。',
        'rules': [
            '108 张数牌，可碰杠，可胡七对，允许一炮多响、一杠多抢。',
            '选中一张打出后能听牌的手牌即可报听，之后只能摸切；听口不变时，可以用刚摸的同张牌补杠或暗杠。',
            '过胡后，再摸牌或成功碰杠前只可胡更高单份分数的牌；过碰锁同张，原本可直杠时仍可杠。',
            '点炮 1，自摸其他每家 2，天胡及杠开其他每家 4，抢杠杠家付 6。报听胡每份再加 1。',
            '七对加 1 或龙七对加 2，对对胡加 1 或单钓加 2，清一色另加 4。'
            '直杠供牌者付 3，补杠其他每家付 1，暗杠其他每家付 2。',
            '流局保留杠分。单胡赢家或多响供牌者接庄，流局连庄。',
        ],
    }},
}

VARIANTS[Profile.DATANG.value] = {
    **_SUITED,
    'id': Profile.DATANG.value,
    'name': 'Shanxi Datang Tuidaohu',
    'aliases': ['Shanxi', 'Datang', 'Tuidaohu', '山西', '山西推倒胡', '大唐推倒胡', '大唐麻将'],
    'rule_version': 'shanxi-datang-tuidaohu-dojo-v2',
    'subtitle': 'Shanxi · 108 tiles · Declared discarder shares payment',
    'badge': 'Shanxi · 108 tiles',
    'description': 'A Shanxi room where a declared-ready discarder shares their discard-win or direct-kong bill with the table.',
    'rules': [
        '108 suited tiles, with no joker or missing-suit requirement. Pon and kong are allowed; use the highest winning pattern.',
        'Declare ready before winning by discarding to a ready hand. Discard only the tile you draw thereafter. '
        'A kong is still allowed if it leaves all your winning tiles unchanged.',
        'Passing a discard or robbed-kong win blocks those wins until your next discard. Self-draw remains available. '
        'Several players can win from one discard or added kong.',
        'Discard / self-draw share: regular 3 / 2, pure straight, pure suit or seven pairs 9 / 9, dragon seven pairs 18 / 18. '
        'On self-draw each opponent pays. On a discard or added-kong robbery, the provider pays.',
        'Once the provider declares ready, each non-winner pays a third of the discard bill, with no transfers between winners. '
        'The provider pays 3 for a direct kong unless already ready, in which case all three pay 1 each. '
        'Added kong pays 1 each; concealed kong pays 2 each. A draw refunds all kong points.',
        'You start as dealer. The sole winner or the provider after multiple wins deals next. On a draw the last player to draw deals.',
    ],
    'sources': [
        {'title': 'Datang Tuidaohu official rules', 'url': 'https://www.dtgames.cn/gonglue/30.html'},
        {'title': 'Datang ready-room options', 'url': 'https://www.dtgames.cn/gonglue/136.html'},
        {'title': 'Datang scoring examples', 'url': 'https://www.dtgames.cn/gonglue/1.html'},
    ],
    'translations': {'zh-CN': {
        'name': '山西大唐推倒胡', 'family': '数牌点炮家族', 'subtitle': '山西 · 108 张 · 报听点炮三家分摊',
        'badge': '山西 · 108 张', 'description': '报听者点炮或点杠后，其他输家也参与分摊。',
        'rules': [
            '108 张数牌版，不带耗子、不定缺，可碰杠，多个胡型取最高。',
            '胡牌前必须先报听。打一张后能听牌时可报听，之后只能摸切；听口不变可开杠。',
            '过点炮胡或抢杠胡后，下次打牌才解除，自摸仍可胡。允许一炮多响和一杠多抢。',
            '点炮／自摸每份：平胡 3／2，一条龙、清一色、七对 9／9，龙七对 18／18。'
            '自摸其他每家付，点炮或抢杠由供牌者付。',
            '供牌者已报听时，非赢家各付点炮应付款的三分之一，赢家之间不转账。'
            '直杠由点杠者付 3，若点杠者已报听则其他每家付 1。补杠其他每家付 1，暗杠其他每家付 2，流局返杠。',
            '你首局坐庄。单胡赢家或多响供牌者接庄，流局由最后摸牌者接庄。',
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

VARIANTS[MACAU] = {
    'id': MACAU,
    'family_id': 'honors-and-chows',
    'family': 'Honors and chows',
    'name': 'Macau Tournament Mahjong',
    'aliases': ['Macau', 'Macao', '澳门', '澳門', '澳门竞技', '澳大校友会'],
    'rule_version': 'macau-alumni-2011-dojo-v1-20261118',
    'subtitle': 'Macau · No minimum faan · Four winds',
    'badge': 'Macau · 136 tiles',
    'description': 'A sixteen-hand match based on the Macau University alumni tournament rules.',
    'rules': [
        '136 tiles and no flowers. Four sets and a pair or Thirteen Orphans can win. '
        'Chow only from the preceding seat; wins beat pong and kong, which beat chow.',
        'There is no minimum faan. A zero-faan discard win pays 5. At 1–6 faan, a discard win pays '
        '20 per faan; for self-draw each opponent pays 10 per faan. Two special tiers use separate fixed payments.',
        'Robbing an added kong pays as a discard win. Only Thirteen Orphans can rob a concealed kong. '
        'A qualifying self-draw may charge all three shares to the player who supplied the decisive meld.',
        'Keep 14 tiles in reserve. No kong on the last live draw. The last discard can only be claimed to win.',
        'Four hands in each of the East, South, West and North rounds. The dealer advances after every hand, '
        'and the match ends after 16 hands. The highest cumulative score wins.',
    ],
    'sources': [{'title': 'University of Macau Alumni Association tournament rules (PDF)',
                 'url': 'https://www.aaaum.org.mo/wp-content/uploads/2011/06/mahjong_rules_2011.pdf'}],
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'translations': {'zh-CN': {
        'name': '澳门竞技麻将', 'family': '字牌与吃牌家族',
        'subtitle': '澳门 · 零番可胡 · 四圈轮庄', 'badge': '澳门 · 136 张',
        'description': '以澳门大学校友会 2011 年赛例为依据，完整打 16 局。',
        'rules': [
            '136 张，无花。四组一对或十三幺可胡。只吃上家，胡优先于碰杠，碰杠优先于吃。',
            '不限起胡番数。零番出铳付 5 分。普通 1–6 番，出铳每番 20，自摸其他每家每番 10。A、B 类奇牌另定分。',
            '抢补杠按出铳算；只有十三幺可抢暗杠。符合清一色十二章、大三元、大四喜或四杠的包牌条件时，由供牌者包自摸。',
            '留 14 张王牌。最后一张摸牌不能杠，最后一张弃牌只能胡。',
            '东南西北各四局，每局必轮庄。16 局打完结束，累计得分最高者获胜。',
        ],
    }},
}

_HORSE_ROOMS = {
    'family_id': 'honors-horse-tiles',
    'family': 'Honors and horse tiles',
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
}
VARIANTS[SHANGSI.VARIANT] = {
    **_HORSE_ROOMS,
    'id': SHANGSI.VARIANT,
    'name': 'Shangsi Mahjong',
    'aliases': ['Shangsi', 'Guangxi', '上思', '上思麻将', '广西'],
    'rule_version': 'shangsi-discard-4-10-dojo-v1',
    'subtitle': 'Guangxi · Reserve 10 tiles · Seat-based horse tiles',
    'badge': 'Shangsi · 136 tiles',
    'description': 'A no-chow Guangxi room with seat-based horses, opening discard payments and kong refunds on a draw.',
    'rules': [
        '136 tiles with honors. Pon and kong are allowed; no chow. Discard, self-draw and added-kong robbery can win. '
        'Standard hands, seven pairs and thirteen orphans use the same base schedule.',
        'Discard and robbery winners receive 1 from the provider; self-draw receives 2 from each opponent. '
        'Reserve ten tiles. Reveal four on a discard or robbery, ten on a self-draw. '
        'Each winner uses the same revealed tiles and multiplies their base by one plus their own seat hits.',
        'The dealer matches 1/5/9, East and Red; the next seat 2/6, South and Green; the opposite seat 3/7, West and White; '
        'the last seat 4/8 and North.',
        'If the first four uninterrupted discards all match, the dealer pays 1 to each opponent. '
        'Passing an ordinary win or pon locks that tile until the next draw; a higher-valued win can bypass the win lock.',
        'Each opponent pays 1 for direct or added kong and 2 for concealed kong. Kong points are refunded on a draw. '
        'You deal first. A sole winner, or the provider on multiple wins, deals next. The dealer stays on a draw.',
    ],
    'sources': [{'title': 'Shangsi operator rules', 'url': 'https://www.hzbianqu.com/game/1091.html'}],
    'translations': {'zh-CN': {
        'name': '上思麻将', 'family': '字牌奖码家族', 'subtitle': '广西 · 留 10 张 · 按庄位中码',
        'badge': '上思 · 136 张', 'description': '无吃，按庄位中码，支持抄庄，流局退杠。',
        'rules': [
            '136 张带字牌，可碰杠，不可吃，可点炮、自摸和抢补杠。标准型、七对、十三幺同底分。',
            '点炮或抢杠由供牌者付 1，自摸其他每家付 2。留 10 张，点炮或抢杠翻 4 张、自摸翻 10 张。'
            '多响共用翻开的码牌，按各自位置命中，胡分乘以中码数加一。',
            '庄家中 1/5/9、东、中；下家中 2/6、南、发；对家中 3/7、西、白；上家中 4/8、北。',
            '开局未碰杠且首圈四家打同一张，庄家各付另外三家 1 分。过普通胡或碰锁同张到下次摸牌，'
            '加分的胡牌可以越过过胡限制。',
            '其他每家付直杠或补杠 1、暗杠 2。流局退杠。首局你坐庄，单胡赢家或多响供牌者接庄，流局连庄。',
        ],
    }},
}

VARIANTS[HUNDRED.VARIANT] = {
    **_HORSE_ROOMS,
    'id': HUNDRED.VARIANT,
    'name': 'Hundred-Tile Mahjong',
    'aliases': ['100 tiles', 'Hundred tiles', '100张', '100张麻将', '百张麻将', '一百张麻将'],
    'rule_version': 'hundred-tiles-4-horses-dojo-v1',
    'subtitle': 'Dots, bamboo and honors · Self-draw · Four horse tiles',
    'badge': '100 tiles',
    'description': 'Remove the character suit. Win by self-draw or by robbing an added kong and reveal seat-based horse tiles.',
    'rules': [
        '100 tiles: dots, bamboo, four winds and three dragons. No chow or ordinary discard wins. '
        'Rob an added kong, including multiple winners. A concealed kong shows one face.',
        'Self-draw base: regular 2, all triplets 4, mixed one suit 4, pure one suit 8, seven pairs 8. '
        'Combine triplets with a flush or seven pairs with a pure flush. Terminals and honors 12 and all honors 16 stand alone.',
        'Keep four tiles in reserve and reveal them after a win. Each win share is multiplied by one plus the '
        'number of matches for that winner’s seat relative to the dealer. The robbed player covers three shares per winner.',
        'Direct kong: the provider pays 3. Added kong: each opponent pays 1. Concealed kong: each pays 2. '
        'Keep kong points on a draw. This room uses no passed-win or passed-pon lock.',
        'You deal first. A sole winner, or the robbed player after multiple wins, deals next. The dealer stays on a draw.',
    ],
    'sources': [{'title': 'Hundred-Tile operator rules', 'url': 'https://www.hzbianqu.com/game/1086.html'}],
    'translations': {'zh-CN': {
        'name': '100 张麻将', 'family': '字牌奖码家族', 'subtitle': '筒条加字牌 · 自摸／抢杠 · 四码',
        'badge': '100 张', 'description': '去除万子。自摸或抢补杠后，按庄位翻码结算。',
        'rules': [
            '100 张，包含筒、条、四风及三元牌。不可吃，不可普通点炮胡，可抢补杠及一杠多抢。暗杠亮一张。',
            '自摸每份：平胡 2、对对胡 4、混一色 4、清一色 8、七对 8。对对胡可叠加清一色或混一色，'
            '七对可叠加清一色。幺九 12、全字牌 16，各自单列。',
            '留四张，胡后翻开，按与庄家的相对座位判码，每份乘以中码数加一。抢杠由杠家向每位赢家包三份。',
            '直杠供牌者付 3，补杠每家付 1，暗杠每家付 2。流局保留杠分。本桌无过碰、过胡。',
            '首局你坐庄，单胡赢家或多抢时被抢杠者接庄，流局连庄。',
        ],
    }},
}

VARIANTS[HEZHOU.VARIANT] = {
    **_HORSE_ROOMS,
    'id': HEZHOU.VARIANT,
    'name': 'Hezhou Mahjong',
    'aliases': ['Hezhou', 'Guangxi', '贺州', '贺州麻将', '广西'],
    'rule_version': 'hezhou-136-four-yao-horses-dojo-v1',
    'subtitle': 'Guangxi · Chows enabled · Three-feeds liability',
    'badge': 'Hezhou · 136 tiles',
    'description': 'A four-seat table with chows, four fixed horse tiles and reciprocal liability after three feeds.',
    'rules': [
        '136 tiles, no jokers. Chow from the previous seat; pon and kong are open to the other players. '
        'Discard and added-kong robbery allow multiple winners. Pass a win or pon and that action is blocked '
        'until your next draw. Self-draw is still allowed.',
        'Regular win starts at 1. All triplets, pure suit, seven pairs, closed self-draw and replacement self-draw '
        'each double it. Dragon seven pairs ×4 replaces ordinary seven pairs; thirteen orphans uses 10. '
        'Robbing a kong or winning on a discard after kong adds 1.',
        'Reserve and reveal four horse tiles after a win. Ranks 1/5/9, East and Red count for each winner. '
        'Multiply the patterned share by one plus the hits.',
        'Three calls sourced from one opponent make the two seats liable for each other’s discard wins and self-draws. '
        'A third seat’s discard only bills that seat. The latest qualifying partner covers a self-draw.',
        'Direct kong: the provider pays 1. Added and concealed kong: each opponent pays 1. Keep kong points on a draw. '
        'You deal first. A sole winner or the multi-win provider deals next. The dealer stays on a draw.',
    ],
    'sources': [{'title': 'Hezhou operator rules', 'url': 'https://www.hzbianqu.com/game/1092.html'}],
    'translations': {'zh-CN': {
        'name': '贺州麻将', 'family': '字牌奖码家族', 'subtitle': '广西 · 可吃 · 三笔互赔',
        'badge': '贺州 · 136 张', 'description': '四人无鬼桌，上家可吃，翻四张幺码，三笔后两席互包。',
        'rules': [
            '136 张，无鬼。只吃上家，可碰杠。点炮及抢补杠允许多响。过胡或过碰后，到下次摸牌才可再胡或再碰，自摸不受限。',
            '平胡 1，碰碰胡、清一色、七对、门清自摸、杠开各翻倍。龙七对 ×4 取代普通七对，'
            '十三幺单列 10。抢杠或杠后炮加 1。',
            '留四张，胡后翻幺码。1/5/9、东、红中为中码，每份按中码数加一乘算。',
            '同一对手提供三笔副露，两家互相包点炮和自摸。其他家点炮仍由该供牌者付，有多组关系时取最后成立的一笔包自摸。',
            '直杠供牌者付 1，补杠和暗杠每家付 1，流局保留杠分。你先坐庄，单胡赢家或多响供牌者接庄，流局连庄。',
        ],
    }},
}

VARIANTS[LIUZHOU.VARIANT] = {
    **_HORSE_ROOMS,
    'id': LIUZHOU.VARIANT,
    'name': 'Liuzhou Exploding Fish Mahjong',
    'aliases': ['Liuzhou', 'Exploding Fish', '柳州', '柳州麻将', '爆炸鱼', '天天爱柳州'],
    'rule_version': 'liuzhou-exploding-fish-dojo-v1',
    'subtitle': 'Guangxi · One fish tile · Three-feeds payments',
    'badge': 'Liuzhou · 136 tiles',
    'description': 'A Liuzhou room with one reveal worth the tile rank or five for an honor. Three-feeds payments apply.',
    'rules': [
        '136 tiles. Chow from the previous seat, but do not immediately discard the claimed tile. '
        'Pon, kong, seven pairs, multiple discard wins and added-kong robbery are allowed.',
        'Regular wins pay 1 per liable seat, all triplets, pure suit or seven pairs pay 3 on discards or 2 on self-draw. '
        'Extra big patterns double the payout. A closed self-draw doubles the share.',
        'An added-kong robbery, a win on the discard after a kong or a four-exposed-meld discard win pays 3 before '
        'big-pattern doubles. Replacement self-draw starts at 2 from each opponent, or 6 from the direct-kong supplier alone.',
        'Reveal one reserved fish tile after any win. It counts its rank, or five for any honor. Multiply the win share '
        'by one plus that number. Kong points are separate.',
        'After three exposed calls from the same supplier, the two seats become partners. A winner’s partner '
        'pays an extra share on any discard win, or three shares total on an ordinary self-draw. Other opponents '
        'still pay on self-draw, except when the direct-kong supplier covers the win.',
        'Direct or added kong pays 1 from the original supplier, concealed kong 1 from each opponent. '
        'Passing a win locks offered wins, and passing a pon locks its tile until the next draw.',
        'You deal first. The first winner in turn order takes the next deal, except three winners return the deal '
        'to the provider. The dealer stays on a draw.',
    ],
    'sources': [{'title': 'Liuzhou operator rules', 'url': 'https://www.hzbianqu.com/game/908.html'}],
    'translations': {'zh-CN': {
        'name': '柳州麻将（爆炸鱼）', 'family': '字牌奖码家族', 'subtitle': '广西 · 一张爆炸鱼 · 三比赔付',
        'badge': '柳州 · 136 张', 'description': '胡后钓一张爆炸鱼，数牌按点数，字牌按五条，三比另行赔付。',
        'rules': [
            '136 张，可吃上家，吃后不能打吃进的同张。可碰杠，可胡七对、一炮多响和抢补杠。',
            '普通胡每份 1，清一色、碰碰胡、七小对点炮 3／自摸每家 2。多个大牌型继续翻倍。门清自摸再翻倍。',
            '抢杠、杠后炮、全求人点炮从 3 起算，每个大胡翻倍。杠开从自摸每家 2 起算，直杠杠开由供牌者从 6 起包付。',
            '胡后翻一张爆炸鱼。数牌按点数、字牌算五条，胡分乘以鱼数加一，杠分另算。',
            '同一对手供三副露后两家互为三比。赢家的三比席在点炮胡时另付一份，普通自摸时付三份，其他输家仍各付一份。'
            '直杠杠开由供牌者包三。',
            '直杠或补杠由最初供牌者付 1，暗杠其他每家付 1。过胡锁接炮，过碰锁同张，自己再摸时解锁。',
            '首局你坐庄，单胡或双响由顺位最先的赢家接庄，三响供牌者接庄，流局连庄。',
        ],
    }},
}

VARIANTS[GUANGZHOU] = {
    'id': GUANGZHOU,
    'family_id': 'honors-and-chows',
    'family': 'Honors and chows',
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Guangzhou Three-Faan Mahjong',
    'aliases': ['Guangzhou', 'Canton', 'Three Faan', '广州', '廣州', '三番封顶', '爆胡'],
    'rule_version': 'guangzhou-three-faan-dojo-v1',
    'subtitle': 'Guangzhou · Three-faan ordinary cap · First-supplier liability',
    'badge': 'Guangzhou · 136 tiles',
    'description': 'An honors-and-chows table with ordinary hands capped at three faan and separate big hands.',
    'rules': [
        '136 tiles, chow, pon and kong. Four sets and one pair, or thirteen orphans, may win. '
        'Passing a win blocks that tile until a successful call or your draw.',
        'No minimum. Chicken hand 0 faan, all sequences 1, all triplets and half flush 2. '
        'This room adds pure suit and all honors at 3, and caps the ordinary total at 3.',
        'Opening wins, thirteen orphans and big three dragons are 6; big four winds is 8. '
        'Kong replacement self-draw, last draw, robbery and a three-way discard win score at least 3.',
        'One share is 2 to the power of the faan. Each opponent pays one share for self-draw; a discarder '
        'pays each winner. A robbed kong maker covers three shares per winner. Only thirteen orphans may rob a concealed kong.',
        'The direct-kong supplier covers its replacement win. The first applicable supplier of a fourth open meld, '
        'third dragon pung or fourth wind pung may cover the self-draw. A supplier can carry only their first twelve-open obligation.',
        'No separate kong transfers in this room. Deal starts at a random seat. The dealer stays when among the winners '
        'and otherwise passes to the next seat.',
    ],
    'sources': [
        {'title': 'Guangzhou special rules', 'url': 'https://www.hzbianqu.com/game/164.html'},
        {'title': 'Bianfeng three-faan base table', 'url': 'https://www.hzbianqu.com/game/489.html'},
    ],
    'translations': {'zh-CN': {
        'name': '广州麻将（三番封顶）', 'family': '字牌与吃牌家族', 'subtitle': '广州 · 普通三番封顶 · 包先不包后',
        'badge': '广州 · 136 张', 'description': '普通牌型三番爆胡，特殊大牌单列，含十二张落地包牌。',
        'rules': [
            '136 张，可吃碰杠。四组一对或十三幺可胡。同张过胡到自己摸牌或成功吃碰杠后解除。',
            '鸡胡 0 番、平胡 1 番、碰碰胡及混一色 2 番。本桌清一色、字一色各按 3，普通牌型叠加后最多 3 番。',
            '天和、地和、人和、十三幺及大三元 6 番，大四喜 8 番。杠开、海底自摸、抢杠或一炮三响至少 3 番。',
            '每份为 2 的番数次方。自摸其他每家付一份，点炮供牌者向每个赢家付一份，抢杠由杠家向每个赢家包三份。'
            '只有十三幺能抢暗杠。',
            '直杠杠开优先由供杠者包。否则十二张落地、第三副三元刻或第四副风刻按最早成立的适用责任包自摸。'
            '同一供牌者只包最早促成的十二张落地。',
            '本桌不另算杠分。首局随机庄，庄家胡则连庄，否则轮庄。',
        ],
    }},
}

VARIANTS[GOLDEN_GUANGDONG] = {
    **_HORSE_ROOMS,
    'id': GOLDEN_GUANGDONG,
    'name': 'Golden Island Guangdong Mahjong',
    'aliases': ['Golden Island', 'Guangdong', '黄金岛', '广东麻将', '换三张', '雙邊馬'],
    'rule_version': 'golden-island-guangdong-dojo-v1',
    'subtitle': 'Guangdong · Exchange three · Bilateral horse payments',
    'badge': 'Guangdong · 136 tiles',
    'description': 'Start by exchanging three tiles. Each winner shares a four-tile horse reveal and scores separately against each payer.',
    'rules': [
        '136 tiles, no chow. The dealer receives 14 tiles. Choose any three tiles and pass them to the next seat. '
        'All four exchanges resolve together, then the dealer plays.',
        'Pon, kong, multiple discard wins and added-kong robbery are allowed. Only thirteen orphans may rob a concealed kong. '
        'Chicken and all-sequence-only hands win by self-draw or robbery. Passing a win locks equal or lower offered scores until a draw or a successful call.',
        'Add compatible hand patterns, including pure suit 12, all triplets 6, thirteen orphans 20 and seven pairs 8/16/24/36. '
        'Opening win adds 20. A late self-draw, kong replacement self-draw, or robbery adds 1.',
        'Reserve four horse tiles and disallow meld calls and kongs during the last four live draws. After a win, '
        'reveal the reserve and count seat hits from each winner. Each payment is multiplied by one plus '
        'the hits for that winner and that payer combined.',
        'Direct-kong replacement or four-exposed-set self-draw may assign all three payments to the supplier. '
        'Direct kong pays 3 from its supplier, added or concealed kong 1 or 2 from each opponent. '
        'Draws refund the kong transfers. Matching the first four uninterrupted discards costs the dealer 1 per opponent.',
    ],
    'sources': [{'title': 'Golden Island Guangdong rules and payout example',
                 'url': 'https://www.80166.com/guide/rule/gdmj.html'}],
    'translations': {'zh-CN': {
        'name': '黄金岛广东麻将', 'family': '字牌奖码家族', 'subtitle': '广东 · 换三张 · 双边中马',
        'badge': '广东 · 136 张', 'description': '开局换三张。四马按每个赢家及对应付款人的中马数结算。',
        'rules': [
            '136 张，无吃。庄家起手 14 张。每家选任意三张给下家，四家一次交换，随后庄家行动。',
            '可碰杠、一炮多响、抢补杠。暗杠只可由十三幺抢。鸡胡、纯平胡只可自摸或抢杠。'
            '过胡后至自己摸牌或成功碰杠，只接更高分的牌。',
            '能叠加的牌型相加。清一色 12，碰碰胡 6，十三幺 20，七对各档 8／16／24／36。'
            '开局和加 20，末圈自摸、杠开或抢杠加 1。',
            '留四张翻马，最后四张活牌不可碰杠。胡后每个赢家从自己起算马位；每笔付款乘以赢家和该付款人中马数之和加一。',
            '直杠杠开或十二张落地自摸由供牌者包三笔。直杠供牌者付 3，补杠或暗杠其他每家付 1 或 2，流局退杠。'
            '首圈无人叫牌且四张首打相同，庄家给其他三家各 1 分。',
        ],
    }},
}

VARIANTS[GUILIU] = {
    **_HORSE_ROOMS,
    'id': GUILIU,
    'name': 'Guiliu Wake-Tile Mahjong',
    'aliases': ['Guiliu', 'Wake tiles', '桂柳', '桂柳麻将', '天天爱桂柳', '上下醒'],
    'rule_version': 'guiliu-wake-dojo-v1',
    'subtitle': 'Guangxi · Repeat wake reveals · Reciprocal liability',
    'badge': 'Guiliu · 136 tiles',
    'description': 'Match adjacent same-suit tiles, reveal again after three hits, and settle reciprocal payments after three feeds.',
    'rules': [
        '136 tiles. Pon and kong, with no chow. Standard hands, seven pairs and distinct ragged hands may win. '
        'An offered win must exceed the last passed win until a draw or a successful call. Multiple winners are allowed.',
        'Each winner uses the shared reveal starting from the first indicator. Count same-suit adjacent ranks and '
        'cycle within winds or dragons. Count concealed tiles and melds. Three hits on one indicator allow another reveal. '
        'For a last-tile win, use the winning tile once as the indicator.',
        'Regular and ragged hands use the dealer/non-dealer table. Bigger patterns start at 15 per opponent '
        'on a self-draw or 35 on a discard, multiplied by 2 to the power of total fan minus one. '
        'Wake hits add one per self-draw payer or three to the discard share before event multipliers.',
        'After three exposed feeds from a seat, that pair has reciprocal liability. Each partner covers double '
        'the entire self-draw bill. On discard wins, each partner adds one share, except equal-scoring co-winners '
        'do not pay each other.',
        'A direct or added kong bills its original supplier 3; each opponent pays 2 for a concealed kong. '
        'An uninterrupted four-way opening discard match is settled at a win, when the first winner pays each other seat 2.',
    ],
    'sources': [{'title': 'Guiliu operator rules', 'url': 'https://www.hzbianqu.com/game/1121.html'}],
    'translations': {'zh-CN': {
        'name': '桂柳麻将（上下醒）', 'family': '字牌奖码家族', 'subtitle': '广西 · 可翻重醒 · 三笔互包',
        'badge': '桂柳 · 136 张', 'description': '同花色数牌和字牌相邻中醒，单张指示中三张可继续翻，三笔互包。',
        'rules': [
            '136 张，无吃，可碰杠，可胡标准手牌、七对、烂牌。过胡只接更高分，自己摸牌或成功碰杠后解除。允许多响。',
            '共用指示牌序列。数牌按同花色上下相邻，风牌和三元牌分别循环。手牌和副露都算，当张中三张可翻重醒。'
            '海底胡用胡牌本身作醒，只看一张。',
            '平胡及烂牌按庄闲表。大牌自摸每家从 15、点炮从 35 起，乘以 2 的总番减一次方。'
            '每中一醒自摸各加 1、点炮加 3，再计门清及杠后翻倍。',
            '同一供牌者三笔后互包，每个关联席对自摸分别包全份再翻倍。点炮每个关联席加付一份，同分的多响赢家互免。',
            '直杠或补杠由原供牌者付 3，暗杠其他每家付 2。首圈四张同牌，胡牌结算时排第一的赢家另付其他每家 2 分。',
        ],
    }},
}

_BLOOD_ROOMS = {
    'family_id': 'sichuan', 'family': 'Sichuan blood battle',
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
}

VARIANTS[SICHUAN_EXCHANGE] = {
    **_BLOOD_ROOMS,
    'id': SICHUAN_EXCHANGE,
    'name': 'Sichuan Xuezhan · Exchange Three',
    'aliases': ['Sichuan exchange', '四川换三张', '川麻换三张'],
    'rule_version': 'sichuan-exchange-dojo-v1',
    'subtitle': 'Same-suit exchange · Void suit · Four-fan cap',
    'badge': 'Sichuan · 108 tiles',
    'description': 'Exchange three tiles of one suit, then choose a void suit and play until three seats win or the wall runs out.',
    'rules': [
        'The dealer starts with 14 tiles. Everyone selects three of one suit in secret. The table reveals '
        'one random direction and swaps simultaneously, then everyone declares a void suit.',
        'No chow. Non-void pons and kongs are allowed even while holding void tiles. '
        'Discard from the void suit first. The last draw allows pon but no kong.',
        'A passed win locks equal and lower fan until a draw or successful call. '
        'Robbing a kong and winning on a discard after kong can override this lock.',
        'Room scoring uses one base point, four fan maximum, self-draw +1 fan and closed-hand/middle-tiles bonuses. '
        'After choosing pon over a direct kong, a later added kong earns no kong points.',
        'At wall exhaustion, unready seats pay ready seats their best hand value and refund kong earnings. '
        'The dealer repeats after wall exhaustion. Otherwise the first winner deals, or the supplier of a first multi-win.',
    ],
    'sources': [{'title': 'Sichuan Xuezhan operator rules', 'url': 'https://www.hzbianqu.com/game/1100.html'}],
    'translations': {'zh-CN': {
        'name': '四川血战到底（换三张）', 'family': '四川血战', 'subtitle': '同门换三张 · 定缺 · 四番封顶',
        'badge': '四川 · 108 张', 'description': '先换三张同门牌，再定缺。打到三家胡牌或牌墙耗尽为止。',
        'rules': [
            '庄家起手 14 张。各家暗选三张同门牌，四家确认后随机顺时针、逆时针或对家交换，然后定缺。',
            '不可吃。定缺牌先打，未打完时仍可碰杠非缺门牌。牌墙空后可碰，不可杠。',
            '过胡后到自己摸牌或成功碰杠前只接更高番；抢杠胡、杠上炮可例外。',
            '本房间一分起、四番封顶，自摸加一番，计门清与中张。可直杠却选择碰的牌，之后可补杠但不计杠钱。',
            '牌墙耗尽时查叫、退未听家杠分，原庄连庄。三家胡牌时首胡者坐下一庄，首胡为多响则点炮者坐庄。',
        ],
    }},
}

VARIANTS[YUNNAN] = {
    **_BLOOD_ROOMS,
    'id': YUNNAN,
    'name': 'Yunnan Xuezhan · Flower Pig',
    'aliases': ['Yunnan Mahjong', 'Flower Pig', '云南血战到底', '云南麻将', '查花猪'],
    'rule_version': 'yunnan-xuezhan-dojo-v1',
    'subtitle': 'Void suit · Passed-pon lock · Flower-pig payments',
    'badge': 'Yunnan · 108 tiles',
    'description': 'Play through three winners or settle a draw with both ready-hand and flower-pig payments.',
    'rules': [
        'Choose a void suit and discard it first. No chow. Non-void pons and kongs are allowed, including added-kong robbery.',
        'Passed wins must increase in fan and a passed pon locks the same tile until the next draw or successful call. '
        'A discard win shows the claimed tile only. A self-draw winner keeps their tiles concealed.',
        'Room scoring uses one base point and a 16-point cap per share. No separate self-draw fan. Types include '
        'seven pairs through triple-dragon pairs, golden hook, roots and four kongs.',
        'At wall exhaustion with fewer than three winners, any seat still holding its void suit pays all '
        'non-flower-pig seats 16 apiece, including seats that already won. Unready seats also pay each ready '
        'seat its best hand value and refund earnings from kongs.',
        'The dealer repeats after wall exhaustion. Otherwise the first winner deals, or the supplier of a first multi-win.',
    ],
    'sources': [{'title': 'Yunnan Xuezhan operator rules', 'url': 'https://www.hzbianqu.com/game/1108.html'}],
    'translations': {'zh-CN': {
        'name': '云南血战到底（查花猪）', 'family': '四川血战', 'subtitle': '定缺 · 过手碰 · 查花猪',
        'badge': '云南 · 108 张', 'description': '三家胡牌即结束，牌墙耗尽时进行查大叫、查花猪和退税。',
        'rules': [
            '开局定缺，缺门牌先打。不可吃，非缺门牌可碰杠，允许抢补杠。',
            '过胡后只接更高番；过碰后同张暂不能碰，摸牌或成功碰杠后解除。点炮只亮胡张，自摸不亮牌。',
            '本房间一分起、每份 16 分封顶。自摸不单独加番。计七对至三龙七对、金钩钩、根和十八罗汉。',
            '牌墙耗尽且不足三家胡牌时，手里还留缺门牌者各付非花猪玩家 16 分，已胡家也照收。'
            '未听牌者另向听牌的未胡者支付最高牌型分，并退回杠收益。',
            '牌墙耗尽原庄连庄。三家胡牌时首胡者坐下一庄，首胡为多响则点炮者坐庄。',
        ],
    }},
}

VARIANTS[KAXINWU] = {
    'id': KAXINWU,
    'family_id': 'continuous-win', 'family': 'Continuous-win family',
    'name': 'Kaxinwu · Continuous Wins',
    'aliases': ['Blood Battle Kaxinwu', 'Kaxinwu', '血战卡心五', '卡心五', '两门血流'],
    'rule_version': 'kaxinwu-72-dojo-v1',
    'subtitle': '72 tiles · Repeated wins · Locked hand after the first win',
    'badge': 'Continuous wins · 72 tiles',
    'description': 'All four seats keep playing after a win. Win repeatedly and settle all charges once the wall is exhausted.',
    'rules': [
        'Dots and bamboo only. No chow and no void suit. Pon, kong, self-draw, discard wins, and added-kong robbery are allowed. '
        'One offered tile can pay several winners.',
        'After your first win, discard only the tile you draw and do not pon. A kong must leave every tile you already won on '
        'as a legal wait. An eligible open kong is mandatory unless you or another seat wins on that tile.',
        'The discarder or robbed kong maker pays each winner. Each other seat pays for self-draw. A single win advances normally. '
        'After a multi-win, the supplier draws again. The last seat to draw deals next.',
        'A basic hand has 1 fan, a middle-five wait or all triplets 2, and seven pairs 2–5 by concealed quads. '
        'Pure suit adds 1. Each fan doubles the payment. Kong points are immediate and doubled for each subsequent kong in a chain.',
        'On wall exhaustion, unready seats pay the maximum eligible hand value of each ready seat, or the best shape previously '
        'won by each winning seat. Pure suit counts for this check; roots and kong-event bonuses do not.',
    ],
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'sources': [{'title': 'Kaxinwu operator rules', 'url': 'https://www.hzbianqu.com/game/1122.html'}],
    'translations': {'zh-CN': {
        'name': '血战卡心五（两门血流）', 'family': '血流多胡家族', 'subtitle': '72 张 · 一局多次胡 · 胡后摸切',
        'badge': '血流 · 72 张', 'description': '哪怕已经胡牌，四家也都继续打，可以反复胡，直到牌墙耗尽统一结算。',
        'rules': [
            '只用筒、条两门。无吃、无定缺。可碰杠，自摸、接炮、抢补杠；同一张牌可以多响。',
            '首次胡牌后只可摸切，不可碰。开杠后须保留所有已胡过的牌作听口；符合条件的直杠必须执行，胡牌优先。',
            '点炮者或被抢杠者逐家付，自摸其他三家付。单响按顺序继续摸；多响由供牌者再摸，末摸者坐下一庄。',
            '平胡一番；卡心五、对对胡两番；七对按四同张组数为二至五番。清一色加一番，每番翻倍。连杠自第二杠起每杠的杠分翻倍。',
            '牌墙耗尽时查叫。未听者给听牌席付最高可能牌型分，给已胡席付已胡过的最高牌型分，均计清一色，不计鬼和杠后额外番。',
        ],
    }},
}

_KA_WU_XING = {
    'family_id': 'ka-wu-xing', 'family': 'Three-player Ka Wu Xing', 'seat_count': 3,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
}
_KA_WU_XING_RULES = [
    'Three players, 84 tiles: dots, bamboo and three dragons. Pon and kong are allowed without chow. '
    'A completed ordinary hand or seven pairs may win from a discard or self-draw.',
    'After a tenpai discard, you may reveal your hand. Open ready doubles the hand and locks play to drawn-tile '
    'discards or wins. In a draw, the first seat to reveal pays each opponent one point. Kong points still stand.',
    'Middle-five wait and pure suit multiply by four. All triplets double; dragon patterns, four-in-one and '
    'seven-pair variants also score. An open kong charges its supplier two; an added or closed kong charges '
    'each opponent one or two. The last wall draw allows no kongs.',
]
_KA_WU_XING_RULES_ZH = [
    '三人 84 张，筒条加中发白。可碰杠、不可吃，标准牌型或七对可接炮、自摸。',
    '打到听牌时可选择亮倒，公开手牌并加倍；此后只可摸切或胡。荒庄首亮者付其他两家各一分，杠分保留。',
    '卡五星和清一色四倍，碰碰胡两倍；三元、四归一及不同七对也计分。直杠供牌者付两分，'
    '补杠与暗杠由其他两家各付一分或两分。末张不可杠。',
]

VARIANTS[SUIZHOU.VARIANT] = {
    **_KA_WU_XING,
    'id': SUIZHOU.VARIANT,
    'name': 'Suizhou Ka Wu Xing',
    'aliases': ['随州卡五星', '随州麻将', '拳打脚踢', 'Suizhou'],
    'rule_version': 'suizhou-open-ready-dojo-v1',
    'subtitle': 'Three seats · Open ready · Double discard wins',
    'badge': 'Hubei · 84 tiles',
    'description': 'Reveal a ready hand for a bigger win. The discarder pays both shares if you win on their tile.',
    'rules': [*_KA_WU_XING_RULES,
              'A discard or robbed kong can produce two winners; their supplier deals next. '
              'Robbing an added or a concealed kong is allowed. A passed offer blocks equal or lower scores until your draw.',
              'Win patterns multiply without a cap. A discard win by an open-ready player charges the supplier twice '
              'the winning share, even if the supplier has also declared. This room keeps the dealer on a draw.'],
    'sources': [{'title': 'Suizhou operator rules', 'url': 'https://www.hzbianqu.com/game/1249.html'},
                {'title': 'Local Ka Wu Xing rules', 'url': 'https://kawuxingmajiang.zocet.com/'}],
    'translations': {'zh-CN': {
        'name': '随州卡五星', 'family': '三人卡五星', 'subtitle': '三人 · 亮倒 · 一炮双响',
        'badge': '湖北 · 84 张', 'description': '亮倒后接炮，由点炮者付两家自摸份。支持一炮双响和升番破漏胡。',
        'rules': [*_KA_WU_XING_RULES_ZH,
                  '可抢补杠和暗杠，允许一炮双响，双响由供牌者接庄。漏胡后到自己摸牌前只接更高倍。',
                  '牌型倍数相乘，不封顶。点炮给亮倒玩家时由点炮者包两份，点炮者已亮倒也照包。荒庄原庄连庄。'],
    }},
}

VARIANTS[XIANGYANG.VARIANT] = {
    **_KA_WU_XING,
    'id': XIANGYANG.VARIANT,
    'name': 'Xiangyang Ka Wu Xing',
    'aliases': ['襄阳卡五星', '襄阳麻将', 'Xiangyang'],
    'rule_version': 'xiangyang-cap-eight-dojo-v1',
    'subtitle': 'Three seats · Open ready · Eightfold cap',
    'badge': 'Hubei · 84 tiles',
    'description': 'This table caps hand multipliers at eight and waives the extra discard share when both seats have declared.',
    'rules': [*_KA_WU_XING_RULES,
              'This room takes the nearest discard winner and allows robbery of an added kong. Passing blocks all further '
              'offered wins until your draw. The winner deals next; after a draw the dealer repeats.',
              'Hand patterns multiply up to eight. The discarder covers two shares for an open-ready winner, or '
              'one share when both seats are open ready. Apply liability after the hand cap.'],
    'sources': [{'title': 'Xiangyang operator rules', 'url': 'https://www.hzbianqu.com/game/1205.html'}],
    'translations': {'zh-CN': {
        'name': '襄阳卡五星', 'family': '三人卡五星', 'subtitle': '三人 · 亮倒 · 八倍封顶',
        'badge': '湖北 · 84 张', 'description': '牌型八倍封顶。亮倒者接炮可包赔，双方都亮倒时只付一份。',
        'rules': [*_KA_WU_XING_RULES_ZH,
                  '本房间取座次最近者胡，允许抢补杠。漏胡后到自己摸牌前任何牌都不能接。胡家接庄，荒庄连庄。',
                  '牌型相乘八倍封顶，包赔在封顶后算。点炮给亮倒者包两份，双方都亮倒则付一份。'],
    }},
}

VARIANTS[SHIYAN.VARIANT] = {
    **_KA_WU_XING,
    'id': SHIYAN.VARIANT,
    'name': 'Shiyan Ka Wu Xing',
    'aliases': ['十堰卡五星', '十堰麻将', 'Shiyan', '平胡自摸'],
    'rule_version': 'shiyan-self-draw-plain-dojo-v1',
    'subtitle': 'Three seats · Plain self-draw · Pattern discard wins',
    'badge': 'Hubei · 84 tiles',
    'description': 'Win a regular hand by self-draw, or score a pattern to win from a discard. One discard can pay both other seats.',
    'rules': [
        'Three players, 84 tiles: dots, bamboo and three dragons. Pon and kong are allowed without chow.',
        'A plain hand pays one point per share and wins by self-draw only. Middle-five, trips and other supported '
        'patterns can win on a discard. Middle-five means completing 4–5–6 on the winning five and adds one double.',
        'Robbing an added kong and wins on the immediate post-kong discard are doubled. This room plays without '
        'open ready. Pattern values multiply without a cap.',
        'Self-draw bills both opponents, and a discard or robbery bills the supplier for each winner. '
        'On two wins the supplier deals; otherwise the winner deals. After a draw the dealer repeats and kong points stand.',
    ],
    'sources': [{'title': 'Shiyan operator rules', 'url': 'https://m.tcy365.com/news/d30413.html'}],
    'translations': {'zh-CN': {
        'name': '十堰卡五星', 'family': '三人卡五星', 'subtitle': '三人 · 平胡自摸 · 有番可接炮',
        'badge': '湖北 · 84 张', 'description': '平胡只可自摸，有番牌可接炮，支持一炮双响。',
        'rules': [
            '三人 84 张，筒条加中发白。不吃，可碰杠。',
            '平胡一份一分，只可自摸。卡五星、碰碰胡等有番可接炮。卡五星按四六夹胡五，计两倍。',
            '可抢补杠；抢杠和杠后首张点炮各翻倍。本桌无亮倒，牌型相乘不封顶。',
            '自摸其他两家各付一份，点炮或被抢杠者向每位赢家付款。单胡者接庄，双响由供牌者接庄。荒庄连庄，杠分保留。',
        ],
    }},
}

VARIANTS[JJ_THREE] = {
    'id': JJ_THREE, 'family_id': 'honors-and-chows', 'family': 'Honors and chows', 'seat_count': 3,
    'name': 'JJ Three-Player · Call Points',
    'aliases': ['JJ三人麻将', '口口香', '吃香', '碰香', 'JJ Sanren'],
    'rule_version': 'jj-three-call-points-dojo-v1',
    'subtitle': 'Three seats · Chows and pons pay · Private ready declaration',
    'badge': 'JJ call points · 84 tiles',
    'description': 'A three-seat room based on the 2017 JJ score table, with immediate chow, pon, kong and ready payments.',
    'rules': [
        '84 tiles: characters, dots and three dragons. Chow from the previous seat; pon or kong from either other seat.',
        'An actual chow or pon pays one or two points from the tile supplier. Wins outrank other claims; '
        'kong and pon outrank chow. Call points are paid only after the response window resolves.',
        'A ready discard can declare tenpai without revealing your hand. Each opponent immediately pays two. '
        'The hand stays locked to drawn-tile discards or wins. A direct kong bills its supplier five; an added or '
        'concealed kong bills each opponent one or five.',
        'Win with a standard shape or seven pairs. Every win is worth five per payer. Both opponents pay on self-draw, '
        'and the supplier pays every winner on a discard or added-kong robbery. Draws keep all earned call points.',
    ],
    'default_policy': 'heuristic',
    'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'sources': [{'title': 'JJ 2017 three-player call-points table', 'url': 'https://www.jj.cn/help/new_74327.shtml'}],
    'translations': {'zh-CN': {
        'name': 'JJ 三人麻将（口口香）', 'family': '字牌与吃牌家族', 'subtitle': '三人 · 吃碰计分 · 暗手报听',
        'badge': 'JJ 口口香 · 84 张', 'description': '采用 JJ 2017 年计分表，吃碰杠和报听即时计分。',
        'rules': [
            '84 张，万筒加中发白。能吃上家，可碰或杠其他两家。',
            '成功吃、碰由供牌者付一分、两分。胡优先，杠碰优先于吃，只给最终实际成立的副露计分。',
            '听牌弃牌时可报听，暗手不亮，另两席立刻各付两分。报听后只可摸切或胡。'
            '直杠供牌者付五分，补杠每家一分，暗杠每家五分。',
            '标准形、七对均可胡。每份胡牌五分。自摸两家付，点炮或抢补杠可多响，由供牌者逐家付。荒庄保留已得分。',
        ],
    }},
}

VARIANTS[JJ_MULTIPLIER] = {
    'id': JJ_MULTIPLIER, 'family_id': 'ka-wu-xing', 'family': 'Three-player Ka Wu Xing', 'seat_count': 3,
    'name': 'JJ Three-Player · Multipliers',
    'aliases': ['JJ 倍率房', 'JJ 暗倒', '三人暗倒', 'JJ 2016'],
    'rule_version': 'jj-2016-multiplier-dojo-v1',
    'subtitle': 'Three seats · Open or dark ready · Win multipliers',
    'badge': 'JJ multipliers · 84 tiles',
    'description': 'Reveal your ready hand or keep its fixed triplets covered. Public waits restrict discards.',
    'rules': [
        '84 tiles: characters, dots and three dragons. No chow. You can declare open ready or dark ready until '
        'three tiles remain. Dark ready keeps eligible concealed triplets covered.',
        'After declaring, discard the drawn tile or take a win. A covered triplet can be konged if the waits stay the same. '
        'Discards avoid publicly provable waits when the player has another legal discard.',
        'Multiply the hand patterns, double once for a winning pair of seats with any reveal, then add two points. '
        'After a passed win, another offer with the same patterns is locked until your next draw.',
        'On a draw, the first revealed seat pays one to each seat that did not reveal. Kong payments stand. '
        'Consecutive own kongs double their immediate kong payments from the second onwards.',
    ],
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'sources': [{'title': 'JJ 2016 three-player multiplier rules', 'url': 'https://www.jj.cn/help/new_72708.shtml'}],
    'translations': {'zh-CN': {
        'name': 'JJ 三人麻将（倍率房）', 'family': '三人卡五星', 'subtitle': '三人 · 亮倒或暗倒 · 倍率结算',
        'badge': 'JJ 倍率房 · 84 张', 'description': '听牌后可亮倒，也可暗倒保留固定暗刻。出牌受公开听口约束。',
        'rules': [
            '84 张，万筒加中发白，不吃。末三张前可亮倒或暗倒，暗倒保留符合条件的暗刻。',
            '亮倒或暗倒后只可摸切或胡；暗倒保留的暗刻可在不改变听口时开杠。有安全牌时须避开对手公开可推知的听口。',
            '牌型乘倍数，付款双方有人亮倒或暗倒时只翻一次，再加两分。同样牌型过胡后，到自己摸牌前不能再接。',
            '荒庄首亮／暗倒者给未亮的每席一分，杠分保留。连杠第二次起即时杠分翻倍。',
        ],
    }},
}

VARIANTS[XIAOGAN.VARIANT] = {
    **_KA_WU_XING,
    'id': XIAOGAN.VARIANT,
    'name': 'Xiaogan Ka Wu Xing · Piao & Horse',
    'aliases': ['孝感卡五星', '压漂', '亮倒买马', 'Xiaogan'],
    'rule_version': 'xiaogan-piao-horse-dojo-v1',
    'subtitle': 'Three seats · Opening piao · Rank horse',
    'badge': 'Hubei · 84 tiles',
    'description': 'Choose piao before seeing your hand. A revealed self-draw flips a horse worth its rank.',
    'rules': [
        '84 tiles: dots, bamboo and three dragons. Choose zero to three piao privately before seeing your hand; reveal all '
        'choices together. This room allows chow. The nearest accepting seat wins when two seats can win on one tile.',
        'Declare open ready after a ready discard, reveal the remaining hand, and play only drawn-tile discards or wins. '
        'Open ready doubles the hand. The supplier pays both shares on a revealed discard win even when both have revealed.',
        'Each actual payer adds their own piao plus the winner’s piao once. A revealed self-draw with tiles left flips the '
        'tail tile, adding its rank or ten for a dragon to each payer’s bill.',
        'This room scores kong payments immediately and keeps them on a draw. No piao or horse points are paid on a draw.',
    ],
    'sources': [{'title': 'Xiaogan operator scoring', 'url': 'https://m.tcy365.com/news/d25805.html'},
                {'title': 'Operator opening and piao choices', 'url': 'https://m.tcy365.com/news/d26962.html'}],
    'translations': {'zh-CN': {
        'name': '孝感卡五星（压漂买马）', 'family': '三人卡五星', 'subtitle': '三人 · 开局压漂 · 亮倒自摸买马',
        'badge': '湖北 · 84 张', 'description': '看手牌前先压漂，亮倒自摸可翻一张马，按牌面加分。',
        'rules': [
            '84 张，筒条和中发白。看手牌前各选 0 至 3 漂，三家选完一起公布。本桌可吃，点炮按座次取最近的有效赢家。',
            '听牌时可亮倒，公开手牌后只可摸切或胡。亮倒胡牌翻倍，接炮由供牌者包两份，双亮仍包。',
            '每个实际付款人另付自己和赢家的漂数之和，只加一次。亮倒自摸且墙有余牌时翻尾牌，数牌按点数、箭牌按 10 分向各家收费。',
            '杠分即时付，荒庄保留。荒庄没有压漂和买马付款。',
        ],
    }},
}

_HANGZHOU = {
    'family_id': 'hangzhou-wildcard', 'family': 'Hangzhou wildcard family',
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
}
_HANGZHOU_HAND = (
    'Self-draw only. White Dragon is wild in concealed sets and pairs. Complete four sets and keep a wildcard '
    'waiting for its partner to win on any draw for a Burst. Six pairs and a wildcard can wait this way too.'
)
_HANGZHOU_FLOAT = (
    'Float an extra wildcard if the resulting hand can still Burst. Until your next draw, opponents can win or '
    'make concealed kongs on their turns and must discard their drawn tile. Consecutive floats and kongs multiply a Burst.'
)
_HANGZHOU_HAND_ZH = '只自摸。白板作财神，可在暗手中代牌。凑齐四副加一张财神，摸任意牌作将就是爆头；六对加财神则可七客。'
_HANGZHOU_FLOAT_ZH = '打掉一张多余的财神并保持爆头或七客，即可飘财。到下次摸牌前，别家只可自摸、暗杠或摸切。连飘和连杠可为爆头加倍。'

VARIANTS[THREE_HANGZHOU.VARIANT] = {
    **_HANGZHOU, 'id': THREE_HANGZHOU.VARIANT, 'seat_count': 3,
    'name': 'Hangzhou Three-Player · Fixed Dealer ×8',
    'aliases': ['三人杭麻', '三人杭州麻将', '笃三老庄', '白板财神'],
    'rule_version': 'hangzhou-three-fixed-dealer-dojo-v1',
    'subtitle': 'Three seats · White Dragon wild · No chow',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'White Dragon can complete any set or pair. Float one to raise the value of your next Burst.',
    'rules': [
        '136 tiles, no flowers. Pon and kong are allowed, with no chow. After you float, you may still make concealed kongs.',
        _HANGZHOU_HAND, _HANGZHOU_FLOAT,
        'Each losing seat pays its own share. Any share involving the dealer is multiplied by eight. '
        'The winner takes the next deal; a draw keeps the dealer, with no payments.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall three-player room rules', 'url': 'https://www.hzbianqu.com/game/581.html'},
                {'title': 'Three-player hand examples', 'url': 'https://www.hzbianqu.com/article/1548.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '三人杭麻（笃三老庄）', 'subtitle': '三人 · 白板财神 · 只碰杠',
        'badge': '浙江 · 136 张', 'description': '白板作财神。保留爆头听口时可打出多余财神，让下一次爆头加倍。',
        'rules': ['136 张，无花牌，只碰杠，不吃。飘财后本人只可暗杠，直到本局结束。',
                  _HANGZHOU_HAND_ZH, _HANGZHOU_FLOAT_ZH,
                  '自摸后按席付款，涉及庄家的每份乘八。赢家接庄，流局庄家连庄且无付款。'],
    }},
}

VARIANTS[FOUR_HANGZHOU.VARIANT] = {
    **_HANGZHOU, 'id': FOUR_HANGZHOU.VARIANT, 'seat_count': 4,
    'name': 'Hangzhou Four-Player · Three-Feed Liability',
    'aliases': ['杭州麻将', '四人杭麻', '平庄', '三摊承包', '白板财神'],
    'rule_version': 'hangzhou-four-three-feeds-dojo-v1',
    'subtitle': 'Four seats · Dealer ×2/4/8 · Three-feed liability',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'Call from an opponent three times to create liability across the dealer boundary.',
    'rules': [
        '136 tiles, no flowers. Chow from the previous seat; pon and kong from any seat. '
        'Non-dealers can take at most two calls from one another.',
        _HANGZHOU_HAND, _HANGZHOU_FLOAT,
        'If a dealer and non-dealer form three feeds in one direction, the feeder covers all shares when the caller wins. '
        'When the feeder wins, the caller covers all shares at double value. An added kong is still one feed; '
        'a third feed taken by a floater during a forced-draw circuit does not establish liability.',
        'Dealer shares begin at double, and increase to ×4 and then ×8 on repeats. The winner takes the next deal; '
        'a draw keeps the dealer and increases the multiplier, with no payments.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall four-player room rules', 'url': 'https://www.hzbianqu.com/game/37.html'},
                {'title': 'GameABC float circuit and liability examples', 'url': 'https://www.gameabc.com/news/201911/6191.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '四人杭麻（平庄三摊承包）', 'subtitle': '四人 · 庄闲 2／4／8 倍 · 三摊承包',
        'badge': '浙江 · 136 张', 'description': '庄闲之间吃碰杠同一家满三摊，可形成承包与反承包。',
        'rules': ['136 张，无花牌，可吃上家、碰杠任意一家，闲闲之间最多两摊。',
                  _HANGZHOU_HAND_ZH, _HANGZHOU_FLOAT_ZH,
                  '庄闲之间吃碰杠同一家满三摊，吃满者胡则供牌者包三份，供牌者胡则吃满者包三份再翻倍。'
                  '补杠仍算原来一摊，飘财人抓打圈内取得第三摊不触发承包。',
                  '涉及庄家每份起乘二，连庄升至四倍、八倍。赢家接庄，流局无付款并连庄。'],
    }},
}

SESSION_OPTIONS = {
    'hongzhong-one-ma': {'variant': 'hongzhong-one-ma', 'profile': ONE_MA},
    'fangchenggang-hongzhong': {'variant': 'fangchenggang-hongzhong', 'profile': FANGCHENGGANG},
    **{profile.value: {'profile': profile.value} for profile in Profile},
    **{room.VARIANT: {'variant': room.VARIANT} for room in (SHANGSI, HUNDRED, HEZHOU, LIUZHOU)},
    GUANGZHOU: {'variant': GUANGZHOU},
    GOLDEN_GUANGDONG: {'variant': GOLDEN_GUANGDONG},
    GUILIU: {'variant': GUILIU},
    MACAU: {'variant': MACAU},
    SICHUAN_EXCHANGE: {'variant': SICHUAN_EXCHANGE},
    YUNNAN: {'variant': YUNNAN},
    KAXINWU: {'profile': KAXINWU},
    SUIZHOU.VARIANT: {'variant': SUIZHOU.VARIANT},
    XIANGYANG.VARIANT: {'variant': XIANGYANG.VARIANT},
    SHIYAN.VARIANT: {'variant': SHIYAN.VARIANT},
    JJ_THREE: {'variant': JJ_THREE},
    JJ_MULTIPLIER: {'variant': JJ_MULTIPLIER},
    XIAOGAN.VARIANT: {'variant': XIAOGAN.VARIANT},
    **{room.VARIANT: {'variant': room.VARIANT} for room in HANGZHOU_ROOMS},
}
