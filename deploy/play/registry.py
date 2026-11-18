"""Versioned playable rules; English is the API default."""
from .anji import ANJI, ANJI_TEA_HOUSE
from .bengbu import BENGBU, BENGBU_FLOWER
from .blood_battle import SICHUAN_EXCHANGE, YUNNAN
from .chongji import CHONGJI, CHONGJI_SEVEN
from .dongyang import DONGYANG
from .guangzhou import VARIANT as GUANGZHOU
from .golden_guangdong import VARIANT as GOLDEN_GUANGDONG
from .guiliu import VARIANT as GUILIU
from .hangzhou import CHUNAN, FOUR_HANGZHOU, HANGZHOU_ROOMS, THREE_HANGZHOU, TONGLU, TWO_HANGZHOU
from .honors_bonus import HEZHOU, HUNDRED, LIUZHOU, SHANGSI
from .hongzhong import FANGCHENGGANG, ONE_MA
from .hongkong import VARIANT as HONG_KONG
from .jj_multiplier import VARIANT as JJ_MULTIPLIER
from .jj_three import VARIANT as JJ_THREE
from .kaxinwu import VARIANT as KAXINWU
from .kawuxing import SHIYAN, SUIZHOU, XIANGYANG
from .macau import VARIANT as MACAU
from .suited import Profile
from .tiantai import TIANTAI_WILD, VARIANT as TIANTAI
from .taizhou import TAIZHOU
from .daxi import DAXI
from .linhai import LINHAI
from .lipu import LIPU
from .jinyun import JINYUN, JINYUN_CANTONESE
from .jiaxing import JIAXING
from .pinghu import PINGHU
from .quzhou import QUZHOU_BANKED, QUZHOU_DUANZI, QUZHOU_LATA
from .xiaogan import XIAOGAN
from .zhejiang_flipped import DEQING, DEQING_XINSHI, SONGYANG, SUICHANG, XIAOSHAN

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

VARIANTS[TONGLU.VARIANT] = {
    **_HANGZHOU, 'id': TONGLU.VARIANT,
    'name': 'Tonglu Mahjong · Two Chows and Two Pons',
    'aliases': ['桐庐麻将', '两吃两碰', '有财必拷响'],
    'rule_version': 'tonglu-two-each-dojo-v1',
    'subtitle': 'Four seats · Separate call limits · Wildcard Burst required',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'A wildcard in your winning hand requires a Burst. Calls from any one player have separate limits.',
    'rules': [
        '136 tiles. Only self-draw wins. White Dragon is the wildcard. With a wildcard, you must win by Burst or '
        'seven-pairs Burst. Without one, regular hands and natural seven pairs can win.',
        'You may take up to two chows and two pons from the same seat. A direct kong uses a pon slot. '
        'An upgrade keeps its original slot. There is no three-feed liability.',
        'Float to retain your Burst wait and double the next Burst. During the circuit, other seats may only '
        'self-draw, make concealed kongs, or discard the drawn tile. The floater can still make open or concealed kongs.',
        'Dealer shares start at ×2, then rise to ×4 and ×8 on repeats. Draw to the last tile. A draw has no payments.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Tonglu rules', 'url': 'https://www.hzbianqu.com/game/62.html'},
                {'title': 'Tonglu scoring and room choices', 'url': 'https://www.hzbianqu.com/article/1444.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '桐庐麻将（两吃两碰）', 'subtitle': '四人 · 两吃两碰 · 有财必拷响',
        'badge': '浙江 · 136 张', 'description': '有财神须以爆头或七客胡牌。同一家能吃、能碰的次数分别计算。',
        'rules': ['136 张，白板财神，只自摸。有财神须爆头或七客，无财神可平胡或清七对。',
                  '同一家最多取两次吃、两次碰。直杠占一次碰，补杠沿用原摊，无三摊承包。',
                  '飘财本人仍可明杠或暗杠。抓打圈内另三家只可自摸、暗杠或摸切。',
                  '庄闲倍率平庄 2 倍，连庄升到 4 倍、8 倍。摸完牌墙无人胡则荒庄，无付款。'],
    }},
}

VARIANTS[CHUNAN.VARIANT] = {
    **_HANGZHOU, 'id': CHUNAN.VARIANT,
    'name': 'Chun’an Mahjong · Dealer Deal-ins',
    'aliases': ['淳安麻将', '淳安杭麻', '守庄', '闲放冲'],
    'rule_version': 'chunan-dealer-guard-dojo-v1',
    'subtitle': 'Four seats · Dealer deal-ins · Guarded pon',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'The dealer and a non-dealer can win from one another’s discards and added kongs.',
    'rules': [
        '136 tiles, White Dragon wild. Self-draw, Burst and floats work as in the Hangzhou family. '
        'Open kongs are unavailable after floating.',
        'Only a dealer and a non-dealer can win from one another’s discards or rob one another’s added kongs. '
        'The nearest accepting seat wins. The supplier alone pays ×8 of the hand value.',
        'Without a wildcard, the seat just before the dealer can only pon from its own preceding seat. '
        'Take up to two chows, pons or direct kongs in total from the same opponent, with no liability.',
        'A self-draw charges all losers, with ×8 on shares involving the dealer. A discard win is suspended '
        'while anyone’s wildcard float circuit is active. Draws have no payments.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Chun’an rules', 'url': 'https://www.hzbianqu.com/game/1055.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '淳安麻将（庄闲放冲）', 'subtitle': '四人 · 庄闲放冲和抢杠 · 守庄碰限制',
        'badge': '浙江 · 136 张', 'description': '庄家与闲家之间可放冲和抢补杠，庄家上家受守庄碰限制。',
        'rules': ['136 张，白板财神，可自摸、爆头、飘财。飘财后只能暗杠。',
                  '只有庄闲之间可以放冲或抢补杠。按座次取最近的接胡者，供牌者独付牌型的 8 倍。',
                  '庄家的上家无财神时，只能碰自己的上家。吃、碰、直杠合计同一家最多两摊，无承包。',
                  '自摸时其他三席各付，涉及庄家的份额乘 8。飘财抓打圈暂停放冲，流局无付款。'],
    }},
}

VARIANTS[TWO_HANGZHOU.VARIANT] = {
    **_HANGZHOU, 'id': TWO_HANGZHOU.VARIANT, 'seat_count': 2,
    'name': 'Hangzhou Two-Player · Flip Wildcard',
    'aliases': ['二人杭麻', '二人杭州麻将', '翻财神', 'Hangzhou heads-up'],
    'rule_version': 'hangzhou-two-flip-dojo-v1',
    'subtitle': 'Heads-up · Rotating wildcard · White Dragon substitute',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'A face-up tile changes the wildcard every hand. The other three copies are wild.',
    'rules': [
        'One opponent and the full 136-tile set. One tile is turned up for the whole hand. Its three other copies '
        'are wild, and White Dragons substitute for the indicated tile’s original face.',
        'The White Dragon substitute keeps its physical face on the table. It can make a chow, pon or kong '
        'as its substituted tile. A wildcard can complete concealed sets, pairs and Burst waits.',
        'Chow, pon, kong and self-draw are available. Nobody may take an exposed call during a float circuit. '
        'Seven pairs without a wildcard score natural seven pairs and actual quads double the hand.',
        'This practice room plays to the last live tile. The other seat pays one share on a win. '
        'The winner deals the next hand and an exhaustive draw keeps the dealer without a payment.',
    ],
    'sources': [{'title': 'Bianfeng two-player Hangzhou rules and White Dragon substitution',
                 'url': 'https://www.hzbianqu.com/game/1611.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '二人杭麻（翻财神）', 'subtitle': '单挑 · 每局翻财神 · 白板替身',
        'badge': '浙江 · 136 张', 'description': '每局翻一张，另外三张同样的牌作财神。白板替翻出牌的原牌面。',
        'rules': ['两人用整副 136 张。翻出的牌留在桌上，其余三张同样的牌作财神，白板替其原牌面。',
                  '白板还是显示白板，但吃碰杠时按替身牌计算。财神可在暗手凑牌、单吊爆头。',
                  '可吃碰杠，只自摸。飘财抓打圈双方都不吃碰明杠。无财神七对算清七对，四张天然同牌另翻倍。',
                  '本桌约定摸完可摸的牌，胡牌时对手付一份。赢家接庄，荒庄连庄且无付款。'],
    }},
}

VARIANTS[XIAOSHAN.VARIANT] = {
    **_HANGZHOU, 'id': XIAOSHAN.VARIANT,
    'name': 'Xiaoshan Mahjong · Ten Winds',
    'aliases': ['萧山麻将', '十风', '正风', '乱风'],
    'rule_version': 'xiaoshan-ten-winds-dojo-v1',
    'subtitle': 'Four seats · Ten Winds · Concealed kong bonuses',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'Win with an opening run of honor discards or build a qualifying self-draw hand.',
    'rules': [
        '136 tiles and a flipped wildcard. White Dragons substitute for the indicated face. Beyond Ten Winds, wins require self-draw. '
        'Hands with a wildcard require a hand multiplier of at least two.',
        'Up to two calls from one provider. A declined pon locks its tile until your draw. '
        'The second call from a player applies a swap-discard restriction if it is a chow.',
        'From your first discard, discard only honors, White Dragons or wildcards to build Ten Winds. '
        'Without open calls, you may win as you throw the tenth eligible tile, or keep going for a larger payment.',
        'All-honor pungs score ×128 and other all-honor holdings ×64. An exposed replacement win is ×2 '
        'and a concealed one is ×4. Dealer shares start at ×4 and repeat at ×8. Add one per winner’s kong '
        'after multiplying the shares, then cap each payer at 128.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Xiaoshan rules and scoring', 'url': 'https://www.hzbianqu.com/game/55.html'},
                {'title': 'Xiaoshan calls, wildcard and payout', 'url': 'https://www.hzbianqu.com/article/1060.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '萧山麻将（十风）', 'subtitle': '四人 · 十风 · 明暗杠开分计',
        'badge': '浙江 · 136 张', 'description': '起手连续打字牌可做十风，普通自摸按牌型与庄闲结算。',
        'rules': ['136 张，翻财神，白板替原牌面。只自摸，有财神须至少两倍牌型。',
                  '同一供牌者最多两摊。过碰锁同张至自己摸牌，第二摊如为吃，禁止吃张打张。',
                  '从首张开始连打字牌、白板或财神，且没有明副露，第十张起可打出并宣告十风。',
                  '正风 128 倍，乱风 64 倍。明杠开 2 倍，暗杠开 4 倍。二老庄 4 倍起，连庄升至 8 倍。'
                  '牌型乘庄闲倍率后每个赢家的杠加一，每位付款人封顶 128 分。'],
    }},
}

VARIANTS[SONGYANG.VARIANT] = {
    **_HANGZHOU, 'id': SONGYANG.VARIANT,
    'name': 'Songyang Mahjong · Kong Liability',
    'aliases': ['松阳麻将', '抢杠包三家', '正规七风', '十三幺'],
    'rule_version': 'songyang-kong-liability-dojo-v1',
    'subtitle': 'Four seats · Rob added kongs · 20-tile reserve',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'Rob an added kong to make its owner pay the whole table. Three feeds change the self-draw bill.',
    'rules': [
        '136 tiles, a flipped wildcard and White Dragon substitution. Self-draw or rob an added kong. '
        'Stop with 20 wall tiles reserved if nobody wins.',
        'Hand and event multipliers combine. This room counts natural or wildcard-assisted thirteen orphans and '
        'Seven Winds by physical face. Seven Winds uses seven distinct honors and seven isolated suited tiles.',
        'Dealer shares are double. On a robbed added kong, its owner pays all three shares. For a self-draw, '
        'a three-feed relationship can increase one payer to the total of all shares, or twice that amount for reverse liability. '
        'The other losers still pay their own shares.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Songyang scoring and reserve', 'url': 'https://www.hzbianqu.com/game/218.html'},
                {'title': 'Songyang hand multipliers', 'url': 'https://www.hzbianqu.com/article/1434.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '松阳麻将（抢杠包赔）', 'subtitle': '四人 · 抢补杠包三家 · 留尾 20 张',
        'badge': '浙江 · 136 张', 'description': '抢补杠时由被抢者包三家，自摸承包时其他输家仍付普通份。',
        'rules': ['136 张，翻财神、白板替身。可自摸或抢补杠，牌墙留 20 张流局。',
                  '牌型与胡法倍率相乘。十三幺与七风按实体牌面，分天然和财神代用。七风是七种字牌加七张数牌，同花色相隔至少三个点数。',
                  '庄闲之间份额翻倍。抢补杠由被抢者包三家。自摸时，三摊承包的一方付总额，反承包的一方付总额两倍，其他输家仍付本份。'],
    }},
}

VARIANTS[SUICHANG.VARIANT] = {
    **_HANGZHOU, 'id': SUICHANG.VARIANT,
    'name': 'Suichang Mahjong · Shared Discard Bill',
    'aliases': ['遂昌麻将', '地胡', '豪华七对', '十三幺'],
    'rule_version': 'suichang-shared-discard-dojo-v1',
    'subtitle': 'Four seats · Shared discard payments · Shrinking wall reserve',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'On a discard win, its supplier pays twice the other losers’ shares.',
    'rules': [
        '136 tiles, a flipped wildcard and White Dragon substitution. Win by self-draw, discard or robbing an added kong. '
        'The nearest accepting seat wins. The hand ends with 34 tiles reserved, releasing two reserve tiles per completed kong.',
        'Every loser pays two shares on a self-draw. On a discard win the supplier pays two and the other losers pay one each. '
        'An added-kong robbery doubles the share and its supplier pays the entire discard bill.',
        'Seven pairs pays twice the base, doubling for each natural quad in the hand. Qualifying hand and event multipliers combine. '
        'Thirteen orphans and Seven Winds have natural and wildcard tiers.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Suichang rules and payments', 'url': 'https://www.hzbianqu.com/game/47.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '遂昌麻将（点炮分摊）', 'subtitle': '四人 · 点炮分摊 · 开杠减留尾',
        'badge': '浙江 · 136 张', 'description': '点炮胡时，放炮者支付其他输家两倍的份额。',
        'rules': ['136 张，翻财神，白板替原牌面。可自摸、接炮或抢补杠，按离出牌者最近的胡家截胡。'
                  '留尾 34 张，每开一杠释放两张留尾牌。',
                  '自摸三家各付两份。接炮时放炮者付两份，其余两家各付一份。抢补杠计双倍份额，被抢者付清整笔点炮账。',
                  '七对子两番，每个天然四张组再翻倍。相容的牌型和胡法倍率相乘。十三幺和七风各分天然和财神代用。'],
    }},
}

VARIANTS[DEQING.VARIANT] = {
    **_HANGZHOU, 'id': DEQING.VARIANT,
    'name': 'Deqing Mahjong · Wukang',
    'aliases': ['德清麻将', '武康麻将', '地胡', '一炮多响', '顺位财神'],
    'rule_version': 'deqing-wukang-plain-dojo-v1',
    'subtitle': 'Four seats · Successor wildcard · Multiple winners',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'The next tile after the indicator becomes the wildcard. Early Earth wins and added-kong robberies can have multiple winners.',
    'rules': [
        '136 tiles with no flowers. The tile after the indicator is the wildcard. Ranks, winds and dragons wrap in their own cycles. '
        'White Dragon keeps its face unless it becomes the wildcard.',
        'Win by self-draw, first-circuit Earth or robbing an added kong. Earth charges the dealer 10 and other losers 5. '
        'Several players can win on the same eligible tile. The owner of a robbed kong pays three shares per winner.',
        'Four opening wildcards collect 10 from each loser. Ten consecutive wind or wildcard discards let you declare Ten Winds. '
        'A dealer’s opening win collects 20 from each loser. This is the plain room at base stake one.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Deqing Wukang rules and payouts', 'url': 'https://www.hzbianqu.com/game/919.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '德清麻将（武康）', 'subtitle': '四人 · 顺位财神 · 一炮多响',
        'badge': '浙江 · 136 张', 'description': '翻牌顺位为财神，地胡和抢补杠可一炮多响。',
        'rules': ['136 张，无花。翻牌下一位是财神，数牌、风牌、三元牌各自循环。白板只有当选财神时才作百搭。',
                  '可自摸、首圈地胡、抢补杠。地胡庄家付 10，其他输家各付 5。合条件的出牌可以多响。'
                  '被抢杠者向每位赢家支付三份。',
                  '起手四财神每位输家付 10，连续打出十张风牌或财神可宣告十风。庄家天胡三家各付 20。采用底分 1 的平胡房。'],
    }},
}

VARIANTS[DEQING_XINSHI.VARIANT] = {
    **_HANGZHOU, 'id': DEQING_XINSHI.VARIANT,
    'name': 'Deqing Mahjong · Xinshi',
    'aliases': ['新市麻将', '德清新市', '起手十风', '碰后补杠'],
    'rule_version': 'deqing-xinshi-plain-dojo-v1',
    'subtitle': 'Four seats · Opening Ten Winds · Immediate pon upgrade',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'Win at the deal with ten winds or wildcards, or upgrade a pon immediately without a robbery window.',
    'rules': [
        '136 tiles, with the tile after the indicator as wildcard. Opening Ten Winds needs ten winds or wildcards. '
        'A non-dealer holding nine may take the dealer’s first natural wind discard for the tenth.',
        'An ordinary win or Seven Pairs pays double if the winner never drew or held a wildcard. '
        'Heavenly Hand collects 10 per loser. Earth and opening wins by a non-dealer collect 10 from the dealer and 5 from other losers.',
        'You may upgrade a pon before your first discard after calling it. This kong cannot be robbed. '
        'Later added kongs can be robbed by multiple winners. The owner pays six times each winner’s own share.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Deqing Xinshi rules and payouts', 'url': 'https://www.hzbianqu.com/game/919.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '德清麻将（新市）', 'subtitle': '四人 · 起手十风 · 同轮碰后补杠',
        'badge': '浙江 · 136 张', 'description': '起手风牌与财神凑足十张可胡，同轮碰后补杠可直接补牌。',
        'rules': ['136 张，翻牌下一位为财神。起手有十张风牌或财神可直接胡。闲家起手九张时可以接庄家首弃的天然风牌。',
                  '从未拿过财神时，普通自摸和七对分数翻倍。天胡每家付 10，地胡及闲家起手特胡庄家付 10、其他输家各 5。',
                  '碰牌后尚未出牌，可立即补同一张牌，无抢杠窗口。之后的补杠可多家抢胡，被抢者按每位赢家自己的份额付六倍。'],
    }},
}

VARIANTS[ANJI.VARIANT] = {
    **_HANGZHOU, 'id': ANJI.VARIANT,
    'name': 'Anji Mahjong · Fifteen Flowers',
    'aliases': ['安吉麻将', '安吉15花', '十五花', '花牌替换'],
    'rule_version': 'anji-fifteen-flowers-dojo-v1',
    'subtitle': 'Four seats · 144 tiles · Automatic flower replacement',
    'badge': 'Zhejiang · 144 tiles',
    'description': 'The indicator turns regular faces into flowers. Reveal and replace each flower you draw.',
    'rules': [
        'Start with 144 tiles. White Dragons and copies of the indicator join the seasonal and plant flowers. '
        'If the indicator is a White Dragon or a flower, North also becomes a flower and East is the wildcard.',
        'Flowers are public and immediately replaced. Win by self-draw or an eligible first-circuit Earth win. '
        'A winning hand with neither flowers nor wildcards collects 10 from each opponent.',
        'You may chow, pon and kong, and upgrade an earlier pon right after a call. Kongs cannot be robbed. '
        'A flower replacement can trigger Kong Burst. Heavenly and Earthly Hand pay 10 per opponent.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Anji flower banks and payouts', 'url': 'https://www.hzbianqu.com/game/886.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '安吉麻将（15 花）', 'subtitle': '四人 · 144 张 · 自动补花',
        'badge': '浙江 · 144 张', 'description': '白板与翻牌同种的常规牌可变成花，摸到后亮出并补牌。',
        'rules': ['144 张。春夏秋冬、梅兰竹菊、白板和翻牌同种的牌为花。翻白板或实体花时，北风改作花，东风为财神。',
                  '花牌公开后立即补牌。可自摸或首圈地胡。无花无财神的和牌，每位对手付 10。',
                  '可吃、碰、杠。吃碰后可以直接补原有的碰牌，杠牌不能被抢。补花可以杠爆，天地胡每位对手付 10。'],
    }},
}

VARIANTS[ANJI_TEA_HOUSE.VARIANT] = {
    **_HANGZHOU, 'id': ANJI_TEA_HOUSE.VARIANT,
    'name': 'Anji Mahjong · Tea House Self-Draw',
    'aliases': ['茶苑安吉', '安吉八花', '安吉纯自摸'],
    'rule_version': 'anji-tea-house-self-draw-dojo-v1',
    'subtitle': 'Four seats · Eight flower faces · Self-draw only',
    'badge': 'Zhejiang · 144 tiles',
    'description': 'Play with eight seasonal and plant flowers. Each flower you expose adds to your own win.',
    'rules': [
        '144 tiles, with only the four seasons and four plants used as flowers. The indicator stays out of play. '
        'If it is a flower or White Dragon, East is the wildcard. Each exposed flower adds one point per opponent to your win.',
        'Win by self-draw, including your first ordinary draw as a non-dealer. Chow, pon and kong are available. '
        'Add a tile to a pon from your next draw onward. Kongs cannot be robbed.',
        'Seven natural pairs pay five before flowers. The loose-honors special uses all seven honors and seven isolated suited tiles. '
        'The room uses the ordinary wildcard float and the listed kong and flower combinations.',
    ],
    'sources': [{'title': 'Game Tea House Anji rules and payout table', 'url': 'https://www.5158wan.com/news/201809/7408.html'},
                {'title': 'Zhejiang Game Hall Anji isolated-honors shape', 'url': 'https://www.hzbianqu.com/game/886.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '茶苑安吉麻将（纯自摸）', 'subtitle': '四人 · 八种实体花 · 纯自摸',
        'badge': '浙江 · 144 张', 'description': '以春夏秋冬、梅兰竹菊作花。赢家每张补出的花都计入账单。',
        'rules': ['144 张，只用春夏秋冬、梅兰竹菊作花。指示牌移出牌墙，翻白板或实体花时东风为财神。'
                  '自己的每张补花在胡牌时使每位对手多付 1 分。',
                  '只可自摸，闲家第一次普通摸牌可地胡。可吃、碰、杠。碰后下次摸牌起可补杠，杠牌不能被抢。',
                  '天然七对每位对手付 5 再加花分。十三不搭使用七种字牌和七张隔张数牌。采用普通财飘及财飘、杠、补花的组合。'],
    }},
}

VARIANTS[DONGYANG.VARIANT] = {
    **_HANGZHOU, 'id': DONGYANG.VARIANT,
    'name': 'Dongyang Mahjong',
    'aliases': ['东阳麻将', '假财神', '真十三不靠', '半份账单'],
    'rule_version': 'dongyang-flexible-white-dojo-v1',
    'subtitle': 'Four seats · Flexible White Dragon · Shared discard bill',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'Use White Dragon at face value or as the ordinary face of the flipped wildcard.',
    'rules': [
        '136 tiles. The flipped face is the wildcard. In concealed ordinary shapes, White Dragon can keep its face '
        'or represent the natural face of the wildcard. Calls use the printed faces.',
        'Regular hands win by self-draw. Special shapes and first-discard Earth can also win on a discard. '
        'On a discard win, its supplier pays the full share and the other two opponents pay half each.',
        'Isolated tile hands score by the number of distinct honors and the spacing of suited ranks. '
        'Three wildcards can win on a draw. Burst, floating and kong-draw wins have separate values.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Dongyang rules and payout table', 'url': 'https://www.hzbianqu.com/game/910.html'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '东阳麻将', 'subtitle': '四人 · 白板两用 · 放铳分摊',
        'badge': '浙江 · 136 张', 'description': '白板可按本身使用，也可在手内补作财神翻出前的牌面。',
        'rules': ['136 张，翻牌同种为财神。白板在手内常规牌型可按本身或翻牌原牌面计算，副露按实际牌面。',
                  '平胡只可自摸，特殊牌型与庄家首弃地胡可接炮。放铳者付一份，另外两位输家各付半份。',
                  '十三不靠按字牌张数与数牌间距分档。手握三财神可以自摸特胡，另有爆头、漂财与杠开。'],
    }},
}

VARIANTS[CHONGJI.VARIANT] = {
    **_HANGZHOU, 'id': CHONGJI.VARIANT,
    'name': 'Bianfeng Chongji · Ten Ten Up',
    'aliases': ['边锋冲击麻将', '十十上', '十一风', '三杠全缴'],
    'rule_version': 'chongji-ten-ten-points-dojo-v1',
    'subtitle': 'Four seats · Successor wildcard · Growing dealer base',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'The dealer’s base starts at 11 and rises by 10 for each consecutive dealer win.',
    'rules': [
        '136 tiles. The next face after the indicator is the wildcard. The first accepting seat wins on a discard. '
        'Add the hand value to the dealer base. The discarder pays the full amount and other opponents pay half rounded up.',
        'An added-kong robbery charges its owner all three shares. First-discard Earth is billed as self-draw. '
        'Declining a self-draw blocks discard wins until your next draw.',
        'Special hands include isolated tiles, paired hands after kongs, eleven consecutive honor discards and '
        'a declaration after three kongs. The latter two collect 100 points from each opponent in this table.',
    ],
    'sources': [{'title': 'Bianfeng Chongji rules', 'url': 'https://app.bianfeng.com/rule/34'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '边锋冲击麻将（十十上）', 'subtitle': '四人 · 顺位财神 · 连庄加底',
        'badge': '浙江 · 136 张', 'description': '庄底从 11 起，庄家连胡每次加 10。',
        'rules': ['136 张，翻牌下一位为财神。同张接炮按座次优先。牌型加庄底，放铳者付全份，另两家付进位后的半份。',
                  '抢补杠由开杠者独付三份，首弃地胡按自摸结算。放弃自摸后，自己下次摸牌前不能接炮。',
                  '可和十三不搭、杠后对子、十一风和三杠全缴。后两项按房间约定由每位对手付 100 分。'],
    }},
}

VARIANTS[CHONGJI_SEVEN.VARIANT] = {
    **_HANGZHOU, 'id': CHONGJI_SEVEN.VARIANT,
    'name': 'Bianfeng Chongji · Seven Gaps',
    'aliases': ['边锋七嵌麻将', '七嵌', '外杠', '隔张成对'],
    'rule_version': 'chongji-seven-gaps-points-dojo-v1',
    'subtitle': 'Four seats · Seven gapped pairs · Separate kong points',
    'badge': 'Zhejiang · 136 tiles',
    'description': 'Win with seven same-suit pairs separated by one rank. Completed kongs settle points immediately.',
    'rules': [
        'Use Ten Ten Up scoring and calls. Seven Gaps adds a closed winning shape of seven same-suit pairs, '
        'each with exactly one intervening rank. Identical gaps can repeat. Wildcards may fill suited members.',
        'On a direct kong, the supplier pays 20 and the other two opponents pay 10 each. '
        'For a concealed or added kong, all opponents pay 20 each. The kong points are kept on a draw.',
        'Rob an added kong to cancel the move and charge three winning shares to its owner. '
        'The points for any earlier completed kongs stay on the ledger.',
    ],
    'sources': [{'title': 'Bianfeng Chongji rules and Seven Gaps appendix', 'url': 'https://app.bianfeng.com/rule/34'}],
    'translations': {'zh-CN': {
        'family': '杭麻财神', 'name': '边锋冲击麻将（七嵌）', 'subtitle': '四人 · 隔张成对 · 独立外杠分',
        'badge': '浙江 · 136 张', 'description': '同花色隔一张可组成一嵌，凑齐七嵌可胡。成杠即时结算。',
        'rules': ['沿用十十上。门清七嵌由七组同花色隔一张的牌组成，可重复，财神可补数牌。',
                  '直杠供牌者付 20，另两位输家各付 10。暗杠和补杠由三家各付 20，流局保留杠分。',
                  '抢补杠时补杠取消，被抢者付三份胡牌账。此前已完成的杠分保留。'],
    }},
}

VARIANTS[TIANTAI] = {
    'id': TIANTAI, 'family_id': 'taizhou-hu', 'family': 'Taizhou hu scoring', 'seat_count': 3,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Tiantai Mahjong · 300 Hu',
    'aliases': ['天台麻将', '天台三人', '天台300胡', '位花'],
    'rule_version': 'tiantai-108-300-hu-dojo-v1',
    'subtitle': 'Three seats · 108 tiles · 15-tile reserve',
    'badge': 'Zhejiang · Two suits and flowers',
    'description': 'Play with characters, circles, honors and eight flowers. Each tai doubles the winning hu.',
    'rules': [
        '108 tiles. Use characters, circles, seven honors and eight flowers. Replace flowers publicly and keep 15 tiles in reserve. '
        'Play with pon and kong, and use every honor at its printed value.',
        'Start at 20 hu and add flower, pair, wait and set points. Each tai doubles the result. Round up to ten and cap at 300. '
        'The dealer’s opening win and Eight Flowers pay 300; an Earth win pays 150.',
        'Several players may win on one discard or added kong. The supplier pays each winner’s full score. '
        'For self-draw, each opponent pays the full score. The last discard can only be taken to win.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Tiantai tiles, hu, tai and play restrictions', 'url': 'https://www.hzbianqu.com/game/970.html'}],
    'translations': {'zh-CN': {
        'family': '台州系胡数', 'name': '天台三人麻将（300 胡）', 'subtitle': '三人 · 108 张 · 留尾 15 张',
        'badge': '浙江 · 万筒字花', 'description': '用万、筒、字牌和八花。每台将胡数翻倍。',
        'rules': ['108 张，万筒字牌和八花。可碰、杠，字牌按本身使用。补花公开，牌墙留 15 张。',
                  '20 胡起算，加花牌、对子、听口及刻杠分。每台翻倍，再向上进十，300 胡封顶。天胡和八花 300 胡，地胡 150 胡。',
                  '放炮和抢补杠允许多响，供牌者向每位赢家付全额。自摸由其余两家各付全额。末张打出时只可胡。'],
    }},
}

VARIANTS[TIANTAI_WILD.VARIANT] = {
    **VARIANTS[TIANTAI], 'id': TIANTAI_WILD.VARIANT,
    'name': 'Tiantai Mahjong · White Dragon Wild',
    'aliases': ['天台白板财神', '天台财神房', '天台四财神', '天台清牌'],
    'rule_version': 'tiantai-108-white-wild-dojo-v1',
    'subtitle': 'Three seats · Four wildcards · 300-hu cap',
    'description': 'White Dragons can complete your hand. Score the highest legal wildcard assignment.',
    'rules': [
        '108 tiles, eight flowers and a 15-tile reserve. All four White Dragons are wild and can stand for '
        'another face in concealed shapes. Calls use ordinary printed faces.',
        'Hu and tai use the best legal assignment, including the winning tile’s role. A winning hand with '
        'no White Dragons earns one extra tai. Four wildcards or Eight Flowers pay the 300-hu maximum.',
        'Standard opening wins use their usual score. Multiple players may win on a discard or added kong. '
        'The supplier pays each winner; on self-draw both opponents pay.',
    ],
    'translations': {'zh-CN': {
        'family': '台州系胡数', 'name': '天台三人麻将（白板财神）', 'subtitle': '三人 · 四张白板财神 · 300 胡封顶',
        'badge': '浙江 · 万筒字花', 'description': '白板可在手内代牌，按合法归位的最高得分结算。',
        'rules': ['108 张，八花，留尾 15 张。四张白板均为财神，手内可代其他牌面。副露使用常规印刷牌面。',
                  '同时核算胡数、台数和胡张位置，取最佳合法归位。手里无白板的和牌计清牌一台。四财神或八花封顶 300 胡。',
                  '开局和牌按常规计分。接炮和抢补杠可多响，供牌家逐家付款，自摸由余下两家付款。'],
    }},
}

VARIANTS[TAIZHOU.VARIANT] = {
    'id': TAIZHOU.VARIANT, 'family_id': 'taizhou-hu', 'family': 'Taizhou hu scoring', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Taizhou Mahjong · Four Players',
    'aliases': ['台州麻将', '台州四人', '台州胡数', '台州承包', '生牌阶段'],
    'rule_version': 'taizhou-136-reciprocal-hu-dojo-v1',
    'subtitle': 'Four seats · Flipped wildcard · Reciprocal scoring',
    'badge': 'Zhejiang · Hu and faan',
    'description': 'Every seat scores its own hand. A liable player may cover all payments.',
    'rules': [
        '136 tiles. The flipped face becomes wild and White Dragons play as the flipped face. Leave 16 tiles in reserve.',
        'Only one player can win. Sets, valued pairs and faan determine each hand value. The winner starts at 10 hu. '
        'Each seat is capped at 100 hu. All six pairs of seats settle, including pairs of opponents who did not win.',
        'The late phase starts at 30 tiles including the reserve. A first-time discard that someone wins on then '
        'assigns all bills to the discarder. Feeding three chows for a pure hand and some live dragon discards can also assign liability.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Taizhou rules and reciprocal settlement',
                 'url': 'https://www.hzbianqu.com/game/936.html'}],
    'translations': {'zh-CN': {
        'family': '台州系胡数', 'name': '台州四人麻将', 'subtitle': '四人 · 翻得 · 四家互算',
        'badge': '浙江 · 胡数与番数', 'description': '四家各算自己的牌点，满足承包条件时由一人代付所有账单。',
        'rules': ['136 张。翻出的牌面作财神，白板替作翻出的牌面。留尾 16 张。',
                  '一炮一响。四家分别算刻杠、对子和番数，赢家 10 胡起算，各家最多 100 胡。六对座位逐对结算，输家之间也结算。',
                  '含尾牌剩 30 张时进入生牌阶段，打出生张被胡须承包。三吃清一色以及部分中发白生张也会触发承包。'],
    }},
}

VARIANTS[DAXI.VARIANT] = {
    'id': DAXI.VARIANT, 'family_id': 'taizhou-hu', 'family': 'Taizhou hu scoring', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Daxi Mahjong · Buy the Base', 'aliases': ['大溪麻将', '大溪买底', '温岭大溪'],
    'rule_version': 'daxi-136-buy-base-dojo-v1', 'subtitle': 'Four seats · Opening buy-in · Direct kong payments',
    'badge': 'Zhejiang · Standard tiles',
    'description': 'Decide whether to buy the base, then match tiles into sets and a pair.',
    'rules': [
        '136 tiles and a 16-tile reserve. A win uses four sets and a pair. Several players may win on one discard.',
        'Nondealers choose whether to buy the base before hands are revealed. The dealer buys automatically. '
        'Each payer pays one point, plus one if they bought and one if the winner bought. On a discard, the supplier pays.',
        'A direct kong is paid by the discarder, an added kong by the player who fed the original pon, and a concealed kong by everyone else. '
        'Kong points settle immediately and stay on a draw.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Daxi rules and buy-in scoring', 'url': 'https://www.hzbianqu.com/article/1123.html'}],
    'translations': {'zh-CN': {
        'family': '台州系胡数', 'name': '大溪麻将（买底）', 'subtitle': '四人 · 开局买底 · 杠分即时结算',
        'badge': '浙江 · 全副基础牌', 'description': '先选是否买底，再凑四副一对。',
        'rules': ['136 张，留尾 16 张，四副一对即可胡，同一张可以一炮多响。',
                  '开局摸牌展示前，闲家选买底或不买，庄家默认买底。自摸由其他三家各付，放炮由供牌家付。'
                  '每笔起付 1 子，付款家、胡家各自有买底就各加 1 子。',
                  '点杠由放牌家付，补杠由原来供碰的那家付，暗杠由另外三家各付。杠分即时结算，荒庄保留。'],
    }},
}

VARIANTS[LINHAI.VARIANT] = {
    'id': LINHAI.VARIANT, 'family_id': 'taizhou-hu', 'family': 'Taizhou hu scoring', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Linhai Mahjong · Six Chong Tiles', 'aliases': ['临海麻将', '临海抓冲', '四风聚齐', '三摊临海'],
    'rule_version': 'linhai-136-six-chong-dojo-v1', 'subtitle': 'Four seats · Wild White Dragons · Wind-based bonus',
    'badge': 'Zhejiang · Six bonus tiles',
    'description': 'Win with White Dragon wildcards and reveal six chong tiles matched to your seat wind.',
    'rules': [
        '136 tiles, 14 tiles in reserve, and White Dragons are wild. Several players may win on one discard.',
        'Seat-wind or dragon sets, pure or mixed suits, and Four Winds score tai. The base doubles each tai up to eight points. '
        'Reveal six chong tiles after a win. Each tile that matches the winner’s seat-wind group adds one to their payment share.',
        'Win on a discard and the supplier pays. Win by self-draw and each opponent pays, unless a reciprocal three-feed link makes one opponent pay all three shares.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Linhai rules, chong tiles and feed liability', 'url': 'https://www.hzbianqu.com/game/847.html'}],
    'translations': {'zh-CN': {
        'family': '台州系胡数', 'name': '临海麻将（六张抓冲）', 'subtitle': '四人 · 白板财神 · 按门风抓冲',
        'badge': '浙江 · 六张抓冲', 'description': '白板可在手内代牌，胡牌后抓六张，按门风找中冲牌。',
        'rules': ['136 张，留尾 14 张，白板为财神，允许一炮多响。',
                  '门风和字牌刻杠、清混一色、四风聚齐计台。底分逐台翻倍，最高 8，胡后抓六张，每中一冲，按份加 1。',
                  '点炮由放铳家付。自摸由其他三家付，双方形成三摊承包时由对应那家付三份。'],
    }},
}

VARIANTS[JINYUN.VARIANT] = {
    'id': JINYUN.VARIANT, 'family_id': 'hangzhou-wildcard', 'family': 'Hangzhou wildcard family', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Jinyun Mahjong · Classic', 'aliases': ['缙云麻将', '缙云经典', '缙云跟牌', '缙云财神会'],
    'rule_version': 'jinyun-136-classic-dojo-v1', 'subtitle': 'Four seats · Flipped wildcard · Dealer doubles',
    'badge': 'Zhejiang · Follow-tile payments',
    'description': 'Play with a flipped wildcard, opening follow-tile payments, and dealer-weighted scoring.',
    'rules': [
        '136 tiles and a 33-tile reserve. White Dragons play as the flipped face. Three wildcards in a valid self-drawn hand score the Wild Gathering bonus.',
        'Only the first player in turn order wins on a supplied tile. All opponents pay even on a discard or robbed added kong. '
        'Double any winner-to-payer bill that includes the dealer.',
        'If the dealer and next two players start with the same discard without a call, the fourth player’s discard triggers a follow-tile payment. '
        'Kong and follow-tile transfers stay on a draw.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Jinyun scoring, kongs and follow-tile payments', 'url': 'https://www.hzbianqu.com/game/61.html'}],
    'translations': {'zh-CN': {
        'family': '杭州财神系', 'name': '缙云麻将（经典）', 'subtitle': '四人 · 翻财神 · 庄闲翻倍',
        'badge': '浙江 · 开局跟牌', 'description': '翻财神，开局有跟牌罚分，庄闲之间胡牌付双份。',
        'rules': ['136 张，留尾 33 张，白板替翻出的牌面。三张财神组成合法自摸手牌时加财神会。',
                  '接炮一炮一响。自摸、点炮或抢补杠都由其他三家付分，胡家或付款家是庄家时该笔翻倍。',
                  '开局前三家无副露地打出同一张，按第四家的首张弃牌结算跟牌。流局保留杠分和跟牌分。'],
    }},
}

VARIANTS[JINYUN_CANTONESE.VARIANT] = {
    'id': JINYUN_CANTONESE.VARIANT, 'family_id': 'honors-and-chows', 'family': 'Honors and chows', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Jinyun Mahjong · Cantonese Style', 'aliases': ['缙云广式', '缙云广式麻将', '缙云自摸'],
    'rule_version': 'jinyun-136-cantonese-dojo-v1', 'subtitle': 'Four seats · Self-draw · Kong refunds on draws',
    'badge': 'Zhejiang · Natural tiles',
    'description': 'Win by self-draw or rob an added kong. A direct-kong win charges the seat that supplied the kong.',
    'rules': [
        '136 tiles with a 9-tile reserve. Pon and kong are available. Wins use natural tiles.',
        'Ordinary wins are self-draw. A player who supplies a successful added-kong robbery or the direct kong for a replacement win pays all three shares.',
        'The ordinary point table awards up to 30 for Thirteen Orphans. Immediate kong bills are refunded if the hand ends in a draw.',
    ],
    'sources': [{'title': 'Zhejiang Game Hall Jinyun Cantonese-style rules', 'url': 'https://www.hzbianqu.com/game/61.html'}],
    'translations': {'zh-CN': {
        'family': '字牌麻将', 'name': '缙云麻将（广式）', 'subtitle': '四人 · 以自摸为主 · 流局退杠',
        'badge': '浙江 · 无财神', 'description': '可碰可杠，自摸或抢补杠胡。点杠杠开由供杠家包付。',
        'rules': ['136 张，留尾 9 张，可碰、可杠。手牌按天然牌面组合。',
                  '普通胡牌只能自摸。抢补杠由补杠家付三份，点杠直接杠开则由供杠家付三份。',
                  '按普通档计分，十三幺 30 分封顶。流局时退回当局所有已付杠分。'],
    }},
}

VARIANTS[JIAXING.VARIANT] = {
    'id': JIAXING.VARIANT, 'family_id': 'hangzhou-wildcard', 'family': 'Hangzhou wildcard family', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Jiaxing Mahjong · Raise the Head', 'aliases': ['嘉兴麻将', '嘉兴抬头胡', '茶苑嘉兴', '嘉兴黄番'],
    'rule_version': 'jiaxing-136-tai-chayuan-dojo-v1', 'subtitle': 'Four seats · Qualifying hand types · Three-feed choices',
    'badge': 'Zhejiang · Carried faan',
    'description': 'Aim for a qualifying hand and choose whether to stay in when rivals form a liability link.',
    'rules': [
        '136 tiles. The flipped face is wild. To win, finish a standard hand as All Pungs, Pure Suit, Mixed Suit or All Honors. '
        'Wins are self-drawn or by robbing an added kong.',
        'After three feeds from one seat, the two seats are linked. Other active players choose whether to continue. '
        'A player who folds discards their draws and does not pay.',
        'Tai establishes the base and each faan doubles it. A win consumes at most one carried bonus. '
        'Feed liability and kong robbery change the shares paid by each seat.',
    ],
    'sources': [{'title': 'Game Tea House Jiaxing rules, qualifying hands and liability choices',
                 'url': 'https://wap.gametea.com/ask/201809/7367.html'}],
    'translations': {'zh-CN': {
        'family': '杭州财神系', 'name': '嘉兴麻将（茶苑抬头胡）', 'subtitle': '四人 · 四类起胡牌型 · 三摊后可退出',
        'badge': '浙江 · 黄番可累计', 'description': '凑够起胡牌型。别人三摊承包时，你可继续或退出本局计分。',
        'rules': ['136 张，同面百搭。四副一对须成对对胡、清一色、混一色或全老头，允许自摸及抢补杠。',
                  '从同一人吃碰杠三次后双方互为承包。其他牌手可继续，或退出计分后随摸随打。',
                  '台数作基数，每番翻倍。胡时最多使用一个累计黄番，承包和抢杠按各家身份分账。'],
    }},
}

VARIANTS[PINGHU.VARIANT] = {
    'id': PINGHU.VARIANT, 'family_id': 'hangzhou-wildcard', 'family': 'Hangzhou wildcard family', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Pinghu Mahjong · Flowers and Faan', 'aliases': ['平湖麻将', '平湖花番', '平湖补花'],
    'rule_version': 'pinghu-144-flower-faan-dojo-v1', 'subtitle': 'Four seats · Eight flowers · Flower-count scoring',
    'badge': 'Zhejiang · 144 tiles',
    'description': 'Replace flowers and score wind sets, kongs and awards together.',
    'rules': [
        '144 tiles, a 16-tile reserve and same-face wildcards. Flowers are shown and replaced automatically. '
        'Four sets and a pair can win on a draw, discard or added kong.',
        'Flowers, wind sets and kongs increase the flower count. Each faan doubles one plus the flower count. '
        'Show the count and faan separately in settlement.',
        'On self-draw each opponent pays. On a discard only its supplier pays. The supplier of a robbed added kong pays all three shares.',
    ],
    'sources': [{'title': 'Bianfeng Pinghu flower counts and faan schedule', 'url': 'https://www.hzbianqu.com/game/510.html'},
                {'title': 'Game Tea House Pinghu draw and flower replacements', 'url': 'https://www.gametea.com/ask/201809/7357.html'}],
    'translations': {'zh-CN': {
        'family': '杭州财神系', 'name': '平湖麻将（花番）', 'subtitle': '四人 · 八花 · 花数乘番',
        'badge': '浙江 · 144 张', 'description': '公开补花，把花牌、风刻和杠折成花数，再乘番。',
        'rules': ['144 张，留 16 张，同面百搭。花牌自动公开补摸。四副一对可自摸、接炮或抢补杠。',
                  '花牌、风刻和杠计花，按「花数加一」逐番翻倍。结算分列花数和番数。',
                  '自摸三家各付，放炮家独付一份，补杠被抢时供牌家独付三份。'],
    }},
}

VARIANTS[QUZHOU_DUANZI.VARIANT] = {
    'id': QUZHOU_DUANZI.VARIANT, 'family_id': 'hangzhou-wildcard', 'family': 'Hangzhou wildcard family', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Quzhou Mahjong · Duanzi', 'aliases': ['衢州段子胡', '段子麻将', '十三百搭', '真百搭'],
    'rule_version': 'quzhou-136-duanzi-dojo-v1', 'subtitle': 'Four seats · No chow · Special hands',
    'badge': 'Zhejiang · Separated suits',
    'description': 'Build a qualifying special hand, including separated ranks and single honors.',
    'rules': ['136 tiles and the full wall. The flipped face is wild. White Dragons can stand for its original face inside a winning hand.',
              'Pon and kong are available. Only special hands can win. A separated hand assigns 147, 258 and 369 to different suits, with five to seven distinct honors.',
              'Opponents each pay on self-draw. A discard supplier pays alone. A robbed added kong charges its supplier double the discard value.'],
    'sources': [{'title': 'Zhejiang Game Hall Quzhou rules and scoring', 'url': 'https://www.hzbianqu.com/game/93.html'},
                {'title': 'Zhejiang Game Hall Quzhou kong robbery', 'url': 'https://www.hzbianqu.com/article/1127.html'}],
    'translations': {'zh-CN': {
        'family': '杭州财神系', 'name': '衢州麻将（段子胡）', 'subtitle': '四人 · 不吃 · 特殊牌型',
        'badge': '浙江 · 十三百搭', 'description': '只胡段子牌型，可以按三门隔张和单字凑十三百搭。',
        'rules': ['136 张，摸尽牌墙。翻同面财神，白板在和牌中可为本面或替原财神牌面。',
                  '可碰杠，只有特殊牌型可以胡。十三百搭把 147、258、369 分给三门，另取五至七种单字。',
                  '自摸三家分别付，点炮供牌家独付，抢补杠供牌家付点炮值两倍。'],
    }},
}

VARIANTS[QUZHOU_LATA.VARIANT] = {
    **VARIANTS[QUZHOU_DUANZI.VARIANT], 'id': QUZHOU_LATA.VARIANT,
    'name': 'Quzhou Mahjong · Lata', 'aliases': ['衢州邋遢胡', '烂百搭', '邋遢胡普通'],
    'rule_version': 'quzhou-136-lata-ordinary-dojo-v1', 'subtitle': 'Four seats · Chow and plain wins · 34-tile reserve',
    'description': 'Play ordinary hands and flexible separated hands with Quzhou’s shared discard bills.',
    'rules': ['136 tiles and a 34-tile reserve. Chow, pon and kong are available. Plain hands and flexible separated hands can win.',
              'A plain self-draw collects 2 per opponent. A special hand collects 4 per opponent. On a discard, the two uninvolved opponents each pay one less.',
              'Opening wins, kong replacement wins and four-set single waits use fixed bills. A robbed added kong charges its supplier 30.'],
    'sources': [{'title': 'Zhejiang Game Hall ordinary and banked Lata rules', 'url': 'https://www.hzbianqu.com/game/93.html'},
                {'title': 'Zhejiang Game Hall fixed Lata payments', 'url': 'https://www.hzbianqu.com/article/1097.html'}],
    'translations': {'zh-CN': {
        'family': '杭州财神系', 'name': '衢州麻将（邋遢胡）', 'subtitle': '四人 · 可吃可平胡 · 留尾 34 张',
        'badge': '浙江 · 普通结算', 'description': '平胡和灵活间隔的烂百搭都能胡，放炮时另两家也分担。',
        'rules': ['136 张，留 17 墩。可吃碰杠，平胡和烂百搭都可胡。',
                  '平胡自摸各家付 2，特殊牌自摸各付 4。接炮时放炮家付全额，另两家各少付 1。',
                  '天地胡、杠开及四副单吊有固定账单。抢补杠由供牌家独付 30。'],
    }},
}

VARIANTS[QUZHOU_BANKED.VARIANT] = {
    **VARIANTS[QUZHOU_LATA.VARIANT], 'id': QUZHOU_BANKED.VARIANT,
    'name': 'Quzhou Mahjong · Lata Banked Match', 'aliases': ['衢州局麻', '邋遢胡局麻', '衢州三十子'],
    'rule_version': 'quzhou-136-lata-banked-dojo-v1', 'subtitle': 'Four seats · 30 points each · Match ends on overdraft',
    'description': 'Manage a 30-point balance. Dealer and supplier positions decide each bill.',
    'rules': ['Use the Lata tile and win rules. Each seat receives 30 points to start. Plain and special hands pay the same position-based amounts.',
              'The dealer’s self-draw charges each opponent 8. A nondealer’s self-draw charges the dealer 8 and the other seats 4 each. Discard wins use the dealer and supplier positions.',
              'If a bill exceeds a player’s balance they pay the balance and the match ends. The largest remaining balance wins.'],
    'translations': {'zh-CN': {
        'family': '杭州财神系', 'name': '衢州麻将（邋遢胡局麻）', 'subtitle': '四人 · 各 30 分 · 付不够则结束',
        'badge': '浙江 · 局麻', 'description': '开局各有 30 分，按庄闲与放炮身份算账。',
        'rules': ['沿用邋遢胡的牌张和胡牌规则。每家起始 30 分，平胡与特殊牌型采用相同的按身份付款。',
                  '庄家自摸每家付 8；闲家自摸庄付 8，其他两家各付 4。点炮按庄闲和放炮方另算。',
                  '应付超过余额时只付余额，整场结束，余额最高者获胜。'],
    }},
}

VARIANTS[BENGBU.VARIANT] = {
    'id': BENGBU.VARIANT, 'family_id': 'honors-and-chows', 'family': 'Regional honors and scoring', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Bengbu Mahjong · Classic', 'aliases': ['蚌埠麻将', '天天爱蚌埠麻将', '蚌埠无花'],
    'rule_version': 'bengbu-136-classic-dojo-v1', 'subtitle': 'Four seats · No chow · 14-tile reserve',
    'badge': 'Anhui · Dealer surcharge',
    'description': 'Keep all the winds and dragons in play. Only winning hands collect kong points.',
    'rules': ['136 tiles. Leave the last 14 tiles. Pon or kong, without chows or seven pairs. A concealed kong shows one face.',
              'A discard pays 1 and a self-draw charges each opponent 2. An exposed kong adds 1 and a concealed kong adds 2 to the winning hand.',
              'A self-draw off a kong doubles the hand value. When an added kong is robbed, its owner covers all three self-draw bills. Each bill involving the dealer adds 1.'],
    'sources': [{'title': 'Zhejiang Game Hall Bengbu room rules', 'url': 'https://www.hzbianqu.com/game/894.html'}],
    'translations': {'zh-CN': {
        'family': '字牌计分系', 'name': '蚌埠麻将（无花）', 'subtitle': '四人 · 不吃 · 留尾 14 张',
        'badge': '安徽 · 庄加一', 'description': '风牌和三元牌都作手牌，赢家才结算自己的杠分。',
        'rules': ['136 张，留尾 14 张。只能碰杠，七对不胡。暗杠亮出一张。',
                  '点炮 1，自摸每家 2。赢家每个明杠加 1，每个暗杠加 2。',
                  '杠开翻倍。抢补杠按自摸计算，杠家包三份。涉及庄家的份额另加 1。'],
    }},
}

VARIANTS[BENGBU_FLOWER.VARIANT] = {
    **VARIANTS[BENGBU.VARIANT], 'id': BENGBU_FLOWER.VARIANT,
    'name': 'Bengbu Mahjong · East Wind Flowers', 'aliases': ['蚌埠东风令', '东风令带花', '蚌埠补花'],
    'rule_version': 'bengbu-136-east-flower-dojo-v1', 'subtitle': 'Four seats · East and dragons as flowers',
    'badge': 'Anhui · Delayed flower replacement',
    'description': 'Expose flowers on your first ordinary draw. An early call leaves your flowers concealed.',
    'rules': ['136 tiles, no reserve. East and the three Dragons are flowers. The dealer replaces flowers before play. '
              'Each other seat waits for its first ordinary draw, even after an early pon or kong.',
              'Only standard hands can win, with no chow. Add 1 for each exposed flower in the winning hand. '
              'On a self-draw, each exposed or concealed kong adds 2 per payer.',
              'Robbing an added kong charges its owner one discard bill. Any bill involving the dealer adds 1.'],
    'translations': {'zh-CN': {
        'family': '字牌计分系', 'name': '蚌埠麻将（东风令带花）', 'subtitle': '四人 · 东风和中发白作花',
        'badge': '安徽 · 延后补花', 'description': '庄家起手补花，闲家碰杠抢先上手时，花牌仍留在手中。',
        'rules': ['136 张，摸尽牌墙。东风和中发白作花。庄家起手补，闲家首次正常摸牌时补，第一圈碰杠也要等。',
                  '不吃、不胡七对。赢家每花加 1；自摸时每个明杠或暗杠每家加 2。',
                  '抢补杠按一份点炮付款，庄家参与的账单另加 1。'],
    }},
}

VARIANTS[LIPU.VARIANT] = {
    'id': LIPU.VARIANT, 'family_id': 'honors-and-chows', 'family': 'Regional honors and scoring', 'seat_count': 4,
    'default_policy': 'heuristic', 'policies': [{'id': 'heuristic', 'name': 'Beginner · Tile efficiency'}],
    'name': 'Lipu Mahjong · Feed Liability', 'aliases': ['荔浦麻将', '荔浦捞杠', '荔浦陪杀', '荔浦三比'],
    'rule_version': 'lipu-136-feed-liability-dojo-v1', 'subtitle': 'Four seats · Three-feed liability · Multiple winners',
    'badge': 'Guangxi · Kong ledger',
    'description': 'A third feed links two seats. Feeding four sets or three pons to the same opponent blocks your wins.',
    'rules': ['136 tiles. Chow, pon, kong and seven pairs are allowed. An added kong can be robbed. Several players can win on one discard.',
              'A third feed to the same seat creates a reciprocal liability. The linked seat pays an extra share on a discard win or covers all shares on an ordinary self-draw.',
              'Four feeds or three pons to the same opponent block wins and new kong points for the supplier. Earlier kong transfers stand.',
              'A discard or pon supplier pays for the exposed kong. The last kong bill is returned on a kong-draw or post-kong discard win. Horses are off.'],
    'sources': [{'title': 'Zhejiang Game Hall Lipu feeds, kongs and payout table', 'url': 'https://www.hzbianqu.com/game/1146.html'}],
    'translations': {'zh-CN': {
        'family': '字牌计分系', 'name': '荔浦麻将（陪杀与捞杠）', 'subtitle': '四人 · 三比互包 · 一炮多响',
        'badge': '广西 · 杠分单算', 'description': '三比结成互包，向同一家供出四比或三碰后封胡。',
        'rules': ['136 张，可吃碰杠，可胡七对。开启捞杠和抢补杠，允许一炮多响。',
                  '向同一家供三比后互包。接炮时互包家陪付一份，普通自摸时互包家承担三份。',
                  '向同一家供四比或三碰后，供牌家封胡、后续杠不计分，之前的杠分保留。',
                  '明杠由点杠或点碰家付。杠开或杠后炮退最近一笔杠账。本桌关闭抓马和坐马。'],
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
    XIAOSHAN.VARIANT: {'variant': XIAOSHAN.VARIANT},
    SONGYANG.VARIANT: {'variant': SONGYANG.VARIANT},
    SUICHANG.VARIANT: {'variant': SUICHANG.VARIANT},
    DEQING.VARIANT: {'variant': DEQING.VARIANT},
    DEQING_XINSHI.VARIANT: {'variant': DEQING_XINSHI.VARIANT},
    ANJI.VARIANT: {'variant': ANJI.VARIANT},
    ANJI_TEA_HOUSE.VARIANT: {'variant': ANJI_TEA_HOUSE.VARIANT},
    DONGYANG.VARIANT: {'variant': DONGYANG.VARIANT},
    CHONGJI.VARIANT: {'variant': CHONGJI.VARIANT},
    CHONGJI_SEVEN.VARIANT: {'variant': CHONGJI_SEVEN.VARIANT},
    TIANTAI: {'variant': TIANTAI},
    TIANTAI_WILD.VARIANT: {'variant': TIANTAI_WILD.VARIANT},
    TAIZHOU.VARIANT: {'variant': TAIZHOU.VARIANT},
    DAXI.VARIANT: {'variant': DAXI.VARIANT},
    LINHAI.VARIANT: {'variant': LINHAI.VARIANT},
    JINYUN.VARIANT: {'variant': JINYUN.VARIANT},
    JINYUN_CANTONESE.VARIANT: {'variant': JINYUN_CANTONESE.VARIANT},
    JIAXING.VARIANT: {'variant': JIAXING.VARIANT},
    PINGHU.VARIANT: {'variant': PINGHU.VARIANT},
    **{rules.VARIANT: {'variant': rules.VARIANT} for rules in (QUZHOU_DUANZI, QUZHOU_LATA, QUZHOU_BANKED)},
    **{rules.VARIANT: {'variant': rules.VARIANT} for rules in (BENGBU, BENGBU_FLOWER, LIPU)},
}
