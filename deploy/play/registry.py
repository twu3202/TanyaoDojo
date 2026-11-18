"""Versioned playable rules; English is the API default."""
from .hongzhong import FANGCHENGGANG, ONE_MA

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

SESSION_OPTIONS = {
    'hongzhong-one-ma': {'variant': 'hongzhong-one-ma', 'profile': ONE_MA},
    'fangchenggang-hongzhong': {'variant': 'fangchenggang-hongzhong', 'profile': FANGCHENGGANG},
}
