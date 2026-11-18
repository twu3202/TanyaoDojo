'use strict';
const I18N={
en:{
 darkReadyTag:'Dark ready', declareDarkReady:'Dark ready · {tile}', promptDarkReady:'Your hand is partly revealed. Discard your draw, win, or make an eligible kong.',
 promptPiao:'Choose your piao for this hand before seeing the tiles.', piaoChoice:'Piao {value}', piaoTag:'Piao {value}',
 piaoTransfers:'Piao points · {amounts}', horseTransfers:'Horse points · {amounts}', rankHorseSummary:'{name} · Horse adds {points} per payer',
 promptPaidReady:'You declared ready. Discard the drawn tile or choose a win.',
 chowTransfers:'Chow points · {amounts}', ponTransfers:'Pon points · {amounts}', paidReadyTransfers:'Ready declaration points · {amounts}',
 openReadyTag:'Open ready', declareOpenReady:'Open ready · {tile}', pickOpenReadyTile:'Choose a marked discard to reveal your ready hand.',
 promptOpenReady:'Your ready hand is revealed. Discard the drawn tile or choose a win.', openReadyTransfers:'First-reveal payments · {amounts}',
 promptLockedWin:'Your hand is locked. Discard the tile you drew, or choose a win or kong.', flowKinds:{ron:'Discard win',zimo:'Self-draw',rob_kong:'Robbed kong'}, winningTiles:'Winning tiles', winCount:'{count} wins', flowWinPayments:'{name} · {action} {tile} · {fan} fan · {patterns} · {payments}',
 promptSameSuitExchange:'Select three tiles of the same suit.', winningTile:'Winning discard',
 flowerPigTransfers:'Flower-pig payments · {amounts}', readyTransfers:'Ready-hand payments · {amounts}', kongRefunds:'Kong refunds · {amounts}',
 directions:{clockwise:'clockwise',counterclockwise:'counterclockwise',opposite:'across the table'},
 homeLabel:'Mahjong Dojo home',setupLabel:'Table setup',tableLabel:'Mahjong table',languageLabel:'Language',sourceLabel:'Rule source',
 brand:'Mahjong Dojo', rules:'Rules', edition:'OPEN PLAY · 02', eyebrow:'ONE TABLE. MANY WAYS TO PLAY.',
 hero1:'Take a seat.',hero2:'Play a hand.',intro:'Choose a ruleset and play against AI.',intro2:'From the first draw to the final settlement.',
 coming:'Regional rulesets are added by family, once they are ready to play.',chooseVariant:'Choose a ruleset',searchRulesets:'Search rulesets',familyFilter:'Family',allFamilies:'All families',noRulesets:'No matching rulesets.',showMoreRulesets:'Show {count} more',filteredRulesets:'{count} of {total} rulesets',live:'{count} available',loadingCatalog:'Preparing the tables…',
 choosePolicy:'Choose your opponents',policyLabel:'AI difficulty',model:'Standard · Dojo model',heuristic:'Beginner · Tile efficiency',
 setupNote:'You face {count} AI opponents. Autoplay is available during the hand.',start:'Start playing',footer:'TanyaoDojo · One hand at a time',footer2:'Complete rules · Play against AI',
 back:'← Back to lobby',inGame:'AT THE TABLE',currentRules:'CURRENT RULESET',viewRules:'Read the rules ↗',helpTitle:'One tile at a time',help:'Select a tile, then confirm your discard. Calls, wins and passes appear when available.',
 autoplay:'Autoplay this hand',cancelAuto:'Stop autoplay',connecting:'Connecting to the table…',connected:'● Table connected',reconnecting:'Connection interrupted. Retrying…',
 preparing:'Preparing the hand',aiReady:'The AI players are taking their seats',yourTurn:'Your turn',thinking:'AI is thinking',handEnd:'Hand complete',matchEnd:'Hanchan complete',matchComplete:'Match complete',matchHand:'Hand {number} of {total}',matchWinner:'{names} finished first after {total} hands.',firstPlace:'First place',tableError:'Table error',tableClosed:'Table closed',autoState:'Autoplaying this hand',
 events:'Table history',liveEvents:'RECENT EVENTS',privacy1:'Public information only',privacy2:'Concealed tiles stay private',before:'BEFORE YOU SIT DOWN',gotIt:'Got it',closeRules:'Close rules',
 resultEyebrow:'THE HAND IS SETTLED',handResult:'Hand result',next:'Next hand ↗',newTable:'Play again ↗',returnLobby:'Back to lobby',
 opening:'Opening your table',openingHint:'The wall and AI players will be ready shortly.',
 bonusSummary:'{name}: {count} bonus tiles, {hits} hit(s), {points} points per share.',bonusTileHeading:'Bonus tiles',fishTile:'Exploding fish',fishSummary:'{name} · {hits} fish · {points} points per share',winShare:'{name} · {fan} fan · {points} points per share · {patterns}',winSharePoints:'{name} · {points} points per share · {patterns}',winPayments:'{name} · {fan} faan · {patterns} · {payments}',tierPayments:'{name} · Special tier {tier} · {patterns} · {payments}',pointPayments:'{name} · {patterns} · {payments}',wakeTiles:'Wake indicators',kongTransfers:'Kong points · {amounts}',dealerEcho:'Opening discard match · {amounts}',wakeSummary:'{name}: {hits} hits from {count} indicators.',oneMaSummary:'{name}: {extra} bonus points · {points} points per share.',wall:'{count} tiles left',dora:'Dora indicators',wild:'Wild tile',pot:'Riichi pool: {count} × 1,000',honba:'{count} honba',
 dealer:'Dealer',won:'Won',riichi:'Riichi',ready:'Ready',void:'Void: {suit}',concealed:'{count} concealed tiles',selectTile:'Select {tile}',selectReadyTile:'Select {tile} (can declare ready)',pickReadyTile:'Choose a marked tile to declare ready.',
 hand:'Hand {number}',round:'{wind} {number}',roundWindLabel:'{wind} round',promptExchange:'Select three tiles to pass to the next seat.',confirmExchange:'Exchange three tiles',exchangeCount:'{count} of 3 tiles selected',bilateralHorses:'{name} wins · Horse hits: {hits}',promptVoid:'Choose the suit you will discard first.',promptTurn:'Your turn. Select a tile or an action.',promptRob:'Rob a kong, or pass.',promptAuto:'AI is playing the rest of this hand for you.',promptWait:'Please wait while the other players act.',promptEnd:'The hand has been settled.',promptError:'This table encountered an error. Please return to the lobby.',
 selected:'Selected: {tile}',clickTile:'Click a tile to select',pickTile:'Select a tile first',discard:'Discard {tile}',declareReady:'Declare ready · {tile}',voidAction:'Void · {suit}',
 action:{gang_bu:'Added kong',gang_an:'Closed kong',gang_ming:'Open kong',none:'Pass',pass:'Pass',reach:'Riichi',declare_ready:'Declare ready',chi:'Chi',pon:'Pon',peng:'Pon',daiminkan:'Open kan',zhigang:'Open kan',ankan:'Closed kan',rob_kong:'Rob a kong',kakan:'Added kan',bugang:'Added kan',hora:'Win',ron:'Ron',zimo:'Tsumo',ryukyoku:'Nine-terminals draw'},
 wind:{E:'East',S:'South',W:'West',N:'North'},suits:['Characters','Dots','Bamboo'],names:['You','Bamboo','Dew','Pine'],
 tileHonors:{E:'East',S:'South',W:'West',N:'North',P:'White dragon',F:'Green dragon',C:'Red dragon'},red:'Red',
 messages:{win_total:'{actor} received {points} win points{patterns}',dark_ready:'{actor} declared dark ready and kept fixed triplets covered.',piao_complete:'Piao choices are set. The hand begins.',paid_ready:'{actor} declared ready and received two points from each opponent.',open_ready:'{actor} revealed a ready hand.',flow_win:'{actor} · {action} {tile} · {fan} fan{patterns}',repeat_win_end:'Wall exhausted. The final ready-hand payments are settled.',exchange_direction:'The table exchanged tiles {direction}. Next, choose a void suit.',yunnan_draw:'Wall exhausted. Flower-pig, ready-hand and kong refund payments are settled.',hand_start:'A new hand begins.',hand_start_void:'Hand {number} begins. Choose a void suit.',discard:'{actor} discarded {tile}.',void:'{actor} chose {suit} as the void suit.',action:'{actor}: {action}.',tsumo:'{actor} won by self-draw.',ron:'{actor} won on {target}’s discard.',win_fan:'{actor} won · {fan} fan{patterns}',win_tier:'{actor} won · Special tier {tier}{patterns}',win_points:'{actor} wins · {points} points per share{patterns}',draw:'The hand ends in a draw.',blood_end:'Three players have won. The blood battle ends.',sichuan_draw:'Wall exhausted. Ready-hand payments and refunds are settled.',hongzhong_draw:'Wall exhausted. Existing kong payments stand.',rob_kong_win:'{actor} robbed {target}’s added kong.',hongzhong_win:'{actor} won by self-draw.',one_ma_bonus:'One-ma bonus: {tiles} · {extra} extra points per share.',bonus:'Bonus tiles: {tiles} · {hits} hit(s).'},
 reason:{repeat_win_end:'Wall exhausted. All wins, kongs and ready-hand payments are settled.',yunnan_draw:'Wall exhausted; flower-pig, ready-hand and kong refund payments settled.',rob_concealed_win:'{names} won by robbing a concealed kong.',suited_self_draw:'{names} won by self-draw.',suited_discard_win:'{names} won on a discard.',winners:'{names} won the hand.',draw:'Draw settlement',blood_end:'Three players have won. The blood battle ends.',sichuan_draw:'Wall exhausted; ready-hand payments and refunds settled.',hongzhong_draw:'Wall exhausted; existing kong payments stand.',rob_kong_win:'{names} won by robbing an added kong.',hongzhong_win:'{names} won by self-draw.',four_red:'Four concealed red dragons.'},
 fans:{'三元七对':'Seven pairs with all dragons','亮倒':'Open ready','八倍封顶':'Eightfold cap','卡五星':'Middle-five wait','手抓一':'Four exposed sets','明四归一':'Exposed four-in-one','暗四归一':'Concealed four-in-one','卡心五':'Middle-five wait','双杠上花':'Win after two kongs','三杠上花':'Win after three kongs','四杠上花':'Win after four kongs','双龙七对':'Double-dragon seven pairs','三龙七对':'Triple-dragon seven pairs','清双龙七对':'Pure double-dragon seven pairs','清三龙七对':'Pure triple-dragon seven pairs','清十八罗汉':'Pure four kongs','清金钩钓':'Pure golden hook','中张':'All simples','扫底胡':'Last-tile self-draw','海底炮':'Last discard win','底分':'Base points','带杠胡':'Kong bonus','天胡':'Heavenly hand','地胡':'Earthly hand','金钩钩':'Golden hook','清金钩钩':'One-suit golden hook','清对':'One-suit triplets','清七对':'One-suit seven pairs','清龙七对':'One-suit dragon seven pairs','平胡':'Regular hand','门清':'Closed hand','豪华大对':'Four concealed sets','单钓':'Four melds and a single wait','全求人':'Four open melds','烂牌':'Ragged hand','七星烂':'Seven-star ragged hand','双豪七':'Double luxury seven pairs','三豪七':'Triple luxury seven pairs','尾牌包三':'Last-four liability','报听':'Declared ready','报听分摊':'Declared discarder: shared payments','一炮三响':'Triple discard win','翻醒':'Wake bonus','鸡胡':'Zero-faan hand','人和':'Win on dealer’s first discard','九莲宝灯':'Nine Gates','一条龙':'Pure Straight','三杠':'Three Kongs','四杠':'Four Kongs','十三幺':'Thirteen Orphans','大四喜':'Big Four Winds','小四喜':'Little Four Winds','大三元':'Big Three Dragons','小三元':'Little Three Dragons','十八罗汉':'Four Kongs','清幺九':'All Terminals','字一色':'All Honors','坎坎胡':'Four Concealed Pungs','混一色':'Half Flush','花幺':'Mixed Terminals and Honors','白板刻':'White Dragon Pung','发财刻':'Green Dragon Pung','红中刻':'Red Dragon Pung','门风刻':'Seat Wind Pung','圈风刻':'Prevailing Wind Pung','杠上杠':'Consecutive Kongs','抢杠上杠':'Robbing a Consecutive Kong','对对胡':'All triplets','清一色':'Pure one suit','七对':'Seven pairs','龙七对':'Dragon seven pairs','金钩钓':'Single-tile wait','杠上花':'Win after kong','杠上炮':'Discard after kong','抢杠':'Robbing a kong','海底':'Last tile','自摸':'Self-draw'},gen:'Roots × {n}',ghost:'Concealed quads × {n}',
 errors:{stale_action:'The table has advanced. Wait for the latest state and try again.',illegal_action:'That action is not legal here.',table_failed:'This table encountered an error. Please start a new table.',table_missing:'This table has expired. Please start a new table.',unsupported_variant:'This ruleset is not supported.',unsupported_policy:'This AI difficulty is not supported.',model_unavailable:'The standard AI is unavailable. Please choose Beginner.',engine_unavailable:'The riichi engine is not ready.',table_full:'All tables are busy. Please try again shortly.',round_starting:'The hand is being prepared. Please wait.',round_active:'Finish the current hand first.',invalid_request:'The request is not valid.',unknown_action:'This operation is not supported.',origin_mismatch:'The request origin does not match this table.',not_found:'This page does not exist.',server_error:'The service encountered an error. Please try again.',catalog_failed:'The service is not ready. Please refresh and try again.'},
},
'zh-CN':{
 darkReadyTag:'暗倒', declareDarkReady:'暗倒 · 打 {tile}', promptDarkReady:'已暗倒，可摸切、胡牌或开符合条件的杠。',
 promptPiao:'看牌前请先选择本局漂数。', piaoChoice:'{value} 漂', piaoTag:'{value} 漂',
 piaoTransfers:'压漂分 · {amounts}', horseTransfers:'买马分 · {amounts}', rankHorseSummary:'{name} · 马牌每家加 {points} 分',
 promptPaidReady:'已报听，只可摸切或胡牌。',
 chowTransfers:'吃香 · {amounts}', ponTransfers:'碰香 · {amounts}', paidReadyTransfers:'报听分 · {amounts}',
 openReadyTag:'亮倒', declareOpenReady:'亮倒 · 打 {tile}', pickOpenReadyTile:'选择标记的牌即可亮倒。',
 promptOpenReady:'已亮倒，只可摸切或胡牌。', openReadyTransfers:'荒庄首亮赔付 · {amounts}',
 promptLockedWin:'胡牌后只能打出摸到的牌，也可选择当前可用的胡牌或开杠。', flowKinds:{ron:'接炮',zimo:'自摸',rob_kong:'抢杠胡'}, winningTiles:'已胡牌张', winCount:'已胡 {count} 次', flowWinPayments:'{name} · {action} {tile} · {fan} 番 · {patterns} · {payments}',
 promptSameSuitExchange:'请选择同一花色的三张牌。', winningTile:'点炮胡张',
 flowerPigTransfers:'查花猪 · {amounts}', readyTransfers:'查大叫 · {amounts}', kongRefunds:'退税 · {amounts}',
 directions:{clockwise:'顺时针',counterclockwise:'逆时针',opposite:'对家'},
 homeLabel:'麻将道场首页',setupLabel:'开桌设置',tableLabel:'麻将牌桌',languageLabel:'语言',sourceLabel:'规则来源',
 brand:'麻将道场',rules:'玩法说明',edition:'开放试玩 · 02',eyebrow:'一张牌桌，多种乐趣',hero1:'开一桌，',hero2:'慢慢打。',intro:'选一个熟悉的玩法，和 AI 坐下来打一局。',intro2:'从第一张摸牌，到最后一笔结算。',coming:'地方玩法按规则家族逐步扩展，通过验证后开放。',chooseVariant:'选择玩法',searchRulesets:'搜索玩法',familyFilter:'规则家族',allFamilies:'全部家族',noRulesets:'没有找到匹配的玩法。',showMoreRulesets:'再显示 {count} 种',filteredRulesets:'显示 {count} 种，共 {total} 种',live:'{count} 种已开放',loadingCatalog:'正在准备牌桌…',choosePolicy:'选择对手',policyLabel:'AI 难度',model:'标准 · 道场模型',heuristic:'入门 · 牌效 AI',setupNote:'你与 {count} 位 AI 对局，可以随时托管本局。',start:'开始对局',footer:'TanyaoDojo · 从一局好牌开始',footer2:'完整规则 · 与 AI 对局',back:'← 返回大厅',inGame:'正在对局',currentRules:'当前规则',viewRules:'查看规则 ↗',helpTitle:'一手一手，慢慢来',help:'点选手牌，再点「打出」。吃碰杠胡等操作会在轮到你时出现。',autoplay:'托管本局',cancelAuto:'取消托管',connecting:'牌桌连接中…',connected:'● 牌桌已连接',reconnecting:'连接暂时中断，正在重试…',preparing:'准备开局',aiReady:'AI 正在就位',yourTurn:'轮到你了',thinking:'AI 思考中',handEnd:'本局结束',matchEnd:'半庄结束',matchComplete:'全场结束',matchHand:'第 {number} / {total} 局',matchWinner:'{total} 局打完，{names}获得首名。',firstPlace:'首名',tableError:'牌桌异常',tableClosed:'牌桌已关闭',autoState:'本局托管中',events:'牌桌记录',liveEvents:'本桌动态',privacy1:'只展示公开信息',privacy2:'未公开的手牌保持私密',before:'落座之前',gotIt:'知道了',closeRules:'关闭规则说明',resultEyebrow:'这一局的答案',handResult:'本局结算',next:'下一局 ↗',newTable:'再开一桌 ↗',returnLobby:'返回大厅',opening:'正在开桌',openingHint:'AI 和牌墙就绪后，就轮到你。',bonusSummary:'{name}：扎 {count} 码，中 {hits} 码，每份 {points} 分。',bonusTileHeading:'扎码',fishTile:'爆炸鱼',fishSummary:'{name} · 中 {hits} 条鱼 · 每份 {points} 分',winShare:'{name} · {fan} 番 · 每份 {points} 分 · {patterns}',winSharePoints:'{name} · 每份 {points} 分 · {patterns}',winPayments:'{name} · {fan} 番 · {patterns} · {payments}',tierPayments:'{name} · {tier} 类奇牌 · {patterns} · {payments}',pointPayments:'{name} · {patterns} · {payments}',wakeTiles:'醒牌指示',kongTransfers:'杠分 · {amounts}',dealerEcho:'抄庄 · {amounts}',wakeSummary:'{name}：看 {count} 张，中 {hits} 醒。',oneMaSummary:'{name}：码牌加 {extra} 分，每份合计 {points} 分。',wall:'牌墙剩 {count} 张',dora:'宝牌指示',wild:'百搭',pot:'供托 {count} × 1,000 点',honba:'{count} 本场',dealer:'庄',won:'已胡',riichi:'立直',ready:'已报听',void:'缺{suit}',concealed:'{count} 张暗牌',selectTile:'选择 {tile}',selectReadyTile:'选择 {tile}（可报听）',pickReadyTile:'选择标记的牌即可报听。',hand:'第 {number} 局',round:'{wind}{number}局',roundWindLabel:'{wind}圈',promptExchange:'选三张牌换给下家。',confirmExchange:'确认换三张',exchangeCount:'已选 {count} / 3 张',bilateralHorses:'{name} 胡 · 各家中马 {hits}',promptVoid:'先定缺：选择本局要舍弃的花色。',promptTurn:'轮到你了。点选一张手牌，或选择操作。',promptRob:'抢杠窗口：胡牌或过。',promptAuto:'AI 正在替你完成本局。',promptWait:'请稍候，其他玩家正在行动。',promptEnd:'本局已结算。',promptError:'牌桌遇到了异常，请返回大厅重开。',selected:'已选 {tile}',clickTile:'点击手牌选择',pickTile:'先选择手牌',discard:'打 {tile}',declareReady:'报听 · 打 {tile}',voidAction:'定缺 · {suit}',
 action:{gang_bu:'补杠',gang_an:'暗杠',gang_ming:'明杠',none:'过',pass:'过',reach:'立直',declare_ready:'报听',chi:'吃',pon:'碰',peng:'碰',daiminkan:'明杠',zhigang:'直杠',ankan:'暗杠',rob_kong:'抢杠胡',kakan:'加杠',bugang:'补杠',hora:'胡牌',ron:'荣和',zimo:'自摸',ryukyoku:'九种九牌流局'},
 wind:{E:'东',S:'南',W:'西',N:'北'},suits:['万','筒','条'],names:['你','青竹','白露','松风'],tileHonors:{E:'东',S:'南',W:'西',N:'北',P:'白',F:'发',C:'中'},red:'赤',
 messages:{win_total:'{actor} 胡牌共收 {points} 分{patterns}',dark_ready:'{actor} 暗倒，保留暗刻。',piao_complete:'三家压漂完成，开始打牌。',paid_ready:'{actor} 报听，其他两家各付两分。',open_ready:'{actor} 亮倒。',flow_win:'{actor} · {action} {tile} · {fan} 番{patterns}',repeat_win_end:'牌墙耗尽，已结算查叫。',exchange_direction:'已完成{direction}换牌，请定缺。',yunnan_draw:'牌墙耗尽，已结算查花猪、查大叫与退税。',hand_start:'新一局开始。',hand_start_void:'第 {number} 局开始，请选择定缺花色。',discard:'{actor} 打出 {tile}',void:'{actor} 定缺{suit}',action:'{actor} {action}',tsumo:'{actor} 自摸',ron:'{actor} 荣和 · {target} 放铳',win_fan:'{actor} 胡牌 · {fan} 番{patterns}',win_tier:'{actor} 胡牌 · {tier} 类奇牌{patterns}',win_points:'{actor} 胡牌 · 每份 {points} 分{patterns}',draw:'本局流局。',blood_end:'三家胡牌，血战结束。',sichuan_draw:'牌墙耗尽，已结算查大叫与退税。',hongzhong_draw:'牌墙耗尽，已结算的杠分保留。',rob_kong_win:'{actor} 抢 {target} 的补杠胡牌。',hongzhong_win:'{actor} 自摸胡牌。',one_ma_bonus:'一码全中：{tiles} · 每份加 {extra} 分。',bonus:'扎码：{tiles} · 中 {hits} 码。'},
 reason:{repeat_win_end:'牌墙耗尽，已结算胡牌、杠分和查叫',yunnan_draw:'牌墙耗尽，已结算查花猪、查大叫与退税',rob_concealed_win:'{names} 以十三幺抢暗杠胡牌',suited_self_draw:'{names} 自摸胡牌',suited_discard_win:'{names} 点炮胡牌',winners:'{names} 胡牌',draw:'流局结算',blood_end:'三家胡牌，血战结束',sichuan_draw:'牌墙耗尽，已结算查大叫与退税',hongzhong_draw:'牌墙耗尽，已结算的杠分保留',rob_kong_win:'{names} 抢杠胡牌',hongzhong_win:'{names} 自摸胡牌',four_red:'手持四张红中胡牌'},
 fans:{},gen:'根 × {n}',ghost:'鬼 × {n}',
 errors:{stale_action:'牌局已推进，请等待画面更新后再操作。',illegal_action:'这一步不是合法动作。',table_failed:'本桌遇到了异常，请重新开桌。',table_missing:'牌桌已过期，请重新开桌。',unsupported_variant:'不支持的玩法。',unsupported_policy:'不支持的 AI 难度。',model_unavailable:'当前标准 AI 暂不可用，请选择入门。',engine_unavailable:'立直引擎尚未就绪。',table_full:'当前牌桌已满，请稍后再试。',round_starting:'正在开局，请稍候。',round_active:'请先完成当前一局。',invalid_request:'请求格式不正确。',unknown_action:'未知操作。',origin_mismatch:'请求来源不匹配。',not_found:'没有这个页面。',server_error:'服务暂时遇到异常，请稍后重试。',catalog_failed:'服务暂未就绪，请刷新后重试。'},
}};
Object.assign(I18N.en, {
  winSealed:'Win blocked',
 wildcardFloatTag:'Float ×{count}', promptFloatCycle:'An opponent floated a wildcard. You can discard your draw, make a concealed kong or win.',
 feedLiability:'{caller} ← {provider} · three feeds', setupNoteOne:'You and one AI opponent. You can autoplay the hand.',
 wildcardIndicator:'Wildcard indicator', wildcardIndicatorTile:'Indicator', whiteSubstitute:'White Dragon plays as {tile}', whiteOptional:'White Dragon or natural {tile}', selectWildcard:'Select {tile} (wildcard)', flowers:'Flowers',
 tenWindsRun:'Ten Winds · {count}',
});
Object.assign(I18N['zh-CN'], {
  winSealed:'已封胡',
 wildcardFloatTag:'飘财 ×{count}', promptFloatCycle:'对手已飘财，你可摸切、暗杠或自摸。',
 feedLiability:'{caller} ← {provider} · 三摊承包', setupNoteOne:'你与一位 AI 单挑，可以随时托管本局。',
 wildcardIndicator:'翻出的财神', wildcardIndicatorTile:'财神指示牌', whiteSubstitute:'白板替 {tile}', whiteOptional:'白板也可作 {tile}', selectWildcard:'选择 {tile}（财神）', flowers:'补花',
 tenWindsRun:'十风 · {count}',
});
I18N.en.action.float_wildcard='Float wildcard';
I18N['zh-CN'].action.float_wildcard='飘财';
I18N.en.action.fangchong_win='Win on discard';
I18N['zh-CN'].action.fangchong_win='接炮胡';
I18N.en.action.ten_winds='Discard and win · Ten Winds';
I18N['zh-CN'].action.ten_winds='打出并胡 · 十风';
I18N.en.reason.ten_winds_win='{names} wins with Ten Winds';
I18N['zh-CN'].reason.ten_winds_win='{names} 十风胡牌';
I18N.en.action.eleven_winds='Discard and win · Eleven Winds';
I18N['zh-CN'].action.eleven_winds='打出并胡 · 十一风';
I18N.en.reason.eleven_winds_win='{names} wins with Eleven Winds';
I18N['zh-CN'].reason.eleven_winds_win='{names} 十一风胡牌';
I18N.en.dealerStake='Base {points}';
I18N['zh-CN'].dealerStake='庄底 {points}';
I18N.en.messages.wildcard_float='{actor} floated a wildcard · {count} in a row.';
I18N['zh-CN'].messages.wildcard_float='{actor} 飘财，已连飘 {count} 次。';
Object.assign(I18N.en.fans, {
 '爆头':'Burst', '财飘':'Wildcard float', '双财飘':'Double float', '三财飘':'Triple float',
 '杠开':'Kong draw', '二连杠开':'Two consecutive kong draws', '三连杠开':'Three consecutive kong draws', '四连杠开':'Four consecutive kong draws',
 '杠暴':'Kong burst', '二连杠暴':'Two-kong burst', '三连杠暴':'Three-kong burst', '四连杠暴':'Four-kong burst',
 '杠飘':'Kong then float', '飘杠':'Float then kong', '七对子':'Seven pairs', '七客':'Seven-pairs burst',
 '豪华七对':'Luxury seven pairs', '双豪华七对':'Double luxury seven pairs', '三豪华七对':'Triple luxury seven pairs',
 '清七对':'Natural seven pairs', '豪华清七对':'Natural luxury seven pairs',
 '双豪华清七对':'Natural double luxury seven pairs', '三豪华清七对':'Natural triple luxury seven pairs',
 '十风':'Ten Winds', '起手十风':'Opening Ten Winds', '乱风':'Unstructured honors', '正风':'All honor pungs', '豪杠开':'Concealed kong draw',
 '大对子':'All pungs', '大碰胡':'All pungs', '大吊车':'Four sets and a pair', '全字':'All honors',
 '财神头':'Wildcard head', '三财神':'Three opening wildcards', '四财神':'Four wildcards', '无花无财神':'No flowers or wildcards', '正规十三幺':'Natural thirteen orphans',
 '十三不搭':'Isolated honors', '真十三不搭':'Strict isolated honors', '七风':'Seven distinct honors', '全求人':'Four exposed sets',
 '十一风':'Eleven Winds', '三杠全缴':'Three-kong declaration', '无财神':'No wildcards', '七嵌':'Seven Gaps',
 '一杠一达':'One kong and five pairs', '二杠二达':'Two kongs and three pairs', '三杠三达':'Three kongs and one pair',
 '全风向':'All honors', '抛龙':'Wildcard-head declaration',
 '位花':'Seat flower', '四花':'Four matching flowers', '八花':'Eight flowers', '靠柄':'Edge wait', '嵌档':'Inside wait',
 '排胡':'All sequences', '门风刻杠':'Seat-wind set', '字牌刻杠':'Dragon set', '清牌':'No wildcards in hand',
 '无得':'No wildcards', '得还原':'All wildcards match their face',
 '四风聚齐':'Four Winds', '门风或字牌刻杠':'Seat-wind or dragon set',
 '财神会':'Wild Gathering', '杠上开花':'Kong replacement win', '海底捞月':'Last draw win', '点杠杠开':'Win on a direct-kong replacement',
 '无百搭':'No wildcards', '三百搭':'Three wildcards', '全老头':'All honors', '杠开花开':'Kong or flower replacement win',
 '黄番':'Draw bonus', '老头黄':'Honor wildcard bonus', '豹子黄':'Doubles bonus', '百搭连庄':'Repeat wildcard bonus',
 '十三百搭':'Separated hand', '真百搭':'Separated hand with seven honors', '烂百搭':'Flexible separated hand',
 '三财倒':'Three-wildcard self-draw', '七风倒':'Seven-honor self-draw', '全风字':'All honors',
 '全风字大碰胡':'All-honor pungs', '全风字七对':'All-honor seven pairs', '混碰':'Mixed-suit pungs',
 '清碰':'Pure-suit pungs', '混七对':'Mixed-suit seven pairs', '清七对':'Pure-suit seven pairs',
 '不正规十三幺':'Wildcard thirteen orphans', '正规七风':'Natural Seven Winds', '不正规七风':'Wildcard Seven Winds',
});
Object.assign(I18N.en.tileHonors, {F1:'Spring', F2:'Summer', F3:'Autumn', F4:'Winter',
 F5:'Plum', F6:'Orchid', F7:'Bamboo', F8:'Chrysanthemum'});
Object.assign(I18N['zh-CN'].tileHonors, {F1:'春', F2:'夏', F3:'秋', F4:'冬', F5:'梅', F6:'兰', F7:'竹', F8:'菊'});
I18N.en.flowerBonus='Flowers × {n}';
I18N['zh-CN'].flowerBonus='花牌 × {n}';
I18N.en.baseHu='Base {n} hu';
I18N['zh-CN'].baseHu='底 {n} 胡';
I18N.en.taiCount='{n} tai';
I18N['zh-CN'].taiCount='{n} 台';
Object.assign(I18N.en, {
  followTransfers:'Opening follow-tile points · {amounts}',
  bankExhausted:'{names} could not cover the full bill. The match is over.',
  liabilityPrompt:'Rivals formed a liability link. Stay in the hand or fold to sit out scoring.',
  continueHand:'Stay in', foldHand:'Fold this hand', foldedSeat:'Folded', yellowFaan:'Carried bonus faan {n}',
  countedFlowers:'{n} flower points',
  kongRefundTransfers:'Kong points returned · {amounts}',
  promptBuyBase:'Choose whether to buy the base before seeing your tiles.', buyBase:'Buy the base', skipBase:'Play without buying',
  boughtBase:'Base bought', buyBaseTransfers:'Buy-in points · {amounts}', chongTiles:'Chong tiles',
  linhaiChongSummary:'{name} · {hits} hits from {count} tiles · {points} per share',
  faanCount:'{n} faan', taizhouFreshStage:'Fresh-tile liability phase', taizhouValues:'Hand values · {amounts}',
  taizhouLiability:'{name} pays all bills · {reason}', taizhouBill:'{payer} pays {receiver} {points} hu',
  taizhou_late_fresh_tile:'late fresh discard', taizhou_pure_suit_feeds:'three chows for a pure suit',
  taizhou_fresh_dragon:'fresh dragon discard',
});
Object.assign(I18N['zh-CN'], {
  followTransfers:'开局跟牌分 · {amounts}',
  bankExhausted:'{names} 的余额不足以付清账单，整场结束。',
  liabilityPrompt:'其他牌手形成三摊承包。继续打牌，或退出本局计分。',
  continueHand:'继续打牌', foldHand:'退出本局计分', foldedSeat:'已退出计分', yellowFaan:'累计黄番 {n}',
  countedFlowers:'花数 {n}',
  kongRefundTransfers:'退杠分 · {amounts}',
  promptBuyBase:'看牌前请先选择是否买底。', buyBase:'买底', skipBase:'不买底',
  boughtBase:'已买底', buyBaseTransfers:'买底分 · {amounts}', chongTiles:'抓冲牌',
  linhaiChongSummary:'{name} · {count} 张中 {hits} 冲 · 每份 {points} 分',
  faanCount:'{n} 番', taizhouFreshStage:'生牌阶段', taizhouValues:'各家牌点 · {amounts}',
  taizhouLiability:'{name} 承包 · {reason}', taizhouBill:'{payer} 付 {receiver} {points} 胡',
  taizhou_late_fresh_tile:'生牌阶段点炮', taizhou_pure_suit_feeds:'三吃清一色', taizhou_fresh_dragon:'中发白生张',
});
I18N.en.messages.buy_base_complete='Buy-in choices are set. Play begins.';
I18N['zh-CN'].messages.buy_base_complete='买底已选定，开始打牌。';
Object.assign(I18N.en.fans, {
  '明杠':'Exposed kong', '暗杠':'Concealed kong', '碰碰胡':'All pons', '抢杠胡':'Robbed added kong',
  '小吊车':'Pair wait with a chow', '超豪华七对':'Super luxury seven pairs',
  '压档':'Unique inside wait', '交':'Exceptional hand', '对对胡':'All pons', '通天':'Pure straight',
  '四核':'Four undeclared copies', '双四核':'Two undeclared quads', '双八支':'Eight suit tiles plus two off-suit kongs',
  '三碰包牌':'Third-pon liability', '四红中':'Four Red Dragons',
});
I18N.en.branchCount='{n} tiles in one suit';
I18N['zh-CN'].branchCount='{n} 支';
I18N.en.longestSuit='Best suit {n} · Need 8';
I18N['zh-CN'].longestSuit='最多一门 {n} 支 · 需 8 支';
Object.assign(I18N.en, {
  goldIndicator:'Gold wildcard', dealerTai:'Dealer +{n} tai', goldCount:'Gold tiles × {n}', passedWinBlock:'Win blocked after pass',
  promptGold:'Choose an opening win or pass. A tile on a gold claim is the dealer’s discard.',
  promptFlowerWin:'Declare the flower win or keep playing.',
});
Object.assign(I18N['zh-CN'], {
  goldIndicator:'翻出的金', dealerTai:'庄加 {n} 台', goldCount:'金牌 × {n}', passedWinBlock:'过水禁胡',
  promptGold:'可选择开局胡牌或过。抢金按钮上的牌是庄家要换出的牌。',
  promptFlowerWin:'可胡花或继续打牌。',
});
Object.assign(I18N.en.action, {claim_gold:'Claim gold', three_gold:'Win with three golds', flower_hu:'Win with flowers'});
Object.assign(I18N['zh-CN'].action, {claim_gold:'抢金', three_gold:'三金倒', flower_hu:'胡花'});
Object.assign(I18N.en.action, {opening_heaven:'Win on the opening deal', travel_gold:'Travel with gold'});
Object.assign(I18N['zh-CN'].action, {opening_heaven:'天胡', travel_gold:'游金'});
Object.assign(I18N.en, {
  travelStatus:'{name} · Gold travel {n}', goldDiscardStatus:'{name} discarded gold · Win restrictions apply',
  flowerKongTransfers:'Flower kong points · {amounts}',
});
Object.assign(I18N['zh-CN'], {
  travelStatus:'{name} · 游金第 {n} 阶', goldDiscardStatus:'{name} 打金 · 本轮胡牌受限',
  flowerKongTransfers:'花杠分 · {amounts}',
});
I18N.en.messages.travel_gold='{actor} discarded {tile} and declared gold travel {level}.';
I18N['zh-CN'].messages.travel_gold='{actor} 打出 {tile}，进入第 {level} 阶游金。';
Object.assign(I18N.en.fans, {'游金':'Gold travel', '双游':'Double gold travel', '三游':'Triple gold travel'});
I18N.en.action.four_gold='Win with four golds';
I18N['zh-CN'].action.four_gold='四金倒';
I18N.en.reason.four_gold_win='{names} won with four gold tiles.';
I18N['zh-CN'].reason.four_gold_win='{names} 四金倒胡牌';
I18N.en.fixedGold='Fixed gold';
I18N['zh-CN'].fixedGold='固定金';
Object.assign(I18N.en, {privateBao:'Your revealed bao', playerBao:'{name}’s bao', promptBaoPeek:'Your bao completes the hand. Win now or take your draw.', extendEgg:'Extend {egg}'});
Object.assign(I18N['zh-CN'], {privateBao:'你看到的宝牌', playerBao:'{name} 的宝牌', promptBaoPeek:'宝牌正好能胡。可对宝，也可继续摸牌。', extendEgg:'补{egg}'});
Object.assign(I18N.en.action, {ready_chi:'Chow and declare ready', ready_pon:'Pon and declare ready', win_bao:'Win by matching bao',
  egg_winds:'Wind egg', egg_dragons:'Dragon egg', egg_ones:'Ones egg', egg_nines:'Nines egg', rob_egg_pon:'Rob the egg extension with pon'});
Object.assign(I18N['zh-CN'].action, {ready_chi:'听吃', ready_pon:'听碰', win_bao:'对宝',
  egg_winds:'旋风蛋', egg_dragons:'喜蛋', egg_ones:'幺蛋', egg_nines:'九蛋', rob_egg_pon:'抢碰补蛋'});
I18N.en.reason.bao_peek_win='{names} won by matching the revealed bao.';
I18N['zh-CN'].reason.bao_peek_win='{names} 对宝胡牌';
I18N.en.reason.tong_bao_win='{names} won with the initial bao.';
I18N['zh-CN'].reason.tong_bao_win='{names} 通宝胡牌';
I18N.en.action.tong_bao='Win with initial bao';
I18N['zh-CN'].action.tong_bao='通宝';
I18N.en.baoRoll='Bao roll · {n}';
I18N['zh-CN'].baoRoll='掷宝 · {n} 点';
I18N.en.messages.declared_ready='{actor} declared ready.';
I18N['zh-CN'].messages.declared_ready='{actor} 报听。';
Object.assign(I18N.en.fans, {'摸宝':'Drawn bao', '对宝':'Matched bao', '摸大宝':'Own bao draw', '摸小宝':'Another player’s bao draw',
  '大宝中宝':'Own bao matches the middle wait', '小宝中宝':'Another player’s bao matches the middle wait', '卡当':'Single middle wait',
  '夹张':'Middle wait', '边张':'Edge wait', '飘胡':'All pungs'});
Object.assign(I18N.en.fans, {'单贴':'Single tile for the eyes', '夹胡':'Middle wait', '边胡':'Edge wait', '七小对':'Seven pairs',
  '通宝':'Initial bao matches the wait', '搂宝':'Drawn bao', '闭门':'Closed hand', '飘中飘':'Closed all pungs',
  '带幺鸡':'Contains One Bamboo', '站立':'Standing hand', '杠翻':'After a kong draw', '手把一':'Four declared sets',
  '四归一':'Four copies outside a kong'});
I18N.en.reason.third_travel_win='{names} completed the third gold travel.';
I18N['zh-CN'].reason.third_travel_win='{names} 三游胡牌';
Object.assign(I18N.en, {tripletPoints:'Set points × {n}', countedKongs:'Kong points × {n}', waterCount:'Water points × {n}', dealerStreak:'Dealer repeats × {n}'});
Object.assign(I18N['zh-CN'], {tripletPoints:'刻分 × {n}', countedKongs:'杠分 × {n}', waterCount:'水数 × {n}', dealerStreak:'连庄 × {n}'});
Object.assign(I18N.en.fans, {'无水胡':'No water', '一水胡':'One water'});
I18N.en.whiteFlower='White Dragons are replacement flowers';
I18N['zh-CN'].whiteFlower='白板作花补牌';
Object.assign(I18N.en.fans, {'二金倒':'Two-gold pair', '三金游':'Three-gold travel', '白胡':'Self-draw without gold'});
I18N.en.goldBalanceTransfers='Gold and kong differences · {amounts}';
I18N['zh-CN'].goldBalanceTransfers='未胡金杠差 · {amounts}';
Object.assign(I18N.en.fans, {'四金倒':'Four gold tiles', '金坎':'Gold triplet', '金杠':'Declared gold kong'});
Object.assign(I18N.en.reason, {gold_claim:'{names} claimed the opening gold.', three_gold_win:'{names} won with three gold tiles.',
  flower_hu:'{names} won with flowers.'});
Object.assign(I18N['zh-CN'].reason, {gold_claim:'{names} 开局抢金胡牌', three_gold_win:'{names} 三金倒胡牌', flower_hu:'{names} 胡花'});
Object.assign(I18N.en.fans, {
  '抢金':'Opening gold claim', '三金倒':'Three gold tiles', '金雀':'Gold pair', '金龙':'Gold pung',
  '无花无杠':'No flowers or kongs', '一张花':'One flower', '八仙过海':'All eight flowers', '七抢一':'Seven flowers claim the eighth',
  '门清自摸':'Closed self-draw', '全求人':'Five exposed sets', '独听':'Single wait', '正花':'Seat flower',
  '花杠':'Complete flower group', '门风':'Seat wind', '圈风':'Round wind', '青发':'Green Dragon', '红中':'Red Dragon',
});
let LANGUAGE='en';
try{const saved=localStorage.getItem('tanyao_language');if(saved&&I18N[saved])LANGUAGE=saved;}catch{}
const tr=(key,args={})=>{let s=I18N[LANGUAGE][key]??I18N.en[key]??key;return String(s).replace(/\{(\w+)\}/g,(_,k)=>args[k]??'');};
const tpl=(text,args={})=>String(text).replace(/\{(\w+)\}/g,(_,k)=>args[k]??'');
const playerName=i=>I18N[LANGUAGE].names[i]??String(i);
const tileText=t=>I18N[LANGUAGE].tileHonors[t]||(LANGUAGE==='en'?`${t.endsWith('r')?'Red ':''}${t[0]} ${I18N.en.suits['mps'.indexOf(t[1])]}`:`${t[0]}${I18N['zh-CN'].suits['mps'.indexOf(t[1])]}${t.endsWith('r')?'赤':''}`);
const actionText=k=>I18N[LANGUAGE].action[k]??k;
const countedPatterns=[['根x','gen'],['鬼x','ghost'],['花牌x','flowerBonus'],['底胡x','baseHu'],['台数x','taiCount'],['番数x','faanCount'],['花数x','countedFlowers'],['八支x','branchCount'],['金牌x','goldCount'],['刻分x','tripletPoints'],['杠分x','countedKongs'],['水数x','waterCount']];
function patternText(pattern){
  const match=countedPatterns.find(([prefix])=>pattern.startsWith(prefix));
  return match?tr(match[1],{n:pattern.slice(match[0].length)}):(I18N[LANGUAGE].fans[pattern]||pattern);
}
function localVariant(v){return {...v,...(v.translations?.[LANGUAGE]||{})};}
function messageText(m){if(!m.code)return m.text||'';const p={...m.params};for(const k of ['actor','target'])if(p[k]!==undefined)p[k]=playerName(p[k]);if(p.tile)p.tile=tileText(p.tile);if(p.direction)p.direction=I18N[LANGUAGE].directions[p.direction];if(p.suit!==undefined)p.suit=I18N[LANGUAGE].suits[p.suit];if(p.action)p.action=m.code==='flow_win'?I18N[LANGUAGE].flowKinds[p.action]:actionText(p.action);if(p.tiles)p.tiles=p.tiles.map(tileText).join(', ');if(p.patterns){p.patterns=p.patterns.map(patternText).join(' · ');p.patterns=p.patterns?' · '+p.patterns:'';}return tpl(I18N[LANGUAGE].messages[m.code]??I18N.en.messages[m.code]??m.code,p);}
