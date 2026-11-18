'use strict';
const $=id=>document.getElementById(id);
let catalog=[],variant='riichi',game=null,selected=null,loading=false,pollTimer=null,lastRevision=-1,lastStatus='',requestBusy=false;
const CATALOG_PAGE_SIZE=8;
const exchangeSelection=new Set();
let visibleLimit=CATALOG_PAGE_SIZE;
const img=t=>{if(/^F[1-8]$/.test(t))return text('span',tileText(t),'flower-face');const i=document.createElement('img');i.src=`/tiles/${encodeURIComponent(t)}.svg`;i.alt=tileText(t);i.draggable=false;return i;};
function text(tag,value,cls=''){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;}
function showError(id,e){const node=$(id);node.dataset.errorCode=e.code||'server_error';node.textContent=I18N[LANGUAGE].errors[node.dataset.errorCode]||e.message||I18N[LANGUAGE].errors.server_error;}
function optionText(o){if(o.kind==='liability_choice')return tr(o.value?'continueHand':'foldHand');if(o.kind==='buy_base')return tr(o.value?'buyBase':'skipBase');if(o.kind==='piao')return tr('piaoChoice',{value:o.value});if(o.win_type)return actionText(o.win_type);if(['discard','dahai'].includes(o.kind))return tr('discard',{tile:tileText(o.tiles[0])});if(o.kind==='void')return tr('voidAction',{suit:I18N[LANGUAGE].suits[o.suit]});return actionText(o.kind);}
function applyLanguage(){document.documentElement.lang=LANGUAGE;document.title='TanyaoDojo · '+tr('brand');for(const e of document.querySelectorAll('[data-language]'))e.value=LANGUAGE;for(const e of document.querySelectorAll('[data-i18n]'))e.textContent=tr(e.dataset.i18n);for(const e of document.querySelectorAll('[data-i18n-aria]'))e.setAttribute('aria-label',tr(e.dataset.i18nAria));for(const e of document.querySelectorAll('[data-wind]'))e.textContent=LANGUAGE==='en'?e.dataset.wind:I18N[LANGUAGE].wind[e.dataset.wind];$('policy').setAttribute('aria-label',tr('policyLabel'));$('close-rules').setAttribute('aria-label',tr('closeRules'));if(catalog.length)renderLobby();if(game){const v=currentVariant();$('room-title').textContent=v.name;$('room-subtitle').textContent=v.subtitle;render(game,true);}if($('rules-dialog').open)fillRules();for(const e of document.querySelectorAll('[data-error-code]'))if(e.textContent)e.textContent=I18N[LANGUAGE].errors[e.dataset.errorCode]||e.textContent;}
for(const picker of document.querySelectorAll('[data-language]'))picker.onchange=()=>{LANGUAGE=picker.value;try{localStorage.setItem('tanyao_language',LANGUAGE);}catch{}applyLanguage();};
applyLanguage();
async function api(url,body){const r=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let d;try{d=await r.json();}catch{const e=Error(tr('reconnecting'));e.code='server_error';throw e;}if(!r.ok){const e=Error(d.error||tr('reconnecting'));e.code=d.error_code||'server_error';throw e;}return d;}
function currentVariant(){const v=catalog.find(v=>v.id===variant);return v?localVariant(v):null;}
function chooseVariant(id){variant=id;$('policy').value='';renderLobby();}
function searchText(value){return value.normalize('NFKD').replace(/\p{M}/gu,'').toLowerCase();}

function renderLobby(){
  const family=$('catalog-family');
  const familyId=family.value;
  const families=new Map(catalog.map(raw=>[raw.family_id,localVariant(raw).family]));
  const options=[new Option(tr('allFamilies'),'')];
  for(const [id,label] of families)options.push(new Option(label,id));
  family.replaceChildren(...options);
  family.value=familyId;
  $('catalog-filters').hidden=false;

  const query=searchText($('catalog-search').value).trim().split(/\s+/).filter(Boolean);
  const matches=catalog.filter(raw=>{
    if(familyId&&raw.family_id!==familyId)return false;
    const names=[raw.id,raw.name,raw.family,raw.subtitle,...(raw.aliases||[])];
    for(const locale of Object.values(raw.translations||{}))names.push(locale.name,locale.family,locale.subtitle);
    const index=searchText(names.join(' '));
    return query.every(word=>index.includes(word));
  });

  const activeIndex=matches.findIndex(entry=>entry.id===variant);
  const previousPolicy=activeIndex>=0?$('policy').value:'';
  if(activeIndex<0&&matches.length)variant=matches[0].id;
  const cards=$('variant-cards');
  const pageEnd=Math.max(visibleLimit,activeIndex+1);
  cards.replaceChildren();
  for(const raw of matches.slice(0,pageEnd)){
    const v=localVariant(raw);
    const b=text('button','','variant-card'+(v.id===variant?' selected':''));
    b.type='button';b.dataset.variantId=v.id;
    b.setAttribute('aria-pressed',v.id===variant);
    b.append(text('span',v.badge,'badge'),text('h3',v.name),text('p',v.subtitle),text('span',v.id===variant?'✓':'','check'));
    b.onclick=()=>{chooseVariant(v.id);cards.querySelector('.selected')?.focus();};
    cards.append(b);
  }
  if(!matches.length)cards.append(text('p',tr('noRulesets'),'catalog-empty'));
  const more=$('catalog-more');
  more.hidden=pageEnd>=matches.length;
  more.textContent=tr('showMoreRulesets',{count:Math.min(CATALOG_PAGE_SIZE,Math.max(0,matches.length-pageEnd))});
  more.onclick=()=>{
    visibleLimit=pageEnd+CATALOG_PAGE_SIZE;
    renderLobby();
    cards.children[pageEnd]?.focus();
  };
  $('ruleset-count').textContent=matches.length===catalog.length?tr('live',{count:catalog.length}):tr('filteredRulesets',{count:matches.length,total:catalog.length});

  const policy=$('policy');
  const opponentCount = (currentVariant()?.seat_count || 4) - 1;
  $('setup-note').textContent=matches.length?tr(opponentCount===1?'setupNoteOne':'setupNote',{count:opponentCount}):'';
  policy.replaceChildren();
  if(matches.length){
    const v=currentVariant();
    for(const p of v.policies)if(p.available)policy.append(new Option(tr(p.id),p.id));
    policy.value=[...policy.options].some(o=>o.value===previousPolicy)?previousPolicy:v.default_policy;
  }
  policy.disabled=!policy.value;
  $('start-button').disabled=!policy.value;
}

function filterRulesets(){visibleLimit=CATALOG_PAGE_SIZE;renderLobby();}
$('catalog-search').addEventListener('input',filterRulesets);
$('catalog-family').addEventListener('change',filterRulesets);
function fillRules(){const v=currentVariant();if(!v)return;$('rules-title').textContent=v.name;$('rules-intro').textContent=v.description;$('rules-list').replaceChildren(...v.rules.map(r=>text('li',r)));for(const source of v.sources||[]){if(!source.url.startsWith('https://'))continue;const li=document.createElement('li');const a=text('a',tr('sourceLabel')+' ↗');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';li.append(a);$('rules-list').append(li);}}
function showRules(){fillRules();if(currentVariant())$('rules-dialog').showModal();}
$('rules-button').onclick=showRules;$('room-rules').onclick=showRules;
for(const id of ['close-rules','rules-ok'])$(id).onclick=()=>$('rules-dialog').close();
async function start(){if(loading||!currentVariant()||!$('policy').value)return;loading=true;$('loading').classList.remove('hidden');$('lobby-error').textContent='';try{if(game)await api(`/api/game/${game.id}/leave`,{}).catch(()=>{});const s=await api('/api/game',{variant,policy:$('policy').value});game=null;lastRevision=-1;lastStatus='';selected=null;sessionStorage.setItem('tanyao_session',s.id);$('lobby').classList.add('hidden');$('room').classList.remove('hidden');$('footer').classList.add('hidden');$('room-title').textContent=currentVariant().name;$('room-subtitle').textContent=currentVariant().subtitle;render(s);schedulePoll();}catch(e){showError('lobby-error',e);}finally{loading=false;$('loading').classList.add('hidden');}}
$('start-button').onclick=start;
async function leave(){if(game)await api(`/api/game/${game.id}/leave`,{}).catch(()=>{});game=null;clearTimeout(pollTimer);sessionStorage.removeItem('tanyao_session');$('result-dialog').close();$('room').classList.add('hidden');$('lobby').classList.remove('hidden');$('footer').classList.remove('hidden');lastRevision=-1;}
$('leave-button').onclick=leave;$('result-leave').onclick=leave;
function schedulePoll(){clearTimeout(pollTimer);if(game&&!['closed','error','finished'].includes(game.status))pollTimer=setTimeout(poll,450);}
async function poll(){if(!game)return;const sid=game.id;try{const s=await api(`/api/game/${sid}`);if(game?.id===sid){$('connection').textContent=tr('connected');render(s);}}catch(e){if(e.code==='table_missing'&&game?.id===sid){render({...game,status:'closed',error_code:'table_missing',choices:[],revision:lastRevision+1});}else $('connection').textContent=tr('reconnecting');}schedulePoll();}
function render(s,force=false){if(game?.id===s.id&&s.revision<lastRevision)return;if(!force&&s.revision===lastRevision&&`${s.status}:${s.automatic}`===lastStatus)return;const changed=s.revision!==lastRevision;game=s;lastRevision=s.revision;lastStatus=`${s.status}:${s.automatic}`;if(changed){selected=null;exchangeSelection.clear();}const table=s.table;if(!table.players)return;
$('table-round').textContent=(table.hand_number?tr(table.match_hands?'matchHand':'hand',{number:table.hand_number,total:table.match_hands}):tr('round',{wind:I18N[LANGUAGE].wind[table.round_wind]||'East',number:table.round_number||1}))+(table.show_winds?' · '+tr('roundWindLabel',{wind:I18N[LANGUAGE].wind[table.round_wind]}):'')+(table.honba?' · '+tr('honba',{count:table.honba}):'');
$('table-state').textContent=s.status==='your_turn'&&!s.automatic?tr('yourTurn'):s.status==='round_end'?tr('handEnd'):s.status==='finished'?tr(table.match_hands?'matchComplete':'matchEnd'):s.status==='error'?tr('tableError'):s.status==='closed'?tr('tableClosed'):s.automatic?tr('autoState'):tr('thinking');
$('remaining').textContent=tr('wall',{count:table.remaining});$('pot').textContent=table.pot?tr('pot',{count:table.pot}):'';
$('dora').replaceChildren();if(table.dora.length){$('dora').append(text('span',tr('dora')));for(const t of table.dora)$('dora').append(img(t));}if(table.winning_tiles?.length){$('dora').append(text('span',tr('winningTiles'),'winning-tile-label'));for(const t of table.winning_tiles)$('dora').append(img(t));}
if(table.wildcard){
  if(table.wildcard_indicator) $('dora').append(text('span',tr('wildcardIndicatorTile')),img(table.wildcard_indicator));
  $('dora').append(text('span',tr(table.wildcard_indicated&&!table.wildcard_indicator?'wildcardIndicator':'wild')),img(table.wildcard));
  if(table.white_substitute) $('dora').append(text('span',tr('whiteSubstitute',{tile:tileText(table.white_substitute)}),'white-substitute'));
  if(table.white_optional) $('dora').append(text('span',tr('whiteOptional',{tile:tileText(table.white_optional)}),'white-substitute'));
}
if(table.fresh_stage) $('dora').append(text('span',tr('taizhouFreshStage'),'tag'));
if(table.yellow_faan) $('dora').append(text('span',tr('yellowFaan',{n:table.yellow_faan}),'tag'));
$('table').classList.toggle('two-player',table.players.length===2);
$('table').classList.toggle('three-player',table.players.length===3);
$('seat-2').hidden=table.players.length<3;
$('seat-3').hidden=table.players.length<4;
for(const p of table.players)renderSeat(p,table,s);
for(const [caller, provider] of table.feed_liabilities || []){
  $('dora').append(text('span',tr('feedLiability',{caller:playerName(caller),provider:playerName(provider)}),'tag'));
}
$('human-river').replaceChildren(river(table.players[0].river));
$('events').replaceChildren(...s.messages.map(x=>text('li',messageText(x),x.kind)));$('events').scrollTop=$('events').scrollHeight;
$('auto-button').textContent=s.automatic?tr('cancelAuto'):tr('autoplay');$('auto-button').disabled=['round_end','finished','error','closed'].includes(s.status);
if(s.error_code||s.status==='closed'){$('room-error').textContent=I18N[LANGUAGE].errors[s.error_code||'table_missing'];}else if(changed){$('room-error').textContent='';delete $('room-error').dataset.errorCode;}renderActions();
if(s.result&&['round_end','finished'].includes(s.status))showResult(s.result);}
function river(items){const r=document.createElement('div');r.className='river';for(const item of items){const e=img(item.tile);if(item.called)e.classList.add('called');if(item.riichi||item.ready)e.classList.add('reach-tile');r.append(e);}return r;}
function exchangeTile(tile, index, snapshot) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'hand-tile legal';
  button.dataset.exchangeIndex = index;
  button.classList.toggle('selected', exchangeSelection.has(index));
  button.disabled = snapshot.status !== 'your_turn' || snapshot.automatic || requestBusy;
  button.setAttribute('aria-label', tr('selectTile', {tile: tileText(tile)}));
  button.setAttribute('aria-pressed', exchangeSelection.has(index));
  button.append(img(tile));
  button.onclick = () => {
    if (exchangeSelection.has(index)) exchangeSelection.delete(index);
    else if (exchangeSelection.size < 3) exchangeSelection.add(index);
    for (const tileButton of document.querySelectorAll('[data-exchange-index]')) {
      const chosen = exchangeSelection.has(Number(tileButton.dataset.exchangeIndex));
      tileButton.classList.toggle('selected', chosen);
      tileButton.setAttribute('aria-pressed', chosen);
    }
    renderActions();
  };
  return button;
}
function exchangeContains(option, tiles) {
  const remaining = option.tiles.slice();
  for (const tile of tiles) {
    const index = remaining.indexOf(tile);
    if (index < 0) return false;
    remaining.splice(index, 1);
  }
  return true;
}
function renderSeat(p,table,s){const seat=$(`seat-${p.id}`);seat.replaceChildren();const title=text('div','','seat-title'+(table.current===p.id?' active':''));title.append(text('span',playerName(p.id)),text('span',p.score.toLocaleString(),'score'));if(s.variant==='riichi'||table.show_winds)title.append(text('span',I18N[LANGUAGE].wind['ESWN'[(p.id-table.dealer+table.players.length)%table.players.length]],'tag'));if(p.id===table.dealer)title.append(text('span',tr('dealer'),'tag'));if(p.void_suit!==undefined&&p.void_suit!==null)title.append(text('span',tr('void',{suit:I18N[LANGUAGE].suits[p.void_suit]}),'tag'));if(p.won)title.append(text('span',p.win_count?tr('winCount',{count:p.win_count}):tr('won'),'tag won'));if(p.riichi)title.append(text('span',tr('riichi'),'tag reach'));if(p.open_ready)title.append(text('span',tr(p.dark_ready?'darkReadyTag':'openReadyTag'),'tag reach'));if(p.folded)title.append(text('span',tr('foldedSeat'),'tag'));if(p.bought_base)title.append(text('span',tr('boughtBase'),'tag'));if(p.piao!==undefined)title.append(text('span',tr('piaoTag',{value:p.piao}),'tag'));if(p.ready)title.append(text('span',tr('ready'),'tag reach'));seat.append(title);
if(p.id===table.dealer&&table.dealer_multiplier>1) title.append(text('span',`×${table.dealer_multiplier}`,'tag'));
if(p.id===table.dealer&&table.dealer_stake) title.append(text('span',tr('dealerStake',{points:table.dealer_stake}),'tag'));
if(p.wildcard_floats) title.append(text('span',tr('wildcardFloatTag',{count:p.wildcard_floats}),'tag reach'));
if(p.ten_winds_run) title.append(text('span',tr('tenWindsRun',{count:p.ten_winds_run}),'tag reach'));
if(p.flowers?.length){
  const flowers=text('div','','flower-row');
  flowers.setAttribute('aria-label',tr('flowers'));
  flowers.append(text('span',tr('flowers')),...p.flowers.map(img));
  seat.append(flowers);
}
if(p.id===0){const row=document.createElement('div');row.className='hand';const hand=p.hand.slice();if(table.drawn){const i=hand.lastIndexOf(table.drawn);if(i>=0)hand.push(...hand.splice(i,1));}hand.forEach((t,i)=>{if(table.phase==='exchange'){row.append(exchangeTile(t,i,s));return;}const options=s.choices.filter(x=>['discard','dahai'].includes(x.kind)&&x.tiles[0]===t);const b=document.createElement('button');b.className='hand-tile'+(options.length?' legal':'')+(table.drawn&&i===hand.length-1?' drawn':'')+(options.length&&options[0].id===selected?' selected':'');b.disabled=!options.length||s.automatic||requestBusy;const canDeclare=s.choices.some(o=>o.kind==='declare_ready'&&o.tiles[0]===t);b.classList.toggle('ready-discard',canDeclare);const indicatedWild=(table.wildcard_indicated||table.wildcard_fixed)&&t===table.wildcard;b.classList.toggle('is-wildcard',!!indicatedWild);b.setAttribute('aria-label',tr(indicatedWild?'selectWildcard':canDeclare?'selectReadyTile':'selectTile',{tile:tileText(t)}));b.append(img(t));if(options.length){b.dataset.option=options[0].id;b.onclick=()=>{selected=options[0].id;for(const x of document.querySelectorAll('.hand-tile'))x.classList.toggle('selected',x.dataset.option===selected);renderActions();};}row.append(b);});seat.append(row);
}else if(p.revealed){const h=document.createElement('div');h.className='melds';for(const t of p.hand){const el=img(t);el.style.width='19px';h.append(el);}if(p.hidden_ready_tiles){const hidden=text('span','','ready-hidden');hidden.setAttribute('aria-label',tr('concealed',{count:p.hidden_ready_tiles}));for(let i=0;i<p.hidden_ready_tiles;i++)hidden.append(text('span','','tile-back'));h.append(hidden);}seat.append(h);}else{const h=document.createElement('div');h.className='backs';h.setAttribute('aria-label',tr('concealed',{count:p.hand_count}));for(let i=0;i<p.hand_count;i++)h.append(text('span','','tile-back'));seat.append(h);}
if(p.melds.length){const ms=document.createElement('div');ms.className='melds';for(const m of p.melds){const meld=document.createElement('div');meld.className='meld';meld.title=actionText(m.kind);for(const t of m.tiles)meld.append(t===null?text('span','','tile-back'):img(t));ms.append(meld);}seat.append(ms);}if(p.win_tile){const claim=text('div','','melds');claim.title=tr('winningTile');claim.append(img(p.win_tile));seat.append(claim);}if(p.id!==0)seat.append(river(p.river));}
function turnPrompt(snapshot) {
  const table = snapshot.table;
  const player = table.players[0];
  if (table.phase === 'exchange') return table.exchange_same_suit ? 'promptSameSuitExchange' : 'promptExchange';
  if (table.phase === 'action') {
    if (table.float_cycle_owner > 0) return 'promptFloatCycle';
    if (player.win_count) return 'promptLockedWin';
    if (player.dark_ready) return 'promptDarkReady';
    if (player.open_ready) return 'promptOpenReady';
    if (player.ready && snapshot.variant === 'jj-three-player') return 'promptPaidReady';
  }
  return {void:'promptVoid',rob_kong:'promptRob',piao:'promptPiao',buy_base:'promptBuyBase',liability_choice:'liabilityPrompt'}[table.phase] || 'promptTurn';
}
function renderActions() {
  const box = $('actions');
  box.replaceChildren();
  if (!game) return;
  const playable = game.status === 'your_turn' && !game.automatic && !requestBusy;
  for (const button of document.querySelectorAll('#seat-0 .hand-tile.legal')) button.disabled = !playable;
  const openReady = currentVariant()?.family_id === 'ka-wu-xing';
  $('prompt').textContent = playable ? tr(turnPrompt(game)) : game.status === 'closed'
    ? I18N[LANGUAGE].errors.table_missing
    : tr({error:'promptError',round_end:'promptEnd',finished:'promptEnd'}[game.status] ||
         (game.automatic ? 'promptAuto' : 'promptWait'));
  $('chosen-tile').textContent = '';
  if (!playable) return;
  if (game.table.phase === 'exchange') {
    const hand = game.table.players[0].hand;
    const tiles = [...exchangeSelection].map(index => hand[index]);
    const candidates = game.choices.filter(option => option.kind === 'exchange' && exchangeContains(option, tiles));
    const choice = tiles.length === 3 ? candidates[0] : undefined;
    for (const tileButton of document.querySelectorAll('[data-exchange-index]')) {
      const index = Number(tileButton.dataset.exchangeIndex);
      tileButton.disabled = !exchangeSelection.has(index) &&
        !candidates.some(option => exchangeContains(option, [...tiles, hand[index]]));
    }
    const button = text('button', tr('confirmExchange'), 'action primary-action');
    button.disabled = !choice;
    button.onclick = () => sendAction(choice.id);
    box.append(button);
    $('chosen-tile').textContent = tr('exchangeCount', {count: exchangeSelection.size});
    return;
  }
  const isDiscard = option => ['discard', 'dahai'].includes(option.kind);
  const discards = game.choices.filter(isDiscard);
  if (discards.length) {
    const choice = discards.find(option => option.id === selected);
    const button = text('button', choice ? optionText(choice) : tr('pickTile'), 'action primary-action');
    button.disabled = !choice;
    button.onclick = () => sendAction(choice.id);
    box.append(button);
    $('chosen-tile').textContent = choice ? tr('selected', {tile: tileText(choice.tiles[0])}) : tr('clickTile');
    const declarations = [
      ['declare_ready', openReady ? 'declareOpenReady' : 'declareReady',
       openReady ? tr('openReadyTag') : actionText('declare_ready')],
      ['declare_dark', 'declareDarkReady', tr('darkReadyTag')],
    ];
    for (const [kind, message, shortLabel] of declarations) {
      const options = game.choices.filter(option => option.kind === kind);
      if (!options.length) continue;
      if (!choice) $('chosen-tile').textContent = tr(openReady ? 'pickOpenReadyTile' : 'pickReadyTile');
      const ready = choice && options.find(option => option.tiles[0] === choice.tiles[0]);
      const label = ready ? tr(message, {tile:tileText(ready.tiles[0])}) : shortLabel;
      const readyButton = text('button', label, 'action');
      readyButton.disabled = !ready;
      readyButton.onclick = () => sendAction(ready.id);
      box.append(readyButton);
    }
  }
  for (const option of game.choices) {
    if (isDiscard(option) || ['declare_ready','declare_dark'].includes(option.kind)) continue;
    const button = text('button', optionText(option), 'action' +
      (['hora', 'ron', 'zimo', 'ten_winds', 'eleven_winds'].includes(option.kind) ? ' primary-action' : ''));
    for (const tile of option.tiles.slice(0, 4)) button.append(img(tile));
    button.onclick = () => sendAction(option.id);
    box.append(button);
  }
}
async function sendAction(id){if(requestBusy||!game)return;requestBusy=true;$('room-error').textContent='';const sid=game.id;const rev=game.revision;renderActions();try{const s=await api(`/api/game/${sid}/action`,{revision:rev,choice:id});if(game?.id===sid)render(s);}catch(e){showError('room-error',e);}finally{requestBusy=false;renderActions();}}
$('auto-button').onclick=async()=>{if(!game)return;try{const s=await api(`/api/game/${game.id}/auto`,{enabled:!game.automatic});render(s);}catch(e){showError('room-error',e);}};
function winDetail(win) {
  const name = playerName(win.actor);
  const patterns = win.patterns.map(patternText).join(' · ');
  if (win.payments) {
    const payments = win.payments.flatMap((points, seat) => points ? [`${playerName(seat)} −${points.toLocaleString()}`] : []);
    const label = win.tile ? 'flowWinPayments' : win.tier ? 'tierPayments' : win.fan == null ? 'pointPayments' : 'winPayments';
    return tr(label, {name, tile: win.tile ? tileText(win.tile) : '', action: I18N[LANGUAGE].flowKinds[win.kind], fan: win.fan, tier: win.tier, patterns, payments: payments.join(' · ')});
  }
  return tr(win.fan == null ? 'winSharePoints' : 'winShare', {name, fan: win.fan, points: win.points, patterns});
}
function renderWake(result, container) {
  if (result.wake_tiles?.length) {
    container.append(text('span', tr('wakeTiles')));
    for (const tile of result.wake_tiles) container.append(img(tile));
  }
  for (const detail of result.wake_details || []) {
    container.append(text('p', tr('wakeSummary', {name: playerName(detail.actor), count: detail.count,
                                                hits: detail.hits}), 'bonus-detail'));
  }
}
function renderTransfers(deltas, label, container) {
  const amounts = deltas?.flatMap((points, seat) =>
    points ? [`${playerName(seat)} ${points > 0 ? '+' : '−'}${Math.abs(points).toLocaleString()}`] : []);
  if (amounts?.length) container.append(text('p', tr(label, {amounts: amounts.join(' · ')}), 'bonus-detail'));
}
function renderTaizhou(settlement, container) {
  if (!settlement) return;
  const {values, liable, liability_reason: reason, bills} = settlement;
  const amounts = values.map((value, seat) => `${playerName(seat)} ${value.toLocaleString()}`);
  container.append(text('p', tr('taizhouValues', {amounts: amounts.join(' · ')}), 'bonus-detail'));
  if (liable !== null) {
    container.append(text('p', tr('taizhouLiability', {name: playerName(liable), reason: tr(`taizhou_${reason}`)}), 'bonus-detail'));
  }
  bills.forEach((row, payer) => row.forEach((points, receiver) => {
    if (points) container.append(text('p', tr('taizhouBill', {payer: playerName(payer), receiver: playerName(receiver),
      points: points.toLocaleString()}), 'bonus-detail'));
  }));
}
function renderBonusTiles(result, container) {
  renderTaizhou(result.taizhou_settlement, container);
  const mode = result.bonus_mode;
  if (result.bonus?.length) {
    const heading = {exploding_fish:'fishTile', linhai_chong:'chongTiles'}[mode] || 'bonusTileHeading';
    container.append(text('span', tr(heading)));
    for (const tile of result.bonus) container.append(img(tile));
  }
  for (const entry of result.horse_breakdown || []) {
    const hits = entry.hits.map((count, seat) => `${playerName(seat)} ${count}`).join(' · ');
    container.append(text('p', tr('bilateralHorses', {name: playerName(entry.actor), hits}), 'bonus-detail'));
  }
  const label = {one_ma:'oneMaSummary', exploding_fish:'fishSummary', xiaogan_rank:'rankHorseSummary',
    linhai_chong:'linhaiChongSummary'}[mode] || 'bonusSummary';
  for (const item of result.bonus_details || []) {
    container.append(text('p', tr(label, {name: playerName(item.actor), count: item.count,
      hits: item.hits, extra: item.extra, points: item.per_loser}), 'bonus-detail'));
  }
}
function showResult(r){const finishedMatch=Boolean(r.match_hands);const leaders=r.match_winners||r.winners;const names=ids=>ids.map(playerName).join(LANGUAGE==='en'?' & ':'、');$('result-title').textContent=tr(r.next?'handResult':finishedMatch?'matchComplete':'matchEnd');$('result-reason').textContent=finishedMatch?tr('matchWinner',{names:names(leaders),total:r.match_hands}):tpl(I18N[LANGUAGE].reason[r.reason_code]||r.reason||'',{names:names(r.winners)});$('result-bonus').replaceChildren();renderBonusTiles(r, $('result-bonus'));renderWake(r, $('result-bonus'));renderTransfers(r.kong_deltas, 'kongTransfers', $('result-bonus'));renderTransfers(r.echo_deltas, 'dealerEcho', $('result-bonus'));renderTransfers(r.follow_deltas, 'followTransfers', $('result-bonus'));renderTransfers(r.pig_deltas, 'flowerPigTransfers', $('result-bonus'));renderTransfers(r.ready_deltas, 'readyTransfers', $('result-bonus'));renderTransfers(r.refund_deltas, 'kongRefunds', $('result-bonus'));renderTransfers(r.kong_refund_deltas, 'kongRefundTransfers', $('result-bonus'));renderTransfers(r.open_ready_deltas,'openReadyTransfers',$('result-bonus'));renderTransfers(r.chow_deltas,'chowTransfers',$('result-bonus'));renderTransfers(r.pon_deltas,'ponTransfers',$('result-bonus'));renderTransfers(r.closed_ready_deltas,'paidReadyTransfers',$('result-bonus'));renderTransfers(r.piao_deltas,'piaoTransfers',$('result-bonus'));renderTransfers(r.buy_base_deltas,'buyBaseTransfers',$('result-bonus'));renderTransfers(r.horse_deltas,'horseTransfers',$('result-bonus'));for(const win of r.win_details||[]){$('result-bonus').append(text('p',winDetail(win),'bonus-detail'));}const rows=r.scores.map((score,i)=>({i,score,delta:r.deltas[i]})).sort((a,b)=>b.score-a.score);$('result-scores').replaceChildren(...rows.map(x=>{const el=text('div','','result-row'+(x.i===0?' self':''));el.append(text('span',playerName(x.i)+(leaders.includes(x.i)?' · '+tr(finishedMatch?'firstPlace':'won'):'')),text('span',(x.delta>0?'+':'')+x.delta.toLocaleString(),'delta'+(x.delta>0?' plus':'')),text('span',x.score.toLocaleString(),'total'));return el;}));$('next-button').textContent=tr(r.next?'next':'newTable');if(!$('result-dialog').open)$('result-dialog').showModal();}
$('next-button').onclick=async()=>{if(!game)return;const next=game.result?.next;$('result-dialog').close();if(!next)return start();try{const s=await api(`/api/game/${game.id}/next`,{});render(s);}catch(e){showError('room-error',e);}};
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&selected&&game?.status==='your_turn'&&!$('rules-dialog').open&&!$('result-dialog').open)sendAction(selected);});
(async()=>{try{const c=await api('/api/catalog');catalog=c.variants;renderLobby();const sid=sessionStorage.getItem('tanyao_session');if(sid){try{const s=await api(`/api/game/${sid}`);if(s.status!=='closed'){variant=s.variant;renderLobby();$('policy').value=s.policy;$('lobby').classList.add('hidden');$('room').classList.remove('hidden');$('footer').classList.add('hidden');$('room-title').textContent=currentVariant().name;$('room-subtitle').textContent=currentVariant().subtitle;render(s);schedulePoll();}}catch{sessionStorage.removeItem('tanyao_session');}}}catch(e){showError('lobby-error',{code:'catalog_failed'});$('start-button').disabled=true;}})();
