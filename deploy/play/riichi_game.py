"""Single-table, per-hand adapter over the existing native riichi engine.

Raw native logs are server-private. Browser snapshots are an explicit whitelist.
"""
from __future__ import annotations
from .errors import ClientError
from collections import Counter
import itertools
import json
import time
import numpy as np
from .core import GameSession, NAMES, TILES, tile_label, tile_sort, Closed

ALL = TILES + ['5mr','5pr','5sr']


def hand_tiles(ps):
    counts=list(ps.tehai)
    out=[]
    for i,c in enumerate(counts):
        if i in (4,13,22) and ps.akas_in_hand[i//9]:
            out.append(ALL[34+i//9]);c-=1
        out += [TILES[i]]*c
    return out


def combinations_for(hand, wanted):
    choices=[[t for t in set(hand) if t.rstrip('r')==w.rstrip('r')] for w in wanted]
    have=Counter(hand)
    seen=set()
    for c in itertools.product(*choices):
        if Counter(c)<=have and c not in seen:
            seen.add(c);yield list(c)


def legal_options(ps, public):
    _,mask=ps.encode_obs(4,False)
    actor=int(ps.player_id);cans=ps.last_cans
    hand=hand_tiles(ps)
    pai=ps.last_kawa_tile()
    target=int(cans.target_actor)
    options=[]
    def add(ev,index, label, tiles=()):
        ps.validate_reaction(json.dumps(ev))
        options.append({'action':ev,'kind':ev['type'],'label':label,'tiles':list(tiles),'index':index,'win_type':('zimo' if ev.get('actor')==ev.get('target') else 'ron') if ev['type']=='hora' else None})
    for i in np.flatnonzero(mask):
        i=int(i)
        if i<37:
            t=ALL[i]
            add({'type':'dahai','actor':actor,'pai':t,'tsumogiri':ps.last_self_tsumo()==t},i,f'Discard {tile_label(t)}',[t])
        elif i==37:add({'type':'reach','actor':actor},i,'Riichi')
        elif i in (38,39,40):
            b=TILES.index(pai.rstrip('r'))
            wanted={38:[b+1,b+2],39:[b-1,b+1],40:[b-2,b-1]}[i]
            for consumed in combinations_for(hand,[TILES[x] for x in wanted]):
                add({'type':'chi','actor':actor,'target':target,'pai':pai,'consumed':consumed},i,'Chi',[*consumed,pai])
        elif i==41:
            for consumed in combinations_for(hand,[pai,pai]):
                add({'type':'pon','actor':actor,'target':target,'pai':pai,'consumed':consumed},i,'Pon',[*consumed,pai])
        elif i==42:
            if cans.can_daiminkan:
                for consumed in combinations_for(hand,[pai]*3):
                    add({'type':'daiminkan','actor':actor,'target':target,'pai':pai,'consumed':consumed},i,'Open kan',[*consumed,pai])
            else:
                for t in ps.ankan_candidates():
                    for consumed in combinations_for(hand,[t]*4):
                        add({'type':'ankan','actor':actor,'consumed':consumed},i,'Closed kan',consumed)
                for t in ps.kakan_candidates():
                    for own in [x for x in set(hand) if x.rstrip('r')==t.rstrip('r')]:
                        meld=next(m for m in public['players'][actor]['melds'] if m['kind']=='pon' and m['tiles'][0].rstrip('r')==t.rstrip('r'))
                        add({'type':'kakan','actor':actor,'pai':own,'consumed':meld['tiles']},i,'Added kan',[*meld['tiles'],own])
        elif i==43:
            add({'type':'hora','actor':actor,'target':actor if cans.can_tsumo_agari else target},i,'Tsumo' if cans.can_tsumo_agari else 'Ron')
        elif i==44:add({'type':'ryukyoku'},i,'Nine-terminals draw')
        elif i==45:add({'type':'none'},i,'Pass')
    if not options:raise RuntimeError('native engine gave no legal choices')
    return options


def public_table(events, viewer=0):
    """Build safe view; no seeds, raw event list, metadata, or concealed opponent tiles."""
    players=[{'id':i,'name':NAMES[i],'score':25000,'hand':[], 'hand_count':13,'river':[], 'melds':[], 'won':False,'riichi':False,'void':None} for i in range(4)]
    hands=[[] for _ in range(4)]
    table={'players':players,'round':'E1','round_wind':'E','round_number':1,'dealer':0,'current':None,'remaining':70,'dora':[],'honba':0,'pot':0,'phase':'action','drawn':None,'prompt':'Choose a tile or action'}
    latest_discard=None
    def remove(actor,tiles):
        for t in tiles:
            if t not in hands[actor]:raise RuntimeError(f'inconsistent hand at {actor}')
            hands[actor].remove(t)
    for e in events:
        kind=e['type'];a=e.get('actor')
        if kind=='start_kyoku':
            hands=[x[:] for x in e['tehais']]
            for i,p in enumerate(players):
                p.update(score=e['scores'][i],river=[],melds=[],won=False,riichi=False)
            table.update(round=f'{e["bakaze"]}{e["kyoku"]}',round_wind=e['bakaze'],round_number=e['kyoku'],dealer=e['oya'],honba=e['honba'],pot=e['kyotaku'],dora=[e['dora_marker']],remaining=70)
        elif kind=='tsumo':
            hands[a].append(e['pai']);table['remaining']=max(0,table['remaining']-1);table['current']=a
            table['drawn']=e['pai'] if a==viewer else None
        elif kind=='dahai':
            remove(a,[e['pai']]);players[a]['river'].append({'tile':e['pai'],'riichi':players[a].get('reach_pending',False),'called':False})
            players[a]['reach_pending']=False
            latest_discard=(a,e['pai']);table['drawn']=None
        elif kind in ('chi','pon','daiminkan'):
            remove(a,e['consumed'])
            players[a]['melds'].append({'kind':kind,'tiles':e['consumed']+[e['pai']]})
            if players[e['target']]['river']:players[e['target']]['river'][-1]['called']=True
            table['current']=a;table['drawn']=None
        elif kind=='ankan':
            remove(a,e['consumed']);players[a]['melds'].append({'kind':kind,'tiles':e['consumed']})
        elif kind=='kakan':
            remove(a,[e['pai']]);latest_discard=(a,e['pai'])
            m=next(m for m in players[a]['melds'] if m['kind']=='pon' and m['tiles'][0].rstrip('r')==e['pai'].rstrip('r'))
            m.update(kind='kakan',tiles=e['consumed']+[e['pai']])
        elif kind=='reach':players[a].update(riichi=True,reach_pending=True)
        elif kind=='reach_accepted':players[a]['score']-=1000;table['pot']+=1
        elif kind=='dora':table['dora'].append(e['dora_marker'])
        elif kind in ('hora','ryukyoku'):
            if e.get('deltas'):
                for i,d in enumerate(e['deltas']):players[i]['score']+=d
            if kind=='hora':
                players[a]['won']=True
                if e['target']!=a and latest_discard:
                    hands[a].append(latest_discard[1])
    for i,p in enumerate(players):
        p['hand_count']=len(hands[i]);p['revealed']=i==viewer or p['won']
        p['hand']=sorted(hands[i],key=tile_sort) if p['revealed'] else []
        p.pop('reach_pending',None)
    return table


def heuristic_index(ps, options, public=None):
    for j,x in enumerate(options):
        if x['kind']=='hora':return j
    for j,x in enumerate(options):
        if x['kind']=='reach':return j
    # Closed-hand beginner policy: riichi supplies a yaku, then shanten and public ukeire.
    discards=[(j,x) for j,x in enumerate(options) if x['kind']=='dahai']
    if discards:
        from libriichi.arena import shanten_after_draw
        hand=list(ps.tehai)
        _,after=shanten_after_draw(hand, max(0,(sum(hand)-2)//3))
        best=min(int(after[TILES.index(x['action']['pai'].rstrip('r'))]) for _,x in discards)
        candidates=[(j,x) for j,x in discards if int(after[TILES.index(x['action']['pai'].rstrip('r'))])==best]
        seen=Counter({i:c for i,c in enumerate(hand)})
        if public:
            for p in public['players']:
                for d in p['river']:
                    if not d.get('called'):seen[TILES.index(d['tile'].rstrip('r'))]+=1
                for m in p['melds']:
                    for t in m['tiles']:seen[TILES.index(t.rstrip('r'))]+=1
            for t in public['dora']:seen[TILES.index(t.rstrip('r'))]+=1
        def key(item):
            j,x=item;t=TILES.index(x['action']['pai'].rstrip('r'))
            h=hand[:];h[t]-=1;effective=0
            for draw in range(34):
                if seen[draw]>=4:continue
                h[draw]+=1
                sh,_=shanten_after_draw(h,max(0,(sum(h)-2)//3))
                if sh<best:effective+=4-seen[draw]
                h[draw]-=1
            return effective, x['action']['pai'].endswith('r') is False, -t
        return max(candidates,key=key)[0]
    for j,x in enumerate(options):
        if x['kind']=='reach':return j
    for j,x in enumerate(options):
        if x['kind']=='none':return j
    return 0


class RiichiSession(GameSession):
    def __init__(self,policy='model',*,model=None,seed=None,pacing=.12):
        super().__init__('riichi',policy,seed=seed,pacing=pacing)
        self.model=model
        self.kyoku=0;self.honba=0;self.pot=0;self.scores=[25000]*4;self.renchan=False
        self.next_hand=(0,0);self._last_log=[];self._seen_count=0

    def begin(self):
        if self.thread is not None and self.thread.is_alive() and self.status == 'round_end':
            self.thread.join(timeout=1)
        with self.cv:
            if self.status == 'starting' and self.thread is not None:
                raise ClientError('round_starting')
            if self.status not in ('starting','round_end') or self.next_hand is None:
                raise ClientError('round_active')
            self.result=None;self.options=[];self.automatic=False;self.status='starting';self.version+=1
            self._seen_count=0;self._last_log=[]
            self.start_worker(self.play_round)

    def observe_events(self,events):
        # Every callback contains the full current hand, not an incremental stream.
        for e in events[self._seen_count:]:
            k=e['type'];a=e.get('actor')
            if k=='start_kyoku':self.log_event('hand_start',kind='round')
            elif k=='dahai':self.log_event('discard',{'actor':a,'tile':e['pai']})
            elif k in ('reach','chi','pon','daiminkan','ankan','kakan'):
                self.log_event('action',{'actor':a,'action':k},'call')
            elif k=='hora':self.log_event('tsumo' if a==e['target'] else 'ron',{'actor':a,'target':e['target']},'win')
            elif k=='ryukyoku':self.log_event('draw',kind='round')
        self._seen_count=len(events);self._last_log=events

    def play_round(self):
        from libriichi.arena import Rollout
        session=self
        class Engine:
            name='TanyaoDojo';engine_type='mjai-log'
            def set_player_ids(self,ids):self.ids=list(ids)
            def start_game(self,index):pass
            def end_kyoku(self,index):pass
            def end_game(self,index,scores):pass  # truncated Rollout scores are not actual inter-hand scores
            def react_batch(self,states):
                out=[]
                for state in states:
                    if session.closed:raise Closed()
                    ps=state.state;actor=int(ps.player_id)
                    events=json.loads(state.events_json)
                    session.observe_events(events)
                    table=public_table(events)
                    table['current']=actor if actor==0 or ps.last_cans.can_discard else None
                    opts=legal_options(ps,table)
                    preferred=session.model.choose(ps,opts) if session.policy=='model' and session.model else heuristic_index(ps,opts,table)
                    if actor==0:
                        if len(opts)==1 and opts[0]['kind']=='none':
                            action=opts[0]['action']
                        else:
                            action=session.ask(table,opts,preferred)
                    else:
                        session.publish(table)
                        if session.pacing and not session.automatic:time.sleep(session.pacing)
                        action=opts[preferred]['action']
                    out.append(json.dumps(action))
                return out
        dummy={'type':'start_kyoku','bakaze':'E','dora_marker':'1m','kyoku':1,'honba':0,'kyotaku':0,'oya':0,'scores':[25000]*4,'tehais':[['?']*13 for _ in range(4)]}
        spec=dict(seed=(self.seed,20261116),kyoku=self.kyoku,honba=self.honba,kyotaku=self.pot,scores=self.scores[:],in_renchan=self.renchan,
                  seat_engines=[0]*4,reference=[json.dumps(dummy)],branch_pos=0,branch_actor=0,replacement=None,horizon='kyoku',commit='',reshuffle=None,wall=None)
        initial=self.scores[:]
        raw,final,first,pot,nxt=Rollout(disable_progress_bar=True).run([Engine()],[spec])[0]
        events=[json.loads(line) for line in raw.splitlines() if line.strip()]
        self.observe_events(events)
        self.scores=list(first if nxt is not None else final)
        self.pot=int(pot) if nxt is not None else 0
        self.next_hand=nxt
        if nxt is not None:
            # A valid `next` has already passed the native west-extension end check.
            self.renchan=nxt[0]==self.kyoku
            self.kyoku,self.honba=nxt
        table=public_table(events)
        table['pot']=self.pot
        for i,p in enumerate(table['players']):p['score']=self.scores[i]
        winners=[int(e['actor']) for e in events if e['type']=='hora']
        with self.cv:
            self.result={'title':'Hand result' if nxt is not None else 'Hanchan complete',
                         'reason':'Winning hand' if winners else 'Draw settlement','reason_code':'winners' if winners else 'draw',
                         'scores':self.scores[:],'deltas':[a-b for a,b in zip(self.scores,initial)],'winners':winners,'next':nxt is not None}
            self.publish(table,'round_end' if nxt is not None else 'finished')
