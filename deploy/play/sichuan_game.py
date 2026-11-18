"""Human-facing adapter for the tested Sichuan reference engine."""
from __future__ import annotations
from .errors import ClientError
from collections import Counter
import copy
import time
import numpy as np
from common.suit_table import load_table, query_np, shanten_np
from sichuan.reference_impl import SichuanGame, is_hu
from .core import GameSession, NAMES, SUITS, tile_name, tile_label, Closed

_TABLE = None


def distance(hand, melds, void, *, allow_seven_pairs=True):
    global _TABLE
    if _TABLE is None:
        _TABLE = load_table()
    void_suit = -1 if void is None else void
    if allow_seven_pairs:
        return shanten_np(_TABLE, hand, melds, void_suit)
    return query_np(_TABLE, hand, melds, void_suit)[0]


class WebGame(SichuanGame):
    def __init__(self, seed):
        self.revealed_hands = {}
        super().__init__(seed)

    def _settle_hu(self, winner, hand_counts, ctx, discarder=None):
        self.revealed_hands[winner] = hand_counts[:]
        super()._settle_hu(winner, hand_counts, ctx, discarder)

    def legal_actions(self):
        if self.phase == 'rob_kong':
            return self.rob_queue[0], [('ron',None),('pass',None)]
        return super().legal_actions()

    def step(self, action):
        if action not in self.legal_actions()[1]:
            raise ClientError('illegal_action')
        # The training reference auto-accepts every legal rob. Web users must be able to pass.
        if self.phase == 'action' and action[0] == 'bugang':
            tile=action[1]
            robbers=[]
            for k in (1,2,3):
                i=(self.cur+k)%4;p=self.players[i];hand=p.hand[:];hand[tile]+=1
                if self.alive(i) and is_hu(hand,len(p.melds),p.void):robbers.append(i)
            if robbers:
                self.rob_actor,self.rob_tile,self.rob_queue,self.rob_winners=self.cur,tile,robbers,[]
                self.drawn_tile = None
                self.phase='rob_kong';self.turn_guard+=1
                return
        if self.phase == 'rob_kong':
            self.turn_guard+=1
            i=self.rob_queue.pop(0);tile=self.rob_tile;actor=self.rob_actor
            if action[0]=='ron':
                h=self.players[i].hand[:];h[tile]+=1
                self._settle_hu(i,h,{'qiang_gang':True},discarder=actor);self.rob_winners.append(i)
            elif action[0]!='pass':raise ValueError('抢杠窗口只允许胡牌或过')
            if self.rob_queue:return
            p=self.players[actor]
            p.hand[tile]-=1
            if self.rob_winners:
                self.discards[actor].append(tile)
                self._begin_turn(self._next_alive(actor))
            else:
                p.melds.remove(('peng',tile));p.melds.append(('gang_bu',tile))
                self._gang_money(actor,'gang_bu');self._begin_turn_same(from_gang=True)
            return
        super().step(action)


def public_features(g, me, acts):
    """Explicit policy boundary: own tiles, public events, public hand lengths only."""
    p = g.players[me]
    public = Counter(t for river in g.discards for t in river)
    for q in g.players:
        if q.hu:
            public.update({t:c for t,c in enumerate(q.hand) if c})
        for kind,t in q.melds:
            public[t] += 4 if kind.startswith('gang') else 3
    return {'hand': p.hand[:], 'melds': p.melds[:], 'void': p.void, 'actions': list(acts),
            'public': dict(public), 'opponents': [{'void':q.void,'finished':q.hu,'melds':len(q.melds)} for j,q in enumerate(g.players) if j != me]}


def heuristic(view):
    hand, melds, void, acts = view['hand'], view['melds'], view['void'], view['actions']
    allow_pairs = view.get('allow_seven_pairs', True)
    for a in acts:
        if a[0] in ('ron','zimo'):
            return a
    if acts[0][0] == 'void':
        return ('void', min(range(3), key=lambda s: sum(hand[s*9:s*9+9])))
    discards = [a for a in acts if a[0] == 'discard']
    if discards:
        scores = []
        for a in discards:
            h = hand[:];h[a[1]]-=1
            scores.append((distance(h,len(melds),void,allow_seven_pairs=allow_pairs),a,h))
        best=min(x[0] for x in scores)
        finalists=[x for x in scores if x[0]==best]
        out=[]
        seen=[hand[t]+view['public'].get(t,0) for t in range(27)]
        for sh,a,h in finalists:
            effective=0
            for t in range(27):
                if t//9 == void or seen[t]>=4:continue
                after=h[:];after[t]+=1
                if distance(after,len(melds),void,allow_seven_pairs=allow_pairs)<sh:
                    effective+=max(0,4-seen[t])
            # Only a tie-break: don't import riichi-style folding into this ruleset.
            danger=sum(0 if q['finished'] or q['void']==a[1]//9 else 1+q['melds']*.15 for q in view['opponents'])
            out.append((effective,-danger,-a[1],a))
        return max(out)[-1]
    for a in acts:
        if a[0] in ('ankan','bugang','zhigang'):
            return a
    if ('peng',None) in acts:
        # Identify the offered tile via the immutable action context supplied by the adapter.
        t=view['pending_tile'];h=hand[:];h[t]-=2
        after_call = distance(h, len(melds) + 1, void, allow_seven_pairs=allow_pairs)
        without_call = distance(hand, len(melds), void, allow_seven_pairs=allow_pairs)
        if after_call <= without_call:
            return ('peng',None)
    return ('pass',None)


def action_id(a):
    k,t=a
    return {'discard':lambda:t,'ankan':lambda:27+t,'bugang':lambda:27+t,
            'peng':lambda:54,'zhigang':lambda:55,'zimo':lambda:56,'ron':lambda:56,
            'pass':lambda:57,'void':lambda:58+t}[k]()


class SichuanSession(GameSession):
    def __init__(self, policy='heuristic', *, model=None, seed=None, pacing=.12):
        super().__init__('sichuan',policy,seed=seed,pacing=pacing)
        self.model=model
        self.round_no=0
        self.totals=[0]*4
        self.game=None

    def begin(self):
        if self.thread is not None and self.thread.is_alive() and self.status == 'round_end':
            self.thread.join(timeout=1)
        with self.cv:
            if self.status == 'starting' and self.thread is not None:
                raise ClientError('round_starting')
            if self.status not in ('starting','round_end'):
                raise ClientError('round_active')
            self.round_no+=1
            self.result=None;self.automatic=False;self.options=[];self.status='starting';self.version+=1
            self.game=WebGame(self.seed+self.round_no)
            self.log_event('hand_start_void',{'number':self.round_no},'round')
            self.start_worker(self.play_round)

    def choice(self,g,i,acts):
        v=public_features(g,i,acts)
        if g.pending_discard:v['pending_tile']=g.pending_discard[1]
        if self.policy=='model' and self.model is not None:
            from .sichuan_model import observation
            return self.model.choose(observation(g,i),acts)
        return heuristic(v)

    def view(self):
        g=self.game
        players=[]
        for i,p in enumerate(g.players):
            shown=g.revealed_hands.get(i,p.hand)
            tiles=[tile_name(t) for t,c in enumerate(shown) for _ in range(c)]
            players.append({'id':i,'name':NAMES[i],'score':self.totals[i]+p.score_delta,'delta':p.score_delta,
                'hand':tiles if i==0 or p.hu else [], 'hand_count':len(tiles),
                'revealed':i==0 or p.hu,'won':p.hu,'void':None if p.void is None else SUITS[p.void],'void_suit':p.void,
                'riichi':False,'river':[{'tile':tile_name(t)} for t in g.discards[i]],
                'melds':[{'kind':k,'tiles':[tile_name(t)]*(4 if k.startswith('gang') else 3)} for k,t in p.melds]})
        i,_=g.legal_actions()
        if i!=0 and g.phase in ('response','rob_kong'):i=None
        return {'players':players,'hand_number':self.round_no,'round':f'Hand {self.round_no}','dealer':0,'current':i,
                'remaining':len(g.wall),'dora':[],'honba':0,'pot':0,'phase':g.phase,
                'prompt':'Choose a void suit' if g.phase=='void' else 'Choose a tile or action',
                'drawn':tile_name(g.drawn_tile) if g.phase=='action' and g.cur==0 and g.drawn_tile is not None else None}

    def options_for(self,acts):
        out=[]
        for a in acts:
            k,t=a
            label= {'void':f'Void: {SUITS[t]}' if k=='void' else '', 'discard':f'Discard {tile_label(tile_name(t))}' if k=='discard' else '',
                    'ankan':'Closed kong','bugang':'Added kong','peng':'Pon','zhigang':'Open kong','zimo':'Self-draw win','ron':'Win','pass':'Pass'}[k]
            out.append({'action':a,'kind':k,'label':label,'suit':t if k=='void' else None,'tiles':[tile_name(t)] if t is not None and k!='void' else []})
        return out

    def play_round(self):
        g=self.game
        while g.phase!='over':
            if self.closed:raise Closed()
            i,acts=g.legal_actions()
            preferred=self.choice(g,i,acts)
            if i==0:
                if len(acts)==1 and acts[0][0]=='pass':
                    a=acts[0]
                else:
                    a=self.ask(self.view(),self.options_for(acts),acts.index(preferred))
            else:
                self.publish(self.view())
                if self.pacing and not self.automatic:time.sleep(self.pacing)
                a=preferred
            before=[p.score_delta for p in g.players]
            old_hu=len(g.hu_order)
            g.step(a)
            k,t=a
            if k=='discard':self.log_event('discard',{'actor':i,'tile':tile_name(t)})
            elif k=='void':self.log_event('void',{'actor':i,'suit':t})
            elif k not in ('pass','ron','zimo'):self.log_event('action',{'actor':i,'action':k},'call')
            for winner,fan,names in g.hu_order[old_hu:]:
                self.log_event('win_fan',{'actor':winner,'fan':fan,'patterns':list(names)},'win')
            assert g.tile_conservation()==108 and sum(g.scores())==0
        delta=g.scores()
        view=self.view()
        # view already includes current delta once; totals update is only for the next hand.
        self.totals=[a+b for a,b in zip(self.totals,delta)]
        with self.cv:
            self.result={'title':'Hand complete','reason':'Three players have won.' if len(g.hu_order)>=3 else 'Wall exhausted; ready-hand payments and refunds settled.','reason_code':'blood_end' if len(g.hu_order)>=3 else 'sichuan_draw',
                         'deltas':delta,'scores':self.totals[:],'winners':[w for w,_,_ in g.hu_order], 'next':True}
            self.log_event(self.result['reason_code'],kind='round')
            self.publish(view,'round_end')
