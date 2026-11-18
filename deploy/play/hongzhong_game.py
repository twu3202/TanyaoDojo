"""Web adapter and public-information heuristic for the Hongzhong family."""
from collections import Counter
from functools import lru_cache
import time
from common.suit_table import load_table,query_np
from .core import GameSession,TILES,Closed
from .errors import ClientError
from .hongzhong import HongzhongGame,can_win,RED,ONE_MA,TWO_MA

_TABLE=None

def tile(t):return 'C' if t==RED else TILES[t]

@lru_cache(maxsize=30000)
def distance(hand,melds,seven_pairs=False):
    global _TABLE
    if _TABLE is None:_TABLE=load_table()
    # Progress estimate only; exact wildcard legality is checked independently for every win.
    standard=query_np(_TABLE,hand[:27],melds,-1)[0]-hand[RED]
    if seven_pairs and not melds:
        natural_pairs=sum(n//2 for n in hand[:RED])
        filled=min(hand[RED],sum(n%2 for n in hand[:RED]))
        pairs=natural_pairs+filled+(hand[RED]-filled)//2
        return max(0,min(standard,6-pairs))
    return max(0,standard)


def policy_view(g,me,acts):
    seen=Counter(t for r in g.discards for t in r)
    for p in g.players:
        for k,t in p.melds:seen[t]+=3 if k=='peng' else 4
    if g.phase=='rob_kong':seen[g.rob_tile]+=1
    return {'hand':g.players[me].hand[:],'melds':g.players[me].melds[:],'actions':acts[:],
            'seven_pairs':g.seven_pairs,'public_counts':dict(seen),
            'pending_tile':None if g.pending_discard is None else g.pending_discard[1]}


def choose(v):
    hand=v['hand'];melds=v['melds'];acts=v['actions']
    seven_pairs=v['seven_pairs']
    for a in acts:
        if a[0] in ('zimo','ron'):return a
    # Preserve wildcards; natural concealed/added kongs are useful fixed-score actions.
    for a in acts:
        if a[0] in ('ankan','bugang') and a[1]!=RED:return a
    discs=[a for a in acts if a[0]=='discard']
    if discs:
        natural=[a for a in discs if a[1]!=RED];discs=natural or discs
        candidates=[]
        for a in discs:
            h=hand[:];h[a[1]]-=1
            candidates.append((distance(tuple(h),len(melds),seven_pairs),a,h))
        lowest=min(x[0] for x in candidates);candidates=[x for x in candidates if x[0]==lowest]
        seen=[hand[t]+v['public_counts'].get(t,0) for t in range(28)]
        def score(item):
            d,a,h=item;wins=improve=0
            for t in range(28):
                left=max(0,4-seen[t])
                if not left:continue
                h[t]+=1
                if can_win(h,len(melds),seven_pairs=seven_pairs):wins+=left
                if distance(tuple(h),len(melds),seven_pairs)<d:improve+=left
                h[t]-=1
            return wins,improve,-a[1]
        return max(candidates,key=score)[1]
    if ('zhigang',None) in acts and v['pending_tile']!=RED:return ('zhigang',None)
    if ('peng',None) in acts and v['pending_tile']!=RED:
        t=v['pending_tile'];h=hand[:];h[t]-=2
        # A pon must be followed by a discard, so price that discard as part of the call.
        after=[]
        for d,n in enumerate(h):
            if n and d!=RED:
                hh=h[:];hh[d]-=1;after.append(distance(tuple(hh),len(melds)+1,seven_pairs))
        if after and min(after)<distance(tuple(hand),len(melds),seven_pairs):return ('peng',None)
    return ('pass',None)


class HongzhongSession(GameSession):
    def __init__(self,policy='heuristic',*,model=None,seed=None,pacing=.12,variant='hongzhong',profile=TWO_MA):
        super().__init__(variant,policy,seed=seed,pacing=pacing)
        self.profile=profile
        self.round_no=0;self.totals=[0]*4;self.dealer=0;self.game=None

    def begin(self):
        if self.thread is not None and self.thread.is_alive() and self.status=='round_end':self.thread.join(timeout=1)
        with self.cv:
            if self.status=='starting' and self.thread is not None:raise ClientError('round_starting')
            if self.status not in ('starting','round_end'):raise ClientError('round_active')
            self.round_no+=1;self.game=HongzhongGame(self.seed+self.round_no,self.dealer,profile=self.profile)
            self.result=None;self.options=[];self.automatic=False;self.status='starting';self.version+=1
            self.log_event('hand_start',kind='round');self.start_worker(self.play_round)

    def view(self):
        g=self.game;players=[]
        for i,p in enumerate(g.players):
            shown=g.revealed.get(i,p.hand);visible=i==0 or i in g.winners
            players.append({'id':i,'score':self.totals[i]+p.score_delta,'delta':p.score_delta,'hand_count':sum(shown),
                            'hand':[tile(t) for t,n in enumerate(shown) for _ in range(n)] if visible else [],
                            'revealed':visible,'won':i in g.winners,'riichi':False,'void_suit':None,
                            'river':[{'tile':tile(t)} for t in g.discards[i]],
                            'melds':[{'kind':k,'tiles':[tile(t)]*(3 if k=='peng' else 4)} for k,t in p.melds]})
        actor,_=g.legal_actions()
        if actor!=0 and g.phase in ('response','rob_kong'):actor=None
        return {'players':players,'hand_number':self.round_no,'round':f'Hand {self.round_no}','dealer':g.dealer,'current':actor,
                'remaining':max(0,len(g.wall)-g.reserve),'dora':[],'wildcard':'C','honba':0,'pot':0,'phase':g.phase,
                'drawn':tile(g.drawn[0]) if g.phase=='action' and g.cur==0 and g.draw_turn else None}

    def play_round(self):
        g=self.game
        while g.phase!='over':
            if self.closed:raise Closed()
            actor,acts=g.legal_actions();preferred=choose(policy_view(g,actor,acts))
            opts=[{'action':a,'kind':a[0],'label':a[0],'tiles':[tile(a[1])] if a[1] is not None else []} for a in acts]
            if actor==0:
                action=acts[0] if len(acts)==1 and acts[0][0]=='pass' else self.ask(self.view(),opts,acts.index(preferred))
            else:
                self.publish(self.view())
                if self.pacing and not self.automatic:time.sleep(self.pacing)
                action=preferred
            g.step(action);k,t=action
            if k=='discard':self.log_event('discard',{'actor':actor,'tile':tile(t)})
            elif k not in ('pass','zimo','ron'):self.log_event('action',{'actor':actor,'action':k},'call')
        view=self.view();delta=g.scores();self.totals=[a+b for a,b in zip(self.totals,delta)];self.dealer=g.next_dealer
        for w in g.winners:
            self.log_event('hongzhong_win' if g.rob_actor is None else 'rob_kong_win',{'actor':w,'target':g.rob_actor},'win')
            b=g.bonus_details[w]
            if b['tiles']:
                code='one_ma_bonus' if g.profile==ONE_MA else 'bonus'
                self.log_event(code,{'tiles':[tile(t) for t in b['tiles']],'hits':b['hits'],'extra':b['extra']},'win')
        if not g.winners:self.log_event('hongzhong_draw',kind='round')
        with self.cv:
            self.result={'title':'Hand result','reason_code':g.reason_code,'scores':self.totals[:],'deltas':delta,'winners':g.winners[:],
                         'bonus_mode':ONE_MA if g.profile==ONE_MA else TWO_MA,'bonus':[tile(t) for t in g.bonus],
                         'bonus_details':[{'actor':i,'count':len(b['tiles']),'hits':b['hits'],'extra':b['extra'],
                                           'per_loser':b['per_loser']} for i,b in g.bonus_details.items()],'next':True}
            self.publish(view,'round_end')
