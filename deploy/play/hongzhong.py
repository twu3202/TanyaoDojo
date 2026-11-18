"""Hongzhong 112 with named and source-backed bonus modes.

Separate single-hand rules engine: suited tiles + four red-dragon wildcards.
See rules/Hongzhong.md for sourced rules and named room choices.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass,field
from functools import lru_cache
import itertools
import random

RED=27
TWO_MA='two_ma'
ONE_MA='one_ma'
FANGCHENGGANG='fangchenggang'
ANHUI='anhui'


@lru_cache(maxsize=100000)
def _sets(counts,wild,need):
    if need<0 or sum(counts)+wild!=need*3:return False
    if not any(counts):return wild==3*need
    first=next(i for i,c in enumerate(counts) if c)
    for take in range(1,min(3,counts[first])+1):
        cost=3-take
        if cost<=wild:
            c=list(counts);c[first]-=take
            if _sets(tuple(c),wild-cost,need-1):return True
    # The first natural tile need not be the low tile; wildcards can fill either side.
    for start in range(max(first//9*9,first-2),min(first,first//9*9+6)+1):
        seq=range(start,start+3)
        choices=[(1,) if t==first else ((0,1) if counts[t] else (0,)) for t in seq]
        for used in itertools.product(*choices):
            cost=3-sum(used)
            if cost>wild:continue
            c=list(counts)
            for t,n in zip(seq,used):c[t]-=n
            if _sets(tuple(c),wild-cost,need-1):return True
    return False


@lru_cache(maxsize=80000)
def standard_win(hand,melds=0):
    if len(hand)!=28 or any(c<0 or c>4 for c in hand):return False
    if sum(hand)!=3*(4-melds)+2:return False
    natural=hand[:27];wild=hand[RED]
    if wild>=2 and _sets(natural,wild-2,4-melds):return True
    for t,c in enumerate(natural):
        for use in (1,2):
            if c>=use and 2-use<=wild:
                h=list(natural);h[t]-=use
                if _sets(tuple(h),wild-(2-use),4-melds):return True
    return False


def can_win(hand,melds=0,four_red=True,*,seven_pairs=False):
    if len(hand)!=28 or not 0<=melds<=4 or any(n<0 or n>4 for n in hand) or sum(hand)!=3*(4-melds)+2:return False
    pairs=seven_pairs and not melds and sum(n%2 for n in hand[:RED])<=hand[RED]
    return (four_red and hand[RED]==4) or pairs or standard_win(tuple(hand),melds)


@dataclass
class Player:
    hand:list=field(default_factory=lambda:[0]*28)
    melds:list=field(default_factory=list)
    score_delta:int=0


class HongzhongGame:
    def __init__(self,seed=0,dealer=0,*,profile=TWO_MA):
        if profile not in (TWO_MA,ONE_MA,FANGCHENGGANG,ANHUI):raise ValueError('Unknown Hongzhong profile')
        self.profile=profile
        self.reserve=4 if profile in (FANGCHENGGANG,ANHUI) else 0
        self.seven_pairs=profile==FANGCHENGGANG
        self.players=[Player() for _ in range(4)]
        self.wall=[t for t in range(28) for _ in range(4)];random.Random(seed).shuffle(self.wall)
        self.discards=[[] for _ in range(4)]
        self.dealer=self.cur=dealer;self.next_dealer=dealer
        self.phase='action';self.draw_turn=False;self.last_drawer=dealer;self.drawn={}
        self.pending_discard=None;self.queue=[];self.pending_claim=None
        self.rob_actor=None;self.rob_tile=None;self.rob_winners=[]
        self.winners=[];self.revealed={};self.claimed=[];self.bonus=[];self.bonus_details={}
        self.gang_ledger=[];self.reason_code=None;self.steps=0
        for p in self.players:
            for _ in range(13):p.hand[self.wall.pop()]+=1
        self._draw(dealer)

    def scores(self):return [p.score_delta for p in self.players]

    def physical_counts(self):
        c=Counter(self.wall+self.bonus+self.claimed)
        for i,p in enumerate(self.players):
            c.update({t:n for t,n in enumerate(p.hand) if n})
            c.update(self.discards[i])
            for kind,t in p.melds:c[t]+=3 if kind=='peng' else 4
        return c

    def _pay(self,payer,winner,amount,gang=False):
        self.players[payer].score_delta-=amount;self.players[winner].score_delta+=amount
        if gang:self.gang_ledger.append((payer,winner,amount))

    def _can_draw(self):
        return len(self.wall)>self.reserve

    def _draw(self,actor,*,replacement=False):
        if not self._can_draw():
            self.phase='over';self.reason_code='hongzhong_draw';self.next_dealer=self.last_drawer
            return
        end=-1
        if replacement and self.reserve:
            end=self.reserve if self.profile==ANHUI else 0
        t=self.wall.pop(end);self.players[actor].hand[t]+=1
        self.drawn[actor]=t;self.last_drawer=self.cur=actor;self.draw_turn=True;self.phase='action'

    def legal_actions(self):
        if self.phase=='over':return None,[]
        if self.phase=='rob_kong':return self.queue[0],[('ron',None),('pass',None)]
        if self.phase=='response':
            i=self.queue[0];t=self.pending_discard[1];p=self.players[i]
            acts=[('pass',None)]
            if self.profile==ANHUI and t==RED:return i,acts
            if p.hand[t]>=2:acts.append(('peng',None))
            if p.hand[t]==3 and self._can_draw():acts.append(('zhigang',None))
            return i,acts  # Ordinary discards can never be won on.
        p=self.players[self.cur];acts=[('discard',t) for t,n in enumerate(p.hand) if n]
        if self.draw_turn:
            if can_win(p.hand,len(p.melds),seven_pairs=self.seven_pairs):acts.append(('zimo',None))
            if self._can_draw():
                for t,n in enumerate(p.hand):
                    if self.profile==ANHUI and t==RED:continue
                    if n==4:acts.append(('ankan',t))
                    if n>=1 and ('peng',t) in p.melds:
                        if not self.reserve or t==self.drawn[self.cur]:acts.append(('bugang',t))
        return self.cur,acts

    def _settle(self,winners,robber=None):
        self.winners=winners[:]
        need={}
        for i in winners:
            p=self.players[i];h=p.hand[:]
            if robber is not None:h[self.rob_tile]+=1
            self.revealed[i]=h
            has_red=h[RED]>0 or any(t==RED for _,t in p.melds)
            if self.reserve:need[i]=self.reserve
            elif self.profile==ONE_MA:need[i]=1
            else:need[i]=2+(not has_red)
        bonus_end=0 if self.reserve else -1
        self.bonus=[self.wall.pop(bonus_end) for _ in range(min(max(need.values()),len(self.wall)))]
        for i in winners:
            tiles=self.bonus[:need[i]]
            if self.profile==ANHUI and tiles and len(tiles)<need[i]:
                tiles += [tiles[-1]]*(need[i]-len(tiles))
            if self.profile==ONE_MA:
                hits=len(tiles)
                extra=sum(11 if t==RED else t%9+2 for t in tiles)
            else:
                hits=sum(t==RED or t%9 in (0,4,8) for t in tiles)
                extra=2*hits
            amount=2+extra
            self.bonus_details[i]={'tiles':tiles[:],'hits':hits,'per_loser':amount,'extra':extra}
            if robber is None:
                for j in range(4):
                    if j!=i:self._pay(j,i,amount)
            else:self._pay(robber,i,3*amount)
        if robber is not None:
            self.players[robber].hand[self.rob_tile]-=1;self.claimed.append(self.rob_tile)
        self.next_dealer=robber if robber is not None and self.profile==FANGCHENGGANG else winners[0]
        self.phase='over'
        self.reason_code='rob_kong_win' if robber is not None else 'four_red' if self.revealed[winners[0]][RED]==4 else 'hongzhong_win'

    def step(self,action):
        actor,legal=self.legal_actions()
        if action not in legal:raise ValueError('Illegal Hongzhong action')
        self.steps+=1
        if self.steps>1500:raise RuntimeError('Hongzhong turn guard reached')
        k,t=action;p=self.players[actor]
        if self.phase=='rob_kong':
            self.queue.pop(0)
            if k=='ron':
                self.rob_winners.append(actor)
                if self.profile==ANHUI:self.queue.clear()
            if not self.queue:
                if self.rob_winners:self._settle(self.rob_winners,self.rob_actor)
                else:self._finish_added_kong()
        elif self.phase=='response':
            self.queue.pop(0)
            if k in ('peng','zhigang') and self.pending_claim is None:self.pending_claim=(actor,k)
            if not self.queue:
                dp,dt=self.pending_discard
                if self.pending_claim is None:self._draw((dp+1)%4)
                else:
                    who,claim=self.pending_claim;q=self.players[who]
                    q.hand[dt]-=2 if claim=='peng' else 3
                    q.melds.append(('peng' if claim=='peng' else 'gang_ming',dt));self.discards[dp].pop()
                    self.cur=who;self.draw_turn=False;self.phase='action'
                    if claim=='zhigang':self._pay(dp,who,3,True);self._draw(who,replacement=True)
                self.pending_claim=None
        elif k=='discard':
            p.hand[t]-=1;self.discards[actor].append(t);self.draw_turn=False
            self.pending_discard=(actor,t);self.pending_claim=None
            if not self._can_draw():self._draw((actor+1)%4)
            else:self.queue=[(actor+i)%4 for i in (1,2,3)];self.phase='response'
        elif k=='zimo':self._settle([actor])
        elif k=='ankan':
            p.hand[t]-=4;p.melds.append(('gang_an',t))
            for j in range(4):
                if j!=actor:self._pay(j,actor,2,True)
            self._draw(actor,replacement=True)
        elif k=='bugang':
            self.rob_actor=actor;self.rob_tile=t;self.rob_winners=[]
            self.queue=[]
            for i in (1,2,3):
                if self.profile==FANGCHENGGANG and t==RED:break
                j=(actor+i)%4;q=self.players[j];h=q.hand[:];h[t]+=1
                if self.profile==ANHUI and h[RED]:continue
                if can_win(h,len(q.melds),four_red=False,seven_pairs=self.seven_pairs):self.queue.append(j)
            if self.queue:self.phase='rob_kong'
            else:self._finish_added_kong()
        assert sum(self.scores())==0
        assert self.physical_counts()==Counter({t:4 for t in range(28)})

    def _finish_added_kong(self):
        p=self.players[self.rob_actor];t=self.rob_tile
        p.hand[t]-=1;p.melds.remove(('peng',t));p.melds.append(('gang_bu',t))
        for j in range(4):
            if j!=self.rob_actor:self._pay(j,self.rob_actor,1,True)
        self._draw(self.rob_actor,replacement=True)
        self.rob_actor=None;self.rob_tile=None;self.rob_winners=[]
