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
    reveal_winners = True

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

    def _can_rob(self, seat, tile):
        player = self.players[seat]
        hand = player.hand[:]
        hand[tile] += 1
        return self.alive(seat) and is_hu(hand, len(player.melds), player.void)

    def _finish_added_kong(self, actor, tile):
        player = self.players[actor]
        player.hand[tile] -= 1
        player.melds.remove(('peng', tile))
        player.melds.append(('gang_bu', tile))
        self._gang_money(actor, 'gang_bu')
        self._begin_turn(actor, from_gang=True)

    def step(self, action):
        if action not in self.legal_actions()[1]:
            raise ClientError('illegal_action')
        # The training reference auto-accepts every legal rob. Web users must be able to pass.
        if self.phase == 'action' and action[0] == 'bugang':
            tile = action[1]
            order = [(self.cur + offset) % 4 for offset in (1, 2, 3)]
            robbers = [seat for seat in order if self._can_rob(seat, tile)]
            self.turn_guard += 1
            if robbers:
                self.rob_actor,self.rob_tile,self.rob_queue,self.rob_winners=self.cur,tile,robbers,[]
                self.drawn_tile = None
                self.phase = 'rob_kong'
            else:
                self._finish_added_kong(self.cur, tile)
            return
        if self.phase == 'rob_kong':
            self.turn_guard+=1
            i=self.rob_queue.pop(0);tile=self.rob_tile;actor=self.rob_actor
            if action[0]=='ron':
                h=self.players[i].hand[:];h[tile]+=1
                self._settle_hu(i,h,{'qiang_gang':True},discarder=actor);self.rob_winners.append(i)
            elif action[0]!='pass':raise ValueError('抢杠窗口只允许胡牌或过')
            if self.rob_queue:return
            if self.rob_winners:
                self.players[actor].hand[tile] -= 1
                self.discards[actor].append(tile)
                self._begin_turn(self._next_alive(actor))
            else:
                self._finish_added_kong(actor, tile)
            return
        super().step(action)


def public_features(g, me, acts):
    """Explicit policy boundary: own tiles, public events, public hand lengths only."""
    p = g.players[me]
    public = Counter(t for river in g.discards for t in river)
    for q in g.players:
        if q.hu and g.reveal_winners:
            public.update({t:c for t,c in enumerate(q.hand) if c})
        for kind,t in q.melds:
            public[t] += 4 if kind.startswith('gang') else 3
    # The first ordinary discard winner physically holds the shared tile.
    # Robbed tiles are already visible in the kong maker's river.
    public.update(getattr(g, 'shown_claimed_discards', ()))
    return {'hand': p.hand[:], 'melds': p.melds[:], 'void': p.void, 'actions': list(acts),
            'public': dict(public), 'opponents': [{'void':q.void,'finished':q.hu,'melds':len(q.melds)} for j,q in enumerate(g.players) if j != me]}


def heuristic(view):
    hand, melds, void, acts = view['hand'], view['melds'], view['void'], view['actions']
    allow_pairs = view.get('allow_seven_pairs', True)
    for a in acts:
        if a[0] in ('ron','zimo'):
            return a
    if acts[0][0] == 'exchange':
        def rank_exchange(action):
            kept = hand[:]
            for tile in action[1]:
                kept[tile] -= 1
            return min((distance(kept, 0, suit), sum(kept[suit * 9:suit * 9 + 9]), action[1])
                       for suit in range(3))

        return min(acts, key=rank_exchange)
    if acts[0][0] == 'void':
        return ('void', min(range(3), key=lambda s: sum(hand[s*9:s*9+9])))
    discards = [a for a in acts if a[0] == 'discard']
    if discards:
        scores = []
        for a in discards:
            h = hand[:];h[a[1]]-=1
            scores.append((distance(h,len(melds),void,allow_seven_pairs=allow_pairs),a,h))
        best=min(x[0] for x in scores)
        if view.get('own_kong'):
            for kind, tile in acts:
                if kind not in ('ankan', 'bugang'):
                    continue
                h = hand[:]
                h[tile] -= 4 if kind == 'ankan' else 1
                if distance(h, len(melds) + (kind == 'ankan'), void) <= best:
                    return kind, tile
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
    def __init__(self, policy='heuristic', *, model=None, seed=None, pacing=.12, variant='sichuan'):
        super().__init__(variant,policy,seed=seed,pacing=pacing)
        self.model=model
        self.round_no=0
        self.totals=[0]*4
        self.game=None
        self.dealer = 0

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
            seed = self.seed + self.round_no
            if self.variant == 'sichuan':
                self.game = WebGame(seed)
            else:
                from .blood_battle import BloodBattleGame
                self.game = BloodBattleGame(seed, self.variant, self.dealer)
            opening = 'hand_start' if self.game.phase == 'exchange' else 'hand_start_void'
            self.log_event(opening, {'number': self.round_no}, 'round')
            self.start_worker(self.play_round)

    def choice(self,g,i,acts):
        v=public_features(g,i,acts)
        v['own_kong'] = self.variant != 'sichuan'
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
            revealed = i == 0 or (p.hu and g.reveal_winners)
            public_win = getattr(g, 'winning_discards', {}).get(i)
            players.append({'id':i,'name':NAMES[i],'score':self.totals[i]+p.score_delta,'delta':p.score_delta,
                'hand':tiles if revealed else [], 'hand_count':len(tiles) - int(not revealed and public_win is not None),
                'revealed':revealed,'won':p.hu,'void':None if p.void is None else SUITS[p.void],'void_suit':p.void,
                'riichi':False,'river':[{'tile':tile_name(t)} for t in g.discards[i]],
                'melds':[{'kind':k,'tiles':[tile_name(t)]*(4 if k.startswith('gang') else 3)} for k,t in p.melds]})
            if not revealed and public_win is not None:
                players[-1]['win_tile'] = tile_name(public_win)
        i,_=g.legal_actions()
        if i!=0 and g.phase in ('response','rob_kong'):i=None
        return {'players':players,'hand_number':self.round_no,'round':f'Hand {self.round_no}','dealer':self.dealer,'current':i,
                'remaining':len(g.wall),'dora':[],'honba':0,'pot':0,'phase':g.phase,
                'exchange_same_suit':g.phase == 'exchange',
                'prompt':'Choose a void suit' if g.phase=='void' else 'Choose a tile or action',
                'drawn':tile_name(g.drawn_tile) if g.phase=='action' and g.cur==0 and g.drawn_tile is not None else None}

    def options_for(self,acts):
        out=[]
        for a in acts:
            k,t=a
            if k == 'exchange':
                out.append({'action':a, 'kind':k, 'label':'Exchange three', 'tiles':[tile_name(tile) for tile in t]})
                continue
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
            old_hu=len(g.hu_order)
            g.step(a)
            k,t=a
            if k=='discard':self.log_event('discard',{'actor':i,'tile':tile_name(t)})
            elif k=='void':self.log_event('void',{'actor':i,'suit':t})
            elif k=='exchange':
                if g.phase == 'void':
                    self.log_event('exchange_direction', {'direction':g.exchange_direction}, 'round')
            elif k not in ('pass','ron','zimo'):self.log_event('action',{'actor':i,'action':k},'call')
            for winner,fan,names in g.hu_order[old_hu:]:
                self.log_event('win_fan',{'actor':winner,'fan':fan,'patterns':list(names)},'win')
            assert g.tile_conservation()==108 and sum(g.scores())==0
        delta=g.scores()
        view=self.view()
        breakdown = {}
        if self.variant != 'sichuan':
            self.dealer = g.next_dealer
            breakdown = g.settlement()
        reason = 'yunnan_draw' if self.variant == 'yunnan-xuezhan' else 'sichuan_draw'
        if len(g.hu_order) >= 3:
            reason = 'blood_end'
        reason_text = {
            'blood_end': 'Three players have won.',
            'sichuan_draw': 'Wall exhausted; ready-hand payments and refunds settled.',
            'yunnan_draw': 'Wall exhausted; flower-pig, ready-hand payments and refunds settled.',
        }
        # view already includes current delta once; totals update is only for the next hand.
        self.totals=[a+b for a,b in zip(self.totals,delta)]
        with self.cv:
            self.result={'title':'Hand complete','reason':reason_text[reason],
                         'reason_code':reason,'deltas':delta,'scores':self.totals[:],
                         'winners':[w for w,_,_ in g.hu_order], 'next':True, **breakdown}
            self.log_event(self.result['reason_code'],kind='round')
            self.publish(view,'round_end')
