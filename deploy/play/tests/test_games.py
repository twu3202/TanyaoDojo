import copy
import json
import threading
import time
import unittest
from deploy.play.core import GameSession
from deploy.play.sichuan_game import SichuanSession,WebGame,public_features,heuristic
from deploy.play.riichi_game import RiichiSession,public_table,legal_options


def wait(s,statuses,limit=30):
    until=time.monotonic()+limit
    while time.monotonic()<until:
        if s.status in statuses:return
        if s.status=='error':raise AssertionError(s.error)
        time.sleep(.01)
    raise AssertionError((s.status,list(s.messages)[-3:]))


class SessionTests(unittest.TestCase):
    def test_stale_and_illegal_actions(self):
        s=GameSession('test','heuristic')
        result=[]
        t=threading.Thread(target=lambda:result.append(s.ask({},[{'action':'good','kind':'pass','label':'过'}],0)))
        t.start();wait(s,{'your_turn'})
        rev=s.version
        with self.assertRaises(ValueError):s.act(rev,'fake')
        with self.assertRaises(ValueError):s.act(rev-1,'0')
        s.act(rev,'0')
        with self.assertRaises(ValueError):s.act(rev,'0')
        t.join(2);self.assertEqual(result,['good']);s.close()

    def test_sichuan_input_hides_opponents(self):
        g=WebGame(5);g.step(('void',0));g.step(('void',0));g.step(('void',0));g.step(('void',0))
        i,acts=g.legal_actions();v=public_features(g,i,acts)
        for j,p in enumerate(g.players):
            if j!=i:p.hand=list(reversed(p.hand))
        self.assertEqual(v,public_features(g,i,acts))

    def test_sichuan_round_and_next(self):
        for seed in [1,2,3,6,8]:
            s=SichuanSession('heuristic',seed=seed,pacing=0);s.begin()
            wait(s,{'your_turn'})
            view=s.snapshot()
            self.assertNotIn('seed',view)
            for p in view['table']['players'][1:]:self.assertEqual(p['hand'],[])
            s.auto();wait(s,{'round_end'})
            self.assertEqual(sum(s.result['scores']),0)
            scores=s.result['scores'][:];s.begin();wait(s,{'your_turn'})
            self.assertEqual([p['score'] for p in s.snapshot()['table']['players']],scores)
            s.close()

    def test_riichi_view_never_contains_hidden_tiles_or_seed(self):
        e=[dict(type='start_game',seed=[999,123]),dict(type='start_kyoku',bakaze='E',kyoku=1,honba=0,kyotaku=0,oya=0,scores=[25000]*4,dora_marker='1m',tehais=[['1m']*13,['2m']*13,['3m']*13,['4m']*13]),dict(type='tsumo',actor=1,pai='9s')]
        a=public_table(e)
        e[0]['seed']=[123,456];e[1]['tehais'][1]=['8p']*13;e[2]['pai']='6s'
        self.assertEqual(a,public_table(e))
        encoded=json.dumps(a)
        self.assertNotIn('seed',encoded)
        self.assertNotIn('9s',encoded)
        self.assertEqual(a['players'][1]['hand'],[])

    def test_riichi_full_hanchan(self):
        s=RiichiSession('heuristic',seed=13,pacing=0)
        for _ in range(40):
            s.begin();s.auto();wait(s,{'round_end','finished'},90)
            self.assertEqual(sum(s.result['scores'])+s.pot*1000,100000)
            if s.status=='finished':break
        self.assertEqual(s.status,'finished');s.close()



class EdgeTests(unittest.TestCase):
    def rob_game(self):
        from collections import Counter
        g=WebGame(10)
        hands=[[0,6,6,7,7,8,8,15,15,16,16],[1,2,3,4,5,9,10,11,12,13,14,17,17],
               [18]*4+[19]*4+[20]*4+[21],[21]*3+[22]*4+[23]*4+[24]*2]
        used=Counter([0]*3)
        for p,tiles in zip(g.players,hands):
            p.hand=[tiles.count(t) for t in range(27)];p.void=2;used.update(tiles)
        self.assertTrue(all(c<=4 for c in used.values()))
        g.players[0].melds=[('peng',0)];g.wall=[t for t in range(27) for _ in range(4-used[t])]
        g.phase='action';g.cur=0;g.void_declared=4;g.drawn_tile=0
        self.assertEqual(g.tile_conservation(),108)
        return g

    def test_rob_kong_can_pass_or_win(self):
        for response in ['ron','pass']:
            g=self.rob_game();g.step(('bugang',0))
            self.assertEqual(g.phase,'rob_kong');self.assertEqual(g.legal_actions()[0],1)
            self.assertEqual(g.gang_ledger,[])
            g.step((response,None))
            self.assertEqual(g.tile_conservation(),108);self.assertEqual(sum(g.scores()),0)
            if response=='ron':
                self.assertTrue(g.players[1].hu);self.assertEqual(g.players[0].melds,[('peng',0)])
                self.assertEqual(g.gang_ledger,[]);self.assertEqual(sum(g.revealed_hands[1]),14)
            else:
                self.assertFalse(g.players[1].hu);self.assertEqual(g.players[0].melds,[('gang_bu',0)])
                self.assertEqual(len(g.gang_ledger),3)

    def test_riichi_red_chi_choices(self):
        from libriichi.state import PlayerState
        ps=PlayerState(0)
        hand=['3m','4m','5m','5mr','6m','7m','1p','1p','2p','3p','4p','E','E']
        ev=[dict(type='start_game'),dict(type='start_kyoku',bakaze='E',kyoku=4,honba=0,kyotaku=0,oya=3,scores=[25000]*4,dora_marker='9s',tehais=[hand]+[['?']*13 for _ in range(3)]),dict(type='tsumo',actor=3,pai='?'),dict(type='dahai',actor=3,pai='4m',tsumogiri=True)]
        for e in ev:ps.update(json.dumps(e))
        options=legal_options(ps,{'players':[{'melds':[]}]*4})
        middle=[x for x in options if x['index']==39]
        self.assertTrue(any('5mr' in x['action']['consumed'] for x in middle))
        self.assertTrue(any('5m' in x['action']['consumed'] for x in middle))
        for x in options:ps.validate_reaction(json.dumps(x['action']))

    def test_riichi_pot_survives_hand_boundary(self):
        s=RiichiSession('heuristic',seed=20,pacing=0)
        s.scores=[24000,24000,25000,25000];s.pot=2
        s.begin();s.auto();wait(s,{'round_end','finished'},90)
        self.assertEqual(sum(s.result['scores'])+s.pot*1000,100000);s.close()


class ConcurrencyTests(unittest.TestCase):
    def test_double_begin_does_not_spawn_two_workers(self):
        s=SichuanSession('heuristic',seed=55,pacing=0)
        with s.cv:
            s.begin()
            first=s.thread
            with self.assertRaises(ValueError):s.begin()
            self.assertIs(s.thread,first)
        s.close()


if __name__=='__main__':unittest.main()
