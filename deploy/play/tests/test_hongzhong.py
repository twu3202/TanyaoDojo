import copy,itertools,random,time,unittest
from collections import Counter
from deploy.play.hongzhong import HongzhongGame,Player,can_win,standard_win,RED,TWO_MA,ONE_MA,FANGCHENGGANG
from deploy.play.hongzhong_game import HongzhongSession,policy_view,choose
from sichuan.reference_impl import _can_form_sets


def hand(tiles):return tuple(tiles.count(t) for t in range(28))


def ordinary(h,melds):
    for t,n in enumerate(h):
        if n>=2:
            c=list(h);c[t]-=2
            if _can_form_sets(c,4-melds):return True
    return False


def assignment_oracle(h,melds):
    for assignment in itertools.combinations_with_replacement(range(27),h[RED]):
        c=list(h[:27])
        for t in assignment:c[t]+=1
        if ordinary(c,melds):return True
    return False


def fixture(hands,melds=None,*,profile=TWO_MA):
    g=HongzhongGame(5,profile=profile)
    g.players=[Player(list(hand(x)),list((melds or {}).get(i,[]))) for i,x in enumerate(hands)]
    used=Counter(t for h in hands for t in h)
    for p in g.players:
        for k,t in p.melds:used[t]+=3 if k=='peng' else 4
    assert all(c<=4 for c in used.values()),used
    g.wall=[t for t in range(28) for _ in range(4-used[t])]
    g.discards=[[] for _ in range(4)];g.bonus=[];g.claimed=[];g.cur=0;g.draw_turn=True;g.phase='action';g.drawn={0:hands[0][-1]};g.last_drawer=0
    return g


def tail(g,tiles):
    for t in tiles:g.wall.remove(t)
    g.wall.extend(reversed(tiles))


class WildcardTests(unittest.TestCase):
    def test_wildcard_and_seven_pair_boundaries(self):
        regular=[0,1,2,3,4,5,9,10,11,18,19,20,7,7]
        self.assertTrue(can_win(hand(regular)))
        self.assertTrue(can_win(hand([RED,RED]+regular[2:])))
        pairs=[0,0,2,2,4,4,6,6,8,8,10,10,13,13]
        self.assertFalse(can_win(hand(pairs)))
        special=[RED]*4+[0,2,4,6,8,9,11,13,15,17]
        self.assertTrue(can_win(hand(special)))
        self.assertFalse(can_win(hand(special),four_red=False))
        self.assertFalse(can_win(hand(special[:-1])))

    def test_solver_against_wildcard_assignment_oracle(self):
        rng=random.Random(919)
        for wild in range(3):
            for _ in range(18):
                tiles=[]
                while len(tiles)<14-wild:
                    t=rng.randrange(27)
                    if tiles.count(t)<4:tiles.append(t)
                h=hand(tiles+[RED]*wild)
                self.assertEqual(standard_win(h),assignment_oracle(h,0),h)
        # Positive generated complete hands, replacing up to two natural tiles by wildcards.
        for _ in range(25):
            tiles=[]
            for k in range(4):
                start=rng.randrange(3)*9+rng.randrange(7);tiles.extend([start,start+1,start+2])
            pair=rng.randrange(27);tiles.extend([pair,pair])
            if max(Counter(tiles).values())>4:continue
            for j in range(rng.randrange(3)):tiles[j]=RED
            h=hand(tiles)
            self.assertTrue(standard_win(h));self.assertTrue(assignment_oracle(h,0))


class RuleTests(unittest.TestCase):
    def rob_fixture(self,red_first=False,*,profile=TWO_MA):
        ready=[1,2,3,4,5,9,10,11,12,13,14,17,17]
        first=ready[:]
        if red_first:first[0]=RED
        return fixture([[6,6,7,7,8,8,15,15,16,16,0],first,ready,[18]*4+[19]*4+[20]*4+[21]],
                       {0:[('peng',0)]},profile=profile)

    def winning_fixture(self,profile):
        winning=[0,1,2,3,4,5,9,10,11,18,19,20,7,7]
        return fixture([winning,[6]*4+[8]*4+[12]*4+[13],[13]*3+[14]*4+[15]*4+[16]*2,
                        [16]*2+[17]*4+[21]*4+[22]*3],profile=profile)

    def test_robbed_added_kong_multiple_winners_shared_bonus(self):
        g=self.rob_fixture(red_first=True);tail(g,[10,11,9])
        g.step(('bugang',0));self.assertEqual(g.phase,'rob_kong');self.assertEqual(g.gang_ledger,[])
        g.step(('ron',None));g.step(('ron',None))
        self.assertEqual(g.reason_code,'rob_kong_win');self.assertEqual(g.scores(),[-18,6,12,0]);self.assertEqual(g.bonus,[10,11,9])
        self.assertEqual(g.next_dealer,1);self.assertEqual(g.claimed,[0]);self.assertEqual(g.players[0].melds,[('peng',0)])
        self.assertEqual(g.physical_counts(),Counter({i:4 for i in range(28)}))

    def test_passing_rob_allows_added_kong(self):
        g=self.rob_fixture();g.step(('bugang',0));g.step(('pass',None));g.step(('pass',None))
        self.assertEqual(g.scores(),[3,-1,-1,-1]);self.assertEqual(g.cur,0)
        self.assertIn(('gang_bu',0),g.players[0].melds);self.assertTrue(g.draw_turn)
        self.assertIsNone(g.rob_actor)

    def test_self_draw_and_short_bonus_wall(self):
        g=self.winning_fixture(TWO_MA)
        tail(g,[RED,10]);extra=g.wall[:-2];g.discards[3].extend(extra);g.wall=g.wall[-2:]
        g.step(('zimo',None))
        self.assertEqual(g.bonus,[RED,10]);self.assertEqual(g.scores(),[12,-4,-4,-4])
        self.assertEqual(g.next_dealer,0)

    def test_one_ma_uses_rank_instead_of_hits(self):
        for tile,extra in [(10,3),(26,10),(RED,11),(None,0)]:
            with self.subTest(tile=tile):
                g=self.winning_fixture(ONE_MA)
                if tile is None:
                    g.discards[3].extend(g.wall);g.wall=[]
                else:tail(g,[tile])
                g.step(('zimo',None))
                self.assertEqual(g.bonus,[] if tile is None else [tile])
                self.assertEqual(g.bonus_details[0]['extra'],extra)
                self.assertEqual(g.scores(),[3*(2+extra)]+[-2-extra]*3)

    def test_fangchenggang_seven_pairs_and_reserve(self):
        pairs=[0,0,2,2,4,4,6,6,8,8,10,10,13,RED]
        self.assertTrue(can_win(hand(pairs),seven_pairs=True))
        self.assertFalse(can_win(hand(pairs)))
        self.assertFalse(can_win(hand(pairs[:-1]+[14]),seven_pairs=True))
        g=self.winning_fixture(FANGCHENGGANG)
        marks=[0,1,RED,19]
        for tile in marks:g.wall.remove(tile)
        g.wall=marks+g.wall
        g.step(('zimo',None))
        self.assertEqual(g.bonus,marks)
        self.assertEqual(g.scores(),[18,-6,-6,-6])
        g=self.rob_fixture(profile=FANGCHENGGANG)
        g.discards[3].extend(g.wall[4:]);g.wall=g.wall[:4]
        self.assertNotIn(('bugang',0),g.legal_actions()[1]);g.step(('discard',6))
        self.assertEqual(g.phase,'over');self.assertEqual(len(g.wall),4)
        self.assertEqual(g.next_dealer,0)

    def test_fangchenggang_immediate_kong_and_robbed_dealer(self):
        g=self.rob_fixture(profile=FANGCHENGGANG)
        g.drawn[0]=16
        self.assertNotIn(('bugang',0),g.legal_actions()[1])
        g.drawn[0]=0
        g.step(('bugang',0));g.step(('ron',None));g.step(('ron',None))
        self.assertEqual(g.winners,[1,2]);self.assertEqual(g.next_dealer,0)
        self.assertEqual(len(g.bonus),4)
        g=self.rob_fixture(profile=FANGCHENGGANG)
        replacement=g.wall[0]
        g.step(('bugang',0));g.step(('pass',None));g.step(('pass',None))
        self.assertEqual(g.drawn[0],replacement)
        self.assertEqual(g.scores(),[3,-1,-1,-1])

    def test_draw_keeps_kong_payments_and_last_drawer(self):
        g=self.rob_fixture();g._pay(1,0,3,True)
        g.discards[3].extend(g.wall);g.wall=[];g.last_drawer=2;g.draw_turn=False
        self.assertFalse(any(a[0] in ('ankan','bugang') for a in g.legal_actions()[1]))
        g.step(('discard',6))
        self.assertEqual(g.phase,'over');self.assertEqual(g.next_dealer,2);self.assertEqual(g.scores(),[3,-3,0,0])

    def test_no_discard_win_or_wildcard_substitution_in_calls(self):
        g=self.rob_fixture();g.players[0].melds=[]
        # Return the former meld to a public zone, keeping a physical four-copy wall.
        g.discards[0]=[0]*3
        g.step(('discard',0))
        self.assertNotIn(('ron',None),g.legal_actions()[1]);self.assertNotIn(('peng',None),g.legal_actions()[1])
        with self.assertRaises(ValueError):g.step(('ron',None))

    def test_pon_requires_discard_and_cannot_fake_self_draw(self):
        g=HongzhongGame(1)
        # Search seeded legal random paths until a pon is offered.
        rng=random.Random(22);found=False
        for seed in range(20):
            g=HongzhongGame(seed)
            while g.phase!='over':
                _,acts=g.legal_actions()
                if ('peng',None) in acts:
                    actor=g.legal_actions()[0];g.step(('peng',None))
                    while g.phase=='response':g.step(('pass',None))
                    self.assertEqual(g.cur,actor);self.assertFalse(g.draw_turn)
                    self.assertTrue(all(a[0]=='discard' for a in g.legal_actions()[1]));found=True;break
                wins=[a for a in acts if a[0]=='zimo'];g.step(rng.choice(wins or acts))
            if found:break
        self.assertTrue(found)

    def test_public_policy_invariant(self):
        g=HongzhongGame(91);i,acts=g.legal_actions();before=policy_view(g,i,acts)
        for j,p in enumerate(g.players):
            if j!=i:p.hand=list(reversed(p.hand))
        g.wall.reverse()
        self.assertEqual(before,policy_view(g,i,acts))
        self.assertEqual(choose(before),choose(policy_view(g,i,acts)))

    def test_random_games_terminate_and_conserve(self):
        for profile in (TWO_MA,ONE_MA,FANGCHENGGANG):
            for seed in range(200):
                g=HongzhongGame(seed,dealer=seed%4,profile=profile);rng=random.Random(seed+777)
                while g.phase!='over':
                    _,acts=g.legal_actions();wins=[a for a in acts if a[0] in ('zimo','ron')]
                    g.step(rng.choice(wins or acts))
                self.assertEqual(sum(g.scores()),0)
                self.assertEqual(g.physical_counts(),Counter({i:4 for i in range(28)}))

    def test_complete_web_hand_and_next_dealer(self):
        s=HongzhongSession(seed=7,pacing=0);s.begin();s.auto();until=time.monotonic()+40
        while s.status not in ('round_end','error') and time.monotonic()<until:time.sleep(.01)
        self.assertEqual(s.status,'round_end',s.error);dealer=s.dealer;scores=s.totals[:]
        s.begin();until=time.monotonic()+5
        while s.status not in ('your_turn','error') and time.monotonic()<until:time.sleep(.01)
        self.assertEqual(s.snapshot()['table']['dealer'],dealer)
        self.assertEqual(sum(s.snapshot()['result']['scores']) if s.result else sum(scores),0);s.close()



class SharedWebBoundaryTests(unittest.TestCase):
    def test_sichuan_pon_cannot_be_reclassified_as_self_draw(self):
        from deploy.play.sichuan_game import WebGame
        from deploy.play.sichuan_model import observation
        g=WebGame(0)
        hands=[[18]*4+[19]*4+[20]*4+[21], [0,0,1,2,3,9,10,11,12,13,14,7,7],
               [4]*4+[5]*4+[6]*4+[8], [15]*4+[16]*4+[17]*4+[22]]
        c=Counter([0])
        for p,tiles in zip(g.players,hands):
            p.hand=[tiles.count(t) for t in range(27)];p.void=2;c.update(tiles)
        self.assertTrue(all(n<=4 for n in c.values()))
        g.wall=[t for t in range(27) for _ in range(4-c[t])];g.discards=[[0],[],[],[]]
        g.pending_discard=(0,0);g.response_queue=[1,2,3];g.ron_winners=[];g.pending_claim=None;g.phase='response'
        g.step(('peng',None));g.step(('pass',None));g.step(('pass',None))
        self.assertIsNone(g.drawn_tile)
        self.assertTrue(all(a[0]=='discard' for a in g.legal_actions()[1]))
        with self.assertRaises(ValueError):g.step(('zimo',None))
        self.assertEqual(float(observation(g,1)['planes'][:,4].sum()),0)
        self.assertEqual(g.tile_conservation(),108)


if __name__=='__main__':unittest.main()
