"""Generate with the existing JAX env, verify with the Torch serving env. CPU only."""
import argparse
from pathlib import Path
import random
import numpy as np
p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','verify']);p.add_argument('--assets',default='runs/mvp_assets/hf');p.add_argument('--output',default='runs/mvp_assets/parity.npz');a=p.parse_args()
if a.mode=='generate':
 import jax,jax.numpy as jnp
 from sichuan import env_jax as E
 from sichuan.obs import observe
 from sichuan.net import SichuanACNet
 from deploy.play.sichuan_game import WebGame
 from deploy.play.sichuan_model import observation
 weights={}
 for name,val in dict(np.load(Path(a.assets)/'sichuan.npz')).items():
  keys=name.split('/');node=weights
  for key in keys[:-1]:node=node.setdefault(key,{})
  node[keys[-1]]=jnp.asarray(val)
 init=jax.jit(E._init_from_wall);encode=jax.jit(observe);forward=jax.jit(SichuanACNet().apply)
 g=WebGame(101);rng=random.Random(919);planes=[];scalars=[];masks=[]
 maxdiff=0;after_pon=0
 for step in range(400):
  if len(planes)>=160:break
  if g.phase=='over':g=WebGame(step+101)
  who,acts=g.legal_actions()
  if g.phase=='action' and g.drawn_tile is None:after_pon+=1
  o=observation(g,who)
  mk=np.zeros((4,4),np.int8);mt=np.full((4,4),-1,np.int8)
  for i,q in enumerate(g.players):
   for k,(kind,t) in enumerate(q.melds):
    mk[i,k]={'peng':1,'gang_ming':2,'gang_an':3,'gang_bu':4}[kind];mt[i,k]=t
  st=init(jnp.repeat(jnp.arange(27,dtype=jnp.int8),4)).replace(hand=jnp.array([q.hand for q in g.players],jnp.int8),
      phase=jnp.int32(E.PH_ACTION if g.phase=='action' else E.PH_RESP),current_player=jnp.int32(who),
      drawn_tile=jnp.int32(-1 if g.drawn_tile is None else g.drawn_tile),
      void=jnp.array([-1 if q.void is None else q.void for q in g.players],jnp.int8),
      river=jnp.array([[r.count(t) for t in range(27)] for r in g.discards],jnp.int8),
      melds_kind=jnp.array(mk),melds_tile=jnp.array(mt),n_melds=jnp.array([len(q.melds) for q in g.players],jnp.int8),
      score=jnp.array(g.scores(),jnp.int32),finished=jnp.array([q.hu for q in g.players]),n_hu=jnp.int32(len(g.hu_order)),
      wall_ix=jnp.int32(108-len(g.wall)),last_draw_was_gang=jnp.bool_(g.last_draw_was_gang),hai_di=jnp.bool_(getattr(g,'hai_di',False)))
  expected=encode(st)
  for key in o:
   diff=float(np.abs(o[key]-np.asarray(expected[key])).max());maxdiff=max(maxdiff,diff)
   np.testing.assert_allclose(o[key],expected[key],atol=1e-6,rtol=0)
  planes.append(o['planes']);scalars.append(o['scalars'])
  from deploy.play.sichuan_game import action_id
  mask=np.zeros(61,bool)
  for act in acts:mask[action_id(act)]=True
  masks.append(mask)
  wins=[x for x in acts if x[0] in ('ron','zimo')];g.step(rng.choice(wins or acts))
 obs={'planes':jnp.array(planes),'scalars':jnp.array(scalars)}
 logits,value=forward(weights,obs)
 np.savez(a.output,planes=np.array(planes),scalars=np.array(scalars),logits=np.asarray(logits),value=np.asarray(value),mask=np.array(masks))
 assert after_pon, 'expected post-pung states in the replay'
 print('samples',len(planes),'encoder max diff',maxdiff,'post-pung states',after_pon)
else:
 import torch
 from deploy.play.sichuan_model import SichuanModel
 torch.set_num_threads(2)
 z=np.load(a.output);m=SichuanModel(Path(a.assets)/'sichuan.npz')
 logits,value=m.forward({'planes':z['planes'],'scalars':z['scalars']})
 np.testing.assert_allclose(logits,z['logits'],atol=1e-4,rtol=2e-4)
 np.testing.assert_allclose(value,z['value'],atol=1e-4,rtol=2e-4)
 actual=np.where(z['mask'],logits,-np.inf).argmax(-1);expected=np.where(z['mask'],z['logits'],-np.inf).argmax(-1)
 assert np.array_equal(actual,expected)
 print('max logits diff',float(np.abs(logits-z['logits']).max()),'max value diff',float(np.abs(value-z['value']).max()),'legal choices agree',len(actual),'/',len(actual))
