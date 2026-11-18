"""Torch CPU forward pass of the existing Flax P2 checkpoint, with public-only features."""
from __future__ import annotations
import numpy as np
from .sichuan_game import distance, action_id
from .policies import TORCH_LOCK


def observation(g,me):
    rel=[(me+k)%4 for k in range(4)];p=g.players[me]
    hand=np.array(p.hand,dtype=np.float32)
    planes=[(hand>=k).astype(np.float32) for k in range(1,5)]
    drawn=np.zeros(27,np.float32)
    if g.phase == 'action' and g.cur == me and g.drawn_tile is not None:
        drawn[g.drawn_tile] = 1
    planes.append(drawn)
    river=np.array([[g.discards[i].count(t) for t in range(27)] for i in rel],np.float32)
    planes.extend(list(river/4))
    melds=np.zeros((4,27),np.float32)
    for j,i in enumerate(rel):
        for k,t in g.players[i].melds:melds[j,t]+=4 if k.startswith('gang') else 3
    planes.extend(list(melds/4))
    for i in rel:
        v=g.players[i].void
        planes.append(np.array([float(v is not None and t//9==v) for t in range(27)],np.float32))
    finished=np.array([g.players[i].hu for i in rel],np.float32)
    planes.extend([np.full(27,f,np.float32) for f in finished])
    # Match the frozen checkpoint's training encoder exactly, including its normalization order.
    planes.append((hand+river.sum(0)+melds.sum(0))/4)
    sh=distance(p.hand,len(p.melds),p.void)
    scalars=[g.players[i].score_delta/16 for i in rel]+[len(g.players[i].melds)/4 for i in rel]
    scalars += [sh/8,float(sh<=0),len(g.wall)/56,len(g.hu_order)/3]
    scalars += [float(p.void==i) for i in range(3)]+[sum(p.hand)/14]
    scalars += list(finished[1:])+[finished[0],float(g.last_draw_was_gang),float(getattr(g,'hai_di',False))]
    scalars += [sum(g.players[i].hand)/14 for i in rel]  # public hand lengths, never concealed tile identities
    return {'planes':np.stack(planes,axis=1),'scalars':np.array(scalars,np.float32)}


class SichuanModel:
    def __init__(self,path):
        import torch
        self.w={k:torch.from_numpy(v.copy()) for k,v in dict(np.load(path)).items()}
        self.blocks=len({k.split('/')[1] for k in self.w if '/ResBlock1D_' in k})

    def forward(self,obs):
        import torch
        import torch.nn.functional as F
        w=self.w
        def dense(x,p):return x@w[p+'/kernel']+w[p+'/bias']
        def conv(x,p):return F.conv1d(x.transpose(1,2),w[p+'/kernel'].permute(2,1,0),w[p+'/bias'],padding=1).transpose(1,2)
        def ln(x,p):return F.layer_norm(x,(x.shape[-1],),w[p+'/scale'],w[p+'/bias'],eps=1e-6)
        x=torch.as_tensor(obs['planes']);s=torch.as_tensor(obs['scalars'])
        if x.ndim==2:x=x[None];s=s[None]
        with TORCH_LOCK,torch.inference_mode():
            embedded=F.relu(dense(s,'params/Dense_0'))[:,None].expand(-1,27,-1)
            x=conv(torch.cat([x,embedded],dim=-1),'params/Conv_0')
            for k in range(self.blocks):
                p=f'params/ResBlock1D_{k}'
                y=conv(F.relu(ln(x,p+'/LayerNorm_0')),p+'/Conv_0')
                y=conv(F.relu(ln(y,p+'/LayerNorm_1')),p+'/Conv_1');x=x+y
            x=F.relu(ln(x,'params/LayerNorm_0')).reshape(x.shape[0],-1)
            h=F.relu(dense(x,'params/Dense_1'))
            return dense(h,'params/Dense_2').numpy(),dense(h,'params/Dense_3').numpy()[:,0]

    def choose(self,obs,acts):
        q,_=self.forward(obs);return max(acts,key=lambda a:float(q[0,action_id(a)]))
