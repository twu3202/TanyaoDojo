"""CPU inference over the already trained checkpoint. No training in the Space."""
from pathlib import Path
import os
import sys
import threading
import numpy as np

TORCH_LOCK = threading.Lock()


class RiichiModel:
    def __init__(self,assets):
        import torch
        import importlib.util
        assets=Path(assets)
        spec=importlib.util.spec_from_file_location('tanyao_mortal_model',assets/'model.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        Brain,DQN=module.Brain,module.DQN
        torch.set_num_threads(int(os.environ.get('TANYAO_TORCH_THREADS','2')))
        st=torch.load(assets/'model.pth',weights_only=True,map_location='cpu')
        cfg=st['config'];self.version=cfg['control'].get('version',4)
        self.brain=Brain(version=self.version,conv_channels=cfg['resnet']['conv_channels'],num_blocks=cfg['resnet']['num_blocks']).eval()
        self.dqn=DQN(version=self.version).eval()
        self.brain.load_state_dict(st['mortal']);self.dqn.load_state_dict(st['current_dqn'])

    def choose(self,ps,opts):
        import torch
        from .riichi_game import TILES
        obs,mask=ps.encode_obs(self.version,False)
        with TORCH_LOCK,torch.inference_mode():
            q=self.dqn(self.brain(torch.as_tensor(obs[None])),torch.as_tensor(mask[None]))[0].numpy()
            if mask[42] and not ps.last_cans.can_daiminkan:
                ko,km=ps.encode_obs(self.version,True)
                kq=self.dqn(self.brain(torch.as_tensor(ko[None])),torch.as_tensor(km[None]))[0].numpy()
            else:kq=None
        def key(i):
            opt=opts[i];score=float(q[opt['index']])
            secondary=0.0
            if kq is not None and opt['index']==42:
                t=opt['tiles'][0].rstrip('r');secondary=float(kq[TILES.index(t)])
            return score,secondary
        return max(range(len(opts)),key=key)
