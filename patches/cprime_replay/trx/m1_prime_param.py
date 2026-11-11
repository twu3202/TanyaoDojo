"""Submit the run's state.pth weights to the online server, exactly as train.py does at startup
(submit_param(mortal, dqn, is_idle=True)), so workers can fill the buffer before the trainer starts."""
import torch
from common import submit_param
from config import config
from model import Brain, DQN

state = torch.load(config['control']['state_file'], weights_only=True, map_location='cpu')
version = config['control']['version']
mortal = Brain(version=version, **config['resnet'])
dqn = DQN(version=version)
mortal.load_state_dict(state['mortal'])
dqn.load_state_dict(state['current_dqn'])
submit_param(mortal, dqn, is_idle=True)
print(f"param submitted: steps {state['steps']}")
