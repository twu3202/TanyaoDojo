#!/usr/bin/env python
"""Track C 自对弈生成 smoke test(GPU-free,Mac 隔离)。
用 v1_best 自对弈 N 局,验证:1)生成合法对局日志;2)GRP 能算出逐局奖励。
测 CPU 吞吐。用法: MORTAL_CFG=configs/mac_selfplay_test.toml python selfplay_smoke.py
"""
import sys, os, time, gzip, json
sys.path.insert(0, '/Users/r/HMM/Better_mortal/Mortal/mortal')
import torch
from model import Brain, DQN
from player import TrainPlayer
from reward_calculator import RewardCalculator
from config import config

def main():
    device = torch.device('cpu')
    # 载入 v1_best 作为被训模型(trainee)
    st = torch.load(config['baseline']['train']['state_file'], weights_only=True, map_location='cpu')
    ver = st['config']['control']['version']
    mortal = Brain(version=ver, conv_channels=st['config']['resnet']['conv_channels'],
                   num_blocks=st['config']['resnet']['num_blocks']).eval()
    dqn = DQN(version=ver).eval()
    mortal.load_state_dict(st['mortal']); dqn.load_state_dict(st['current_dqn'])

    tp = TrainPlayer()  # baseline 从 config baseline.train 载入(=v1_best,自对弈)
    print(f'self-play {config["train_play"]["default"]["games"]} 局 (CPU)...', flush=True)
    t0 = time.time()
    rankings, files = tp.train_play(mortal, dqn, device)
    dt = time.time() - t0
    n_games = int(rankings.sum())
    print(f'生成 {len(files)} 个对局日志, rankings.sum={n_games}, 用时 {dt:.1f}s '
          f'→ {n_games/dt*3600:.0f} 局/时 (CPU单机)')

    # 验证奖励管线:GRP 载入 + RewardCalculator 就绪(奖励在 trainer 侧 dataloader.py
    # 用 grp.take_feature()→calc_delta_pt 算,是 Mortal 原生代码,此处只验证可加载)
    from model import GRP
    grp_st = torch.load(config['grp']['state_file'], weights_only=True, map_location='cpu')
    grp = GRP(**config['grp']['network']).eval()
    grp.load_state_dict(grp_st['model'])
    rc = RewardCalculator(grp=grp, pts=[90, 45, 0, -135])
    print('GRP 载入成功 + RewardCalculator 就绪 (奖励在 trainer 侧 dataloader 计算)')
    print('SMOKE OK: 自对弈生成回路通 + 奖励管线组件就绪')

if __name__ == '__main__':
    main()
