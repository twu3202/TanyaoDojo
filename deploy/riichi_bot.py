#!/usr/bin/env python
"""riichi.dev (RiichiLab) 的 Mortal 壳。

两种模式:
  离线自检(不需要账号):
    python riichi_bot.py --weights /path/to/mortal.pth --selfcheck 4
  在线(验证 / 天梯):
    python riichi_bot.py --weights ... --online validate
    python riichi_bot.py --weights ... --online ranked --loop
Token 读取顺序:env RIICHI_DEV_TOKEN > ~/.config/riichi_dev_token 文件。
依赖:pip install riichienv websockets;以及 Better_mortal/Mortal/mortal 里的
model.py/engine.py + 编译好的 libriichi(通过 sys.path 注入)。
"""
import argparse
import json
import os
import sys
import pathlib

MORTAL_DIR = pathlib.Path(__file__).resolve().parent.parent / 'Mortal' / 'mortal'
sys.path.insert(0, str(MORTAL_DIR))

import torch  # noqa: E402
from model import Brain, DQN  # noqa: E402
from engine import MortalEngine  # noqa: E402
from libriichi.mjai import Bot  # noqa: E402
from riichienv import RiichiEnv, Observation  # noqa: E402


def load_engine(weights: str, device_str: str) -> MortalEngine:
    device = torch.device(device_str)
    state = torch.load(weights, weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    brain = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        brain, dqn,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=False,
        enable_quick_eval=True,
        enable_rule_based_agari_guard=True,
        name='better-mortal',
    )


class MortalAgent:
    """riichienv Observation <-> libriichi mjai Bot 桥。每局一个实例。"""

    def __init__(self, engine: MortalEngine, seat: int):
        self.bot = Bot(engine, seat)

    def act(self, obs: Observation):
        resp = None
        for ev in obs.new_events():  # 每座位增量 mjai 事件流,不要自己再维护游标
            resp = self.bot.react(ev)
        if resp is None:
            return self._fallback(obs)
        # 不匹配合法动作时 select_action_from_mjai 返回 None(不抛异常)
        try:
            action = obs.select_action_from_mjai(resp)
        except Exception:
            action = None
        if action is None:
            try:
                action = obs.select_action_from_mjai(json.loads(resp))
            except Exception:
                action = None
        return action if action is not None else self._fallback(obs)

    @staticmethod
    def _fallback(obs: Observation):
        """Mortal 无响应/转换失败时的兜底:摸切 > 第一个合法动作。"""
        legal = obs.legal_actions()
        drawn = obs.drawn_tile
        if callable(drawn):
            drawn = drawn()
        for a in legal:
            d = a.to_dict() if hasattr(a, 'to_dict') else {}
            if drawn is not None and d.get('tile') == drawn:
                return a
        return legal[0]


class TsumogiriAgent:
    """基线对手:永远摸切。"""

    def act(self, obs: Observation):
        return MortalAgent._fallback(obs)


def selfcheck(engine: MortalEngine, games: int, seed0: int = 7):
    """本地 RiichiEnv 打 N 个半庄:座位 0 = Mortal,其余摸切。"""
    ranks_hist = [0, 0, 0, 0]
    for g in range(games):
        env = RiichiEnv(game_mode=2, seed=seed0 + g)
        agents = {0: MortalAgent(engine, 0)}
        for pid in (1, 2, 3):
            agents[pid] = TsumogiriAgent()
        observations = env.get_observations()
        guard = 0
        while not env.done():
            acted = False
            for pid, obs in observations.items():
                if obs.legal_actions():
                    observations = env.step({pid: agents[pid].act(obs)})
                    acted = True
                    break
            guard += 1
            if not acted or guard > 20000:
                raise RuntimeError(f'game {g}: stuck (acted={acted}, guard={guard})')
        rank0 = env.ranks()[0]  # 1-indexed
        ranks_hist[rank0 - 1] += 1
        print(f'game {g}: scores={env.scores()} mortal_rank={rank0}')
    n = sum(ranks_hist)
    avg = sum((i + 1) * c for i, c in enumerate(ranks_hist)) / n
    print(f'\nselfcheck vs 3x tsumogiri: rank_hist={ranks_hist} avg_rank={avg:.3f} (期望 << 2.5)')


def read_token() -> str:
    tok = os.environ.get('RIICHI_DEV_TOKEN', '').strip()
    if tok:
        return tok
    p = pathlib.Path.home() / '.config' / 'riichi_dev_token'
    if p.exists():
        return p.read_text().strip()
    raise SystemExit('没有找到 token:设 RIICHI_DEV_TOKEN 或写入 ~/.config/riichi_dev_token')


async def play_online(engine: MortalEngine, endpoint: str):
    import websockets
    token = read_token()
    import asyncio
    url = f'wss://game.riichi.dev/ws/{endpoint}'
    agent = None
    n_actions = 0
    game_over = False
    async with websockets.connect(url, additional_headers={'Authorization': f'Bearer {token}'}) as ws:
        print(f'connected: {url}')
        while True:
            try:
                # end_game 之后再等最多 10s,收 validation_result 等尾部消息
                raw = await asyncio.wait_for(ws.recv(), timeout=10 if game_over else 120)
            except asyncio.TimeoutError:
                if game_over:
                    break
                print('recv timeout (120s), closing')
                break
            except websockets.ConnectionClosed:
                break
            msg = json.loads(raw)
            mtype = msg.get('type')
            if mtype == 'start_game':
                seat = msg.get('id', msg.get('seat', 0))
                agent = MortalAgent(engine, seat)
                print(f'start_game, seat={seat}')
            elif mtype == 'request_action':
                obs = Observation.deserialize_from_base64(msg['observation'])
                if agent is None:  # 保险:错过 start_game 时按 obs 的座位重建
                    pid = obs.player_id() if callable(obs.player_id) else obs.player_id
                    agent = MortalAgent(engine, pid)
                action = agent.act(obs)
                reply = action.to_mjai()
                if isinstance(reply, str):
                    reply = json.loads(reply)
                reply['request_id'] = msg['request_id']
                await ws.send(json.dumps(reply))
                n_actions += 1
            elif mtype == 'end_game':
                print(f'end_game after {n_actions} actions: {msg}')
                game_over = True
            elif mtype in ('validation_result', 'error'):
                print(f'{mtype}: {msg}')
                if mtype == 'error' or (mtype == 'validation_result' and game_over):
                    break
            # 其余广播事件(非 request_action)由 obs.new_events() 兜底重放,忽略即可
    print(f'session done: {n_actions} actions sent')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--device', default='cpu', help='cpu / mps / cuda:0')
    ap.add_argument('--selfcheck', type=int, metavar='N', help='本地打 N 个半庄自检')
    ap.add_argument('--online', choices=['validate', 'ranked'])
    ap.add_argument('--loop', action='store_true', help='ranked 模式打完一局自动重连')
    args = ap.parse_args()

    engine = load_engine(args.weights, args.device)
    print(f'engine loaded: {args.weights} on {args.device}')

    if args.selfcheck:
        selfcheck(engine, args.selfcheck)
    if args.online:
        import asyncio
        import time
        while True:
            try:
                asyncio.run(play_online(engine, args.online))
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f'connection error: {e}; retry in 10s')
                time.sleep(10)
                continue
            if not (args.online == 'ranked' and args.loop):
                break
            time.sleep(2)


if __name__ == '__main__':
    main()
