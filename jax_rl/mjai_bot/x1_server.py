"""
X1 变体 D:集中推理服务。一个进程在 GPU 上常驻挑战者与冠军两个模型,多个竞技场进程(x1_bench --variant client)
把观测写进共享内存、经 Unix socket 发请求;服务把同一模型的请求合成一批前向,结果写回共享内存。

  python x1_server.py <challenger.pth> --clients 4 --mode lockstep --socket /tmp/x1.sock [--amp off]

mode:
  lockstep = 每轮等齐所有仍在跑的客户端对同一模型的请求,按客户端编号顺序合批:同样的输入得到同样的批,可复现
  free     = 已到的请求立即合批(vLLM 式连续批处理):批组成取决于时序,对局不可复现
协议(每个客户端每个模型一条连接):hello(client_id, model, cap) -> 服务建共享内存 obs[cap, *obs_shape] f32、
masks[cap, A] bool、q[cap, A] f32 并回名字;之后每次请求发 n、收 "ok",再读 q[:n];发 0 表示该客户端结束。
"""
from __future__ import annotations
import argparse
import os
import time
from multiprocessing import shared_memory
from multiprocessing.connection import Listener, wait

import numpy as np
import torch

import run_eval
from libriichi.consts import ACTION_SPACE, obs_shape


class Slot:
    """一个客户端对一个模型的共享缓冲与连接。"""

    def __init__(self, conn, client_id, model, cap, version):
        self.conn, self.client_id, self.model, self.cap = conn, client_id, model, cap
        shp = (cap, *obs_shape(version))
        self.shms = [shared_memory.SharedMemory(create=True, size=int(np.prod(s)) * np.dtype(d).itemsize)
                     for s, d in ((shp, np.float32), ((cap, ACTION_SPACE), np.bool_), ((cap, ACTION_SPACE), np.float32))]
        self.obs = np.ndarray(shp, np.float32, self.shms[0].buf)
        self.masks = np.ndarray((cap, ACTION_SPACE), np.bool_, self.shms[1].buf)
        self.q = np.ndarray((cap, ACTION_SPACE), np.float32, self.shms[2].buf)
        conn.send([s.name for s in self.shms])
        self.pending = 0        # 已收到、未答复的请求的 n
        self.done = False

    def close(self):
        for s in self.shms:
            s.close()
            try:
                s.unlink()
            except FileNotFoundError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--mode", choices=("lockstep", "free"), default="lockstep")
    ap.add_argument("--socket", default="/tmp/x1.sock")
    ap.add_argument("--champion", default=os.environ.get("MORTAL_V4", "/home/r/Projects/better_mortal/baseline/mortal_v4.pth"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", default="off", choices=("on", "off"))
    args = ap.parse_args()

    engines = {"champion": run_eval.load_champion(args.champion, args.device, args.amp),
               "challenger": run_eval.load_champion(args.params, args.device, args.amp)}
    dev = torch.device(args.device)
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    listener = Listener(args.socket, family="AF_UNIX")
    slots = []
    for _ in range(2 * args.clients):
        conn = listener.accept()
        client_id, model, cap = conn.recv()
        slots.append(Slot(conn, client_id, model, cap, engines[model].version))
    print(f"server: {args.clients} clients connected, mode {args.mode}", flush=True)
    stats = {m: [0, 0] for m in engines}          # 前向次数、决策数
    t0 = time.time()

    def run(model, group):
        e = engines[model]
        with torch.autocast(dev.type, enabled=e.enable_amp), torch.inference_mode():
            obs = torch.cat([torch.from_numpy(s.obs[:s.pending]).to(dev, non_blocking=True) for s in group])
            masks = torch.cat([torch.from_numpy(s.masks[:s.pending]).to(dev, non_blocking=True) for s in group])
            q = e.dqn(e.brain(obs), masks).float().cpu().numpy()
        i = 0
        for s in group:
            s.q[:s.pending] = q[i:i + s.pending]
            i += s.pending
            s.pending = 0
            s.conn.send("ok")
        stats[model][0] += 1
        stats[model][1] += len(q)

    by_client = {}
    for s in slots:
        by_client.setdefault(s.client_id, []).append(s)

    def receive(timeout):
        listen = [s for s in slots if not s.done and s.pending == 0]
        for conn in wait([s.conn for s in listen], timeout):
            s = next(x for x in listen if x.conn is conn)
            n = conn.recv()
            if n == 0:
                s.done = True
            else:
                s.pending = n

    def run_pending():
        for model in engines:           # 固定的模型顺序、客户端编号顺序
            group = sorted((s for s in slots if s.model == model and s.pending), key=lambda s: s.client_id)
            if group:
                run(model, group)

    while not all(s.done for s in slots):
        live = [c for c, ss in by_client.items() if not all(s.done for s in ss)]
        if args.mode == "lockstep":
            # 竞技场是同步的:每个客户端同一时刻最多等一个请求。所有仍在跑的客户端都在等时,挂起的请求集合
            # 只由各自确定的历史决定,与时序无关,所以合出的批可复现;不必等同一模型(某些步不调某个模型,等齐会死锁)
            if all(any(s.pending for s in by_client[c]) for c in live):
                run_pending()
            else:
                receive(None)
        else:
            receive(0)                  # 先把已经到的都收下
            if any(s.pending for s in slots):
                run_pending()
            else:
                receive(None)
    wall = time.time() - t0
    for model, (calls, n) in stats.items():
        print(f"server {model}: {calls:,} forwards, {n:,} decisions, {n / max(calls, 1):.0f} per forward", flush=True)
    print(f"server wall {wall:.1f} s", flush=True)
    for s in slots:
        s.close()
    listener.close()


if __name__ == "__main__":
    main()
