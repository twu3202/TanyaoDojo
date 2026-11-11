"""
X1 推理架构基准(research/2026-11-03_rl_infra_survey):同一 1v3 评测负载,换引擎变体量吞吐与逐局一致性。

变体(挑战者与冠军两个引擎同时换):
  base   = run_eval 的 MortalEngine 原样
  timed  = base 的同一计算,加 cuda 同步计时:拼接 / 拷进 GPU / 前向 / 回传,看推理占墙钟多少(Amdahl)
  pinned = 预分配锁页缓冲,np.stack 直接写进去,non_blocking 拷进 GPU;批形状不变,应逐局一致
  graph  = 按批宽分桶(64 的倍数)补齐后回放 CUDA graph;批形状变了,可能翻转近似平局的决策
  graphx = 每个实际批宽一张 CUDA graph,不补齐;批形状不变,有望逐局一致
用法(与 run_eval 同一环境变量 MORTAL_DIR / MORTAL_V4):
  python x1_bench.py <challenger.pth> --variant timed --games 400 --iters 1 --seed-start 10000 --log-dir DIR
逐局一致性:各变体的 log-dir 去掉 eval_time_ns 后比 md5(见 patches/snapshot_ensemble 的做法)。
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import torch

import run_eval  # 复用模型加载与路径设置(MORTAL_DIR)
from libriichi.arena import OneVsThree

PTS = np.array([90.0, 45.0, 0.0, -135.0])


class Variant:
    """包一个 MortalEngine(版本 2-5、非 oracle、不开 Boltzmann),只换 react_batch 的实现。
    libriichi 只读 engine_type / name / is_oracle / version / enable_quick_eval / enable_rule_based_agari_guard。"""

    def __init__(self, e):
        assert e.version != 1 and not e.is_oracle and e.boltzmann_epsilon == 0
        self.e = e
        self.engine_type, self.name = "mortal", e.name
        self.is_oracle, self.version, self.device = e.is_oracle, e.version, e.device
        self.enable_quick_eval = e.enable_quick_eval
        self.enable_rule_based_agari_guard = e.enable_rule_based_agari_guard
        self.calls = self.decisions = 0

    def forward(self, obs_t, masks_t):
        return self.e.dqn(self.e.brain(obs_t), masks_t)

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with torch.autocast(self.device.type, enabled=self.e.enable_amp), torch.inference_mode():
                self.calls += 1
                self.decisions += len(obs)
                return self.react(obs, masks)
        except Exception as ex:
            import traceback
            raise Exception(f"{ex}\n{traceback.format_exc()}")

    def report(self) -> str:
        return f"{self.name}: {self.calls:,} calls, {self.decisions:,} decisions"


class Base(Variant):
    def react(self, obs, masks):
        obs_t = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        masks_t = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        q = self.forward(obs_t, masks_t)
        return q.argmax(-1).tolist(), q.tolist(), masks_t.tolist(), [True] * len(obs)


class Timed(Variant):
    PARTS = ("stack", "to_gpu", "forward", "back")

    def __init__(self, e):
        super().__init__(e)
        self.t = dict.fromkeys(self.PARTS, 0.0)

    def react(self, obs, masks):
        sync = torch.cuda.synchronize
        t0 = time.perf_counter()
        obs_np, masks_np = np.stack(obs, axis=0), np.stack(masks, axis=0)
        t1 = time.perf_counter()
        obs_t = torch.as_tensor(obs_np, device=self.device)
        masks_t = torch.as_tensor(masks_np, device=self.device)
        sync(); t2 = time.perf_counter()
        q = self.forward(obs_t, masks_t)
        actions = q.argmax(-1)
        sync(); t3 = time.perf_counter()
        out = actions.tolist(), q.tolist(), masks_t.tolist(), [True] * len(obs)
        t4 = time.perf_counter()
        for k, a, b in zip(self.PARTS, (t0, t1, t2, t3), (t1, t2, t3, t4)):
            self.t[k] += b - a
        return out

    def report(self) -> str:
        tot = sum(self.t.values())
        parts = ", ".join(f"{k} {v:.1f} s ({v / tot:.0%})" for k, v in self.t.items())
        return f"{super().report()}; in react {tot:.1f} s: {parts}"


class Pinned(Variant):
    def __init__(self, e):
        super().__init__(e)
        from libriichi.consts import ACTION_SPACE, obs_shape
        self.obs_shape, self.n_actions = tuple(obs_shape(self.version)), ACTION_SPACE
        self.cap = 0

    def ensure(self, n):
        """锁页缓冲至少 n 行;不够就按 2 倍扩。"""
        if n > self.cap:
            self.cap = max(n, 2 * self.cap, 1024)
            self.obs_h = torch.empty((self.cap, *self.obs_shape), dtype=torch.float32).pin_memory()
            self.masks_h = torch.empty((self.cap, self.n_actions), dtype=torch.bool).pin_memory()

    def react(self, obs, masks):
        n = len(obs)
        self.ensure(n)
        np.stack(obs, axis=0, out=self.obs_h[:n].numpy())
        np.stack(masks, axis=0, out=self.masks_h[:n].numpy())
        obs_t = self.obs_h[:n].to(self.device, non_blocking=True)
        masks_t = self.masks_h[:n].to(self.device, non_blocking=True)
        q = self.forward(obs_t, masks_t)
        q_h = q.to("cpu")                 # 一次拷回,再在 CPU 上转 list
        return q_h.argmax(-1).tolist(), q_h.tolist(), self.masks_h[:n].tolist(), [True] * n


class Graph(Pinned):
    """每个批宽桶(64 的倍数)一张 CUDA graph;补齐的行掩码全开、观测为 0,只取前 n 行。"""
    STEP = 64

    def __init__(self, e):
        super().__init__(e)
        self.graphs, self.pool = {}, None

    def graph_for(self, b):
        if b not in self.graphs:
            x = torch.zeros((b, *self.obs_shape), device=self.device)
            m = torch.ones((b, self.n_actions), dtype=torch.bool, device=self.device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.forward(x, m)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):    # 各桶共用一个显存池,一次只回放一张
                y = self.forward(x, m)
            if self.pool is None:
                self.pool = g.pool()
            self.graphs[b] = (g, x, m, y)
        return self.graphs[b]

    def react(self, obs, masks):
        n = len(obs)
        self.ensure(n)
        np.stack(obs, axis=0, out=self.obs_h[:n].numpy())
        np.stack(masks, axis=0, out=self.masks_h[:n].numpy())
        b = -(-n // self.STEP) * self.STEP
        g, x, m, y = self.graph_for(b)
        x[:n].copy_(self.obs_h[:n], non_blocking=True)
        m[:n].copy_(self.masks_h[:n], non_blocking=True)
        if n < b:
            x[n:].zero_()
            m[n:].fill_(True)
        g.replay()
        q_h = y[:n].to("cpu")
        return q_h.argmax(-1).tolist(), q_h.tolist(), self.masks_h[:n].tolist(), [True] * n


class GraphExact(Pinned):
    """每个实际批宽一张 CUDA graph,不补齐:批形状与 base 相同,算子选择应不变。
    所有 graph 读同一块静态输入(取前 b 行)、共用一个显存池,所以多录几百张也只多占输出那点显存。"""

    def __init__(self, e):
        super().__init__(e)
        self.graphs, self.pool, self.x = {}, None, None

    def graph_for(self, b):
        if self.x is None or b > self.x.shape[0]:
            rows = max(b, 1024)
            self.x = torch.zeros((rows, *self.obs_shape), device=self.device)
            self.m = torch.ones((rows, self.n_actions), dtype=torch.bool, device=self.device)
            self.graphs, self.pool = {}, None      # 旧 graph 绑在旧缓冲上,作废
        if b not in self.graphs:
            x, m = self.x[:b], self.m[:b]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.forward(x, m)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                y = self.forward(x, m)
            if self.pool is None:
                self.pool = g.pool()
            self.graphs[b] = (g, y)
        return self.graphs[b]

    def react(self, obs, masks):
        n = len(obs)
        self.ensure(n)
        np.stack(obs, axis=0, out=self.obs_h[:n].numpy())
        np.stack(masks, axis=0, out=self.masks_h[:n].numpy())
        g, y = self.graph_for(n)
        self.x[:n].copy_(self.obs_h[:n], non_blocking=True)
        self.m[:n].copy_(self.masks_h[:n], non_blocking=True)
        g.replay()
        q_h = y.to("cpu")
        return q_h.argmax(-1).tolist(), q_h.tolist(), self.masks_h[:n].tolist(), [True] * n

    def report(self) -> str:
        return f"{super().report()}, {len(self.graphs)} graphs"


class Client(Variant):
    """变体 D 的客户端:模型不在本进程(只在 CPU 上加载以取属性),观测写进 x1_server 建的共享内存,发 n 等 "ok"。"""

    def __init__(self, e, role, client_id, address, cap=1024):
        from multiprocessing import shared_memory
        from multiprocessing.connection import Client as Conn
        from libriichi.consts import ACTION_SPACE, obs_shape
        super().__init__(e)
        self.conn = Conn(address, family="AF_UNIX")
        self.conn.send((client_id, role, cap))
        self.shms = [shared_memory.SharedMemory(name=n) for n in self.conn.recv()]
        from multiprocessing import resource_tracker
        for s in self.shms:     # 段归服务端所有;否则本进程退出时 resource_tracker 会替它 unlink
            resource_tracker.unregister(s._name, "shared_memory")
        self.obs = np.ndarray((cap, *obs_shape(self.version)), np.float32, self.shms[0].buf)
        self.masks = np.ndarray((cap, ACTION_SPACE), np.bool_, self.shms[1].buf)
        self.q = np.ndarray((cap, ACTION_SPACE), np.float32, self.shms[2].buf)

    def react(self, obs, masks):
        n = len(obs)
        np.stack(obs, axis=0, out=self.obs[:n])
        np.stack(masks, axis=0, out=self.masks[:n])
        self.conn.send(n)
        assert self.conn.recv() == "ok"
        q = torch.from_numpy(self.q[:n].copy())     # 服务端已把非法动作置为 -inf
        return q.argmax(-1).tolist(), q.tolist(), self.masks[:n].tolist(), [True] * n

    def finish(self):
        self.conn.send(0)
        for s in self.shms:
            s.close()


VARIANTS = {"base": Base, "timed": Timed, "pinned": Pinned, "graph": Graph, "graphx": GraphExact, "client": Client}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=10000)
    ap.add_argument("--seed-key", type=int, default=20260711)
    ap.add_argument("--champion", default=run_eval.os.environ.get(
        "MORTAL_V4", "/home/r/Projects/better_mortal/baseline/mortal_v4.pth"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", default="off", choices=("on", "off"), help="off = 评测口径 GPU fp32;on = 训练 worker 口径")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--challenger-name", default="challenger", help="写进牌谱的名字;q25_final 可与 bench_eval 的牌谱逐字节对照")
    ap.add_argument("--client-id", type=int, default=0, help="variant client:本进程在服务端的编号")
    ap.add_argument("--socket", default="/tmp/x1.sock", help="variant client:x1_server 的 socket")
    args = ap.parse_args()

    V = VARIANTS[args.variant]
    if args.variant == "client":
        # 模型只在 CPU 上加载取属性;推理在服务端。名字与 run_eval 相同,牌谱可比
        cham = Client(run_eval.load_champion(args.champion, "cpu", args.amp), "champion", args.client_id, args.socket)
        e = run_eval.load_champion(args.params, "cpu", args.amp)
        e.name = args.challenger_name
        chal = Client(e, "challenger", args.client_id, args.socket)
    else:
        cham = V(run_eval.load_champion(args.champion, args.device, args.amp))
        e = run_eval.load_champion(args.params, args.device, args.amp)
        e.name = args.challenger_name
        chal = V(e)
    total = np.zeros(4, np.int64)
    t0 = time.time()
    for i in range(args.iters):
        seed = args.seed_start + i * (args.games // 4)
        env = OneVsThree(disable_progress_bar=True, log_dir=args.log_dir)
        total += np.array(env.py_vs_py(challenger=chal, champion=cham,
                                       seed_start=(seed, args.seed_key), seed_count=args.games // 4))
    wall = time.time() - t0
    n = int(total.sum())
    mem = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"variant {args.variant} amp {args.amp}: {n} games in {wall:.1f} s = {n / wall:.3f} games/s; "
          f"avg_pt {total @ PTS / n:+.3f}; peak reserved {mem:.2f} GiB", flush=True)
    for x in (chal, cham):
        print("  " + x.report(), flush=True)
        if isinstance(x, Client):
            x.finish()
    if isinstance(chal, Timed):
        r = sum(chal.t.values()) + sum(cham.t.values())
        print(f"  inference share of wall: {r / wall:.1%} ({r:.1f} s of {wall:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
