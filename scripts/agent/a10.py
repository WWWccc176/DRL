#!/usr/bin/env python3
"""
a10.py — Dimension-agnostic Dueling-DDQN + 独立 G6K-GPU sieve 服务。

进程/GPU 布局（本机检测到 4 张 48G 卡）：
  main (learner)   : 物理 GPU0 —— CNN/DDQN 推理 + PER 训练（torch, cuda:0）
  sieve worker(s)  : 4 卡均摊 —— 常驻 g6k gpu_sieve 服务（每进程一次 CUDA context）
  env workers      : CPU      —— LLL / LOCAL_BKZ / ENUM / 状态构建（禁 CUDA）

调度理念：网络极小(2-6GB, 毫秒级)，sieve 才是吞吐瓶颈。故 4 张卡全部跑
  sieve 以最大化 env-steps/s；learner 寄生 GPU0（占用可忽略，自动限速到经验产出）。

动作执行策略（同 g6k_oracle.py）：
  beta <  ENUM_MIN_BETA            -> LOCAL_BKZ           (CPU, env 内)
  ENUM_MIN <= beta < SIEVE_MIN     -> ORACLE_ENUM_BLOCK   (CPU, env 内)
  beta >= SIEVE_MIN_BETA(=40)      -> dump_block -> sieve 队列 -> GPU g6k
                                      -> insert_coeff_vector；失败回退 ENUM

sieve 协议：env 只传 (env_id, seq, 块的整数行向量)，绝不跨进程传 pool_id。
"""

# ============================================================
# 角色守卫：必须在 import torch / my_project_backend 之前执行。
# spawn 子进程会重新 import 本文件，此段保证：
#   env   worker: 完全看不到 CUDA
#   sieve worker: 只看到分配给它的那张物理 GPU（映射为它的 cuda:0），backend 走 CPU
# ============================================================
import os

_ROLE = os.environ.get("A10_ROLE", "main")
if _ROLE == "env":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LATTICE_DISABLE_CUDA"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
elif _ROLE == "sieve":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("A10_SIEVE_GPU", "0")
    os.environ["LATTICE_DISABLE_CUDA"] = "1"  # backend 走 CPU；GPU 留给 g6k
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
else:  # main / learner
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

import re
import sys
import glob
import math
import time
import random
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp

import my_project_backend

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(4 if _ROLE == "main" else 1)

# ============================================================
# Config
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
if _ROLE == "main":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

FEAT_C = 64
GS_EMB = 16
CTX_DIM = 64
ACT_EMB = 5
GS_LOC = 3
DILATIONS = [1, 2, 4, 8]
BETA_REF = 60.0
DIM_REF = 60.0
NUM_GLOBALS = 7
LLL_FREQ = 3

# --- 路径（显式写死，允许环境变量覆盖）---
PROJECT_ROOT = os.environ.get("DRL_ROOT", "/home/amax/projects/DRL")
G6K_ROOT = os.environ.get("G6K_ROOT", "/home/amax/workspace/builds/g6k")
# g6k_env.py 实际所在目录：优先 A10_G6K_HELPERS；否则在 main 里自动搜索 PROJECT_ROOT
G6K_HELPER_DIR = os.environ.get("A10_G6K_HELPERS", "")

# --- sieve 分工阈值 ---
ENUM_MIN_BETA = 30  # 30~39 用 ORACLE_ENUM_BLOCK
SIEVE_MIN_BETA = 40  # >=40 送 GPU sieve（block dim 40 起 sieve 才划算）
SIEVE_TIMEOUT_S = 180.0

# --- 4 卡调度：learner 在 GPU0，sieve 8 个 context 均摊 4 卡（每卡 2 个）---
NUM_SIEVE_WORKERS = 8
SIEVE_GPUS = [
    "0",
    "1",
    "2",
    "3",
    "0",
    "1",
    "2",
    "3",
]  # worker i -> 物理 GPU SIEVE_GPUS[i]
SIEVE_THREADS = 2  # 每个 g6k GPU sieve 的 host 侧 CPU 线程数


def _status(msg):
    sys.stdout.write("\r\033[K" + msg)
    sys.stdout.flush()


def _log(msg):
    sys.stdout.write("\r\033[K" + msg + "\n")
    sys.stdout.flush()


# ============================================================
# I/O helpers
# ============================================================
def matrix_to_string(basis):
    lines = [" ".join(str(x) for x in row) for row in basis]
    return "[" + "\n".join(f"[{l}]" for l in lines) + "]"


def parse_fplll(s):
    """容错解析 fplll 括号格式（dump_matrix / dump_block 通用）。"""
    out = []
    for line in s.strip().splitlines():
        line = line.strip().lstrip("[").rstrip("]").strip()
        if line:
            out.append([int(x) for x in line.split()])
    return out


def parse_challenge_file(filepath):
    matrix = []
    with open(filepath, "r") as f:
        content = f.read().replace("[", "").replace("]", "")
        for line in content.strip().split("\n"):
            if line.strip():
                matrix.append([int(x) for x in line.split()])
    return matrix


def parse_dim_seed(path):
    b = os.path.basename(path)
    d = re.search(r"dim(\d+)", b)
    s = re.search(r"seed(\d+)", b)
    return (int(d.group(1)) if d else 0, int(s.group(1)) if s else 0)


def discover_helper_dir(explicit, project_root):
    """确定 g6k_env.py 真实目录：优先显式；否则在 project_root 下搜索。不猜测。"""
    if explicit:
        if os.path.isfile(os.path.join(explicit, "g6k_env.py")):
            return explicit
        print(
            f"[warn] A10_G6K_HELPERS={explicit} 下未找到 g6k_env.py，转为自动搜索",
            flush=True,
        )
    for root, _dirs, fnames in os.walk(project_root):
        if "g6k_env.py" in fnames:
            return root
    return ""


# ============================================================
# 动作空间
# ============================================================
def build_action_list(dim):
    beta_max = min(int(0.8 * dim), 64)  # dim>=50 时最大 beta 可达 40+，会触发 sieve 档
    beta_min = max(8, int(0.15 * dim))
    raw = np.geomspace(beta_min, max(beta_min + 1, beta_max), 7)
    betas = sorted(set(max(2, int(round(x))) for x in raw))
    action_list = []
    for b in betas:
        if b > dim:
            continue
        pos_step = max(1, b // 2)
        for p in range(0, dim - b + 1, pos_step):
            action_list.append((b, p))
        if dim - b >= 0 and (b, 0) not in action_list:
            action_list.insert(0, (b, 0))
    return action_list


def build_action_spec(dim, device):
    al = build_action_list(dim)
    betas = np.array([b for b, _ in al], dtype=np.int64)
    poss = np.array([p for _, p in al], dtype=np.int64)
    r0, r1 = poss, poss + betas
    end_idx = np.clip(poss + betas - 1, 0, dim - 1)
    area = (betas.astype(np.float32)) ** 2
    emb = np.stack(
        [
            betas / dim,
            poss / dim,
            (poss + betas) / dim,
            betas / BETA_REF,
            (poss + betas / 2.0) / dim,
        ],
        axis=1,
    ).astype(np.float32)

    t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)

    groups = {}
    for k, (b, p) in enumerate(al):
        groups.setdefault(b, ([], []))
        groups[b][0].append(k)
        groups[b][1].append(p)
    beta_groups = {
        int(b): (t(idx, torch.long), t(pos, torch.long))
        for b, (idx, pos) in groups.items()
    }

    return {
        "action_list": al,
        "num_actions": len(al),
        "r0": t(r0, torch.long),
        "r1": t(r1, torch.long),
        "c0": t(r0, torch.long),
        "c1": t(r1, torch.long),
        "pos": t(poss, torch.long),
        "end_idx": t(end_idx, torch.long),
        "area": t(area, torch.float32),
        "emb": t(emb, torch.float32),
        "beta_groups": beta_groups,
    }


# ============================================================
# NoisyNet
# ============================================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=1.0):
        super().__init__()
        self.in_features, self.out_features, self.std_init = (
            in_features,
            out_features,
            std_init,
        )
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))

    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self.scale_noise(self.in_features)
        eps_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        return F.linear(x, self.weight_mu, self.bias_mu)


# ============================================================
# 网络组件
# ============================================================
class GSEncoder(nn.Module):
    def __init__(self, out_ch=GS_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, out_ch, 5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.LeakyReLU(0.01),
        )

    def forward(self, gs):
        return self.net(gs.unsqueeze(1))


class FiLMAxialBlock(nn.Module):
    def __init__(self, c, ctx_dim, dilation):
        super().__init__()
        self.col = nn.Conv2d(
            c, c, (3, 1), padding=(dilation, 0), dilation=(dilation, 1)
        )
        self.row = nn.Conv2d(
            c, c, (1, 3), padding=(0, dilation), dilation=(1, dilation)
        )
        self.pw = nn.Conv2d(2 * c, c, 1)
        self.film = nn.Linear(ctx_dim, 2 * c)

    def forward(self, x, ctx):
        h = self.pw(torch.cat([self.col(x), self.row(x)], dim=1))
        gamma, beta = self.film(ctx).chunk(2, dim=1)
        h = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(
            -1
        ).unsqueeze(-1)
        return x + F.leaky_relu(h, 0.01)


def _integral_region_mean(H, r0, r1, c0, c1, area):
    """积分图区域均值。H:[B,C,D,D] -> [B,C,A]。"""
    I = F.pad(H, (1, 0, 1, 0)).cumsum(2).cumsum(3)
    s = I[:, :, r1, c1] - I[:, :, r0, c1] - I[:, :, r1, c0] + I[:, :, r0, c0]
    return s / area.view(1, 1, -1)


def _region_max(H, spec):
    """区域极值池化：每个唯一 beta 做一次 max_pool2d(kernel=beta, stride=1)，
    再取对角位置 (p,p)。H:[B,C,D,D] -> [B,C,A]。"""
    B, C, D, _ = H.shape
    out = H.new_empty(B, C, spec["num_actions"])
    for b, (idx, pos) in spec["beta_groups"].items():
        k = min(b, D)
        pooled = F.max_pool2d(H, kernel_size=k, stride=1)
        out[:, :, idx] = pooled[:, :, pos, pos]
    return out


class DimAgnosticQNet(nn.Module):
    def __init__(
        self, num_globals=NUM_GLOBALS, c=FEAT_C, gs_emb=GS_EMB, ctx_dim=CTX_DIM
    ):
        super().__init__()
        self.c = c
        self.gs_encoder = GSEncoder(gs_emb)
        in_ch = 1 + 3 * gs_emb + 2
        self.stem = nn.Conv2d(in_ch, c, 1)
        self.global_mlp = nn.Sequential(
            nn.Linear(num_globals, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, ctx_dim),
            nn.LeakyReLU(0.01),
        )
        self.blocks = nn.ModuleList([FiLMAxialBlock(c, ctx_dim, d) for d in DILATIONS])

        self.value_mlp = nn.Sequential(
            NoisyLinear(2 * c + ctx_dim, 128), nn.LeakyReLU(0.01), NoisyLinear(128, 1)
        )
        act_feat = 2 * c + ACT_EMB + GS_LOC + ctx_dim
        self.action_mlp = nn.Sequential(
            NoisyLinear(act_feat, 128), nn.LeakyReLU(0.01), NoisyLinear(128, 1)
        )

    def _build_input(self, cos, gs_emb):
        B, D, _ = cos.shape
        Cg = gs_emb.shape[1]
        gi = gs_emb.unsqueeze(3).expand(B, Cg, D, D)
        gj = gs_emb.unsqueeze(2).expand(B, Cg, D, D)
        idx = torch.linspace(0.0, 1.0, D, device=cos.device)
        pi = idx.view(1, 1, D, 1).expand(B, 1, D, D)
        pj = idx.view(1, 1, 1, D).expand(B, 1, D, D)
        return torch.cat([cos.unsqueeze(1), gi, gj, gi - gj, pi, pj], dim=1)

    def forward(self, cos, gs, glob, spec):
        gs_emb = self.gs_encoder(gs)
        x = self.stem(self._build_input(cos, gs_emb))
        ctx = self.global_mlp(glob)
        for blk in self.blocks:
            x = blk(x, ctx)

        value = self.value_mlp(
            torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3)), ctx], dim=1)
        )

        rmean = _integral_region_mean(
            x, spec["r0"], spec["r1"], spec["c0"], spec["c1"], spec["area"]
        ).permute(0, 2, 1)
        rmax = _region_max(x, spec).permute(0, 2, 1)
        B, A, _ = rmean.shape
        aemb = spec["emb"].unsqueeze(0).expand(B, A, -1)
        gs_start = gs.gather(1, spec["pos"].unsqueeze(0).expand(B, A))
        gs_end = gs.gather(1, spec["end_idx"].unsqueeze(0).expand(B, A))
        gs_loc = torch.stack([gs_start, gs_end, gs_start - gs_end], dim=2)
        ctx_b = ctx.unsqueeze(1).expand(B, A, -1)
        adv = self.action_mlp(
            torch.cat([rmean, rmax, aemb, gs_loc, ctx_b], dim=2)
        ).squeeze(-1)

        return value + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ============================================================
# 分层 PER
# ============================================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        return (
            self._retrieve(left, s)
            if s <= self.tree[left]
            else self._retrieve(left + 1, s - self.tree[left])
        )

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


class PERBuffer:
    PER_e, PER_a, PER_b, PER_b_inc = 1e-5, 0.6, 0.4, 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _prio(self, err):
        return (np.abs(err) + self.PER_e) ** self.PER_a

    def add(self, err, sample):
        self.tree.add(self._prio(err), sample)

    def sample(self, n):
        batch, idxs, prios = [], [], []
        total = self.tree.total()
        seg = total / n
        self.PER_b = min(1.0, self.PER_b + self.PER_b_inc)
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                idx, p, data = self.tree.get(random.uniform(0, total))
            prios.append(p)
            idxs.append(idx)
            batch.append(data)
        probs = np.array(prios) / (self.tree.total() + 1e-10)
        w = (self.tree.n_entries * probs + 1e-10) ** (-self.PER_b)
        w /= w.max()
        return batch, idxs, torch.as_tensor(w, dtype=torch.float32)

    def update(self, idx, err):
        self.tree.update(idx, self._prio(err))

    def __len__(self):
        return self.tree.n_entries


class MultiDimReplay:
    def __init__(self, capacity_per_dim=12000):
        self.cap = capacity_per_dim
        self.buffers = {}

    def _buf(self, dim):
        if dim not in self.buffers:
            self.buffers[dim] = PERBuffer(self.cap)
        return self.buffers[dim]

    def add(self, dim, sample):
        self._buf(dim).add(1.0, sample)

    def ready_dims(self, min_size):
        return [d for d, b in self.buffers.items() if len(b) >= min_size]


# ============================================================
# Agent
# ============================================================
class DQNAgent:
    def __init__(
        self,
        num_globals=NUM_GLOBALS,
        batch_size=128,
        dims_per_update=3,
        capacity_per_dim=12000,
    ):
        self.device = DEVICE
        self.batch_size = batch_size
        self.dims_per_update = dims_per_update
        self.gamma = 0.99
        self.tau = 0.0025

        self.q_net = DimAgnosticQNet(num_globals).to(self.device)
        self.target_net = DimAgnosticQNet(num_globals).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.q_net.parameters(), lr=6e-5, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2000, eta_min=1e-6
        )
        self.memory = MultiDimReplay(capacity_per_dim)
        self._spec_cache = {}

    def spec(self, dim):
        if dim not in self._spec_cache:
            self._spec_cache[dim] = build_action_spec(dim, self.device)
        return self._spec_cache[dim]

    def act_envs(self, state_by_eid, epsilon):
        groups = defaultdict(list)
        for eid, st in state_by_eid.items():
            groups[st["dim"]].append(eid)
        out = {}
        self.q_net.train()
        self.q_net.reset_noise()
        for dim, eids in groups.items():
            spec = self.spec(dim)
            cos = torch.as_tensor(
                np.stack([state_by_eid[e]["cos"] for e in eids]),
                dtype=torch.float32,
                device=self.device,
            )
            gs = torch.as_tensor(
                np.stack([state_by_eid[e]["gs"] for e in eids]),
                dtype=torch.float32,
                device=self.device,
            )
            glob = torch.as_tensor(
                np.stack([state_by_eid[e]["globals"] for e in eids]),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                greedy = self.q_net(cos, gs, glob, spec).argmax(1).cpu().numpy()
            A = spec["num_actions"]
            for k, e in enumerate(eids):
                out[e] = (
                    random.randint(0, A - 1)
                    if (epsilon > 0 and random.random() < epsilon)
                    else int(greedy[k])
                )
        return out

    def remember(self, dim, s, a, r, ns, done):
        self.memory.add(
            dim,
            (
                s["cos"].astype(np.float16),
                s["gs"].astype(np.float16),
                s["globals"].astype(np.float32),
                int(a),
                float(r),
                ns["cos"].astype(np.float16),
                ns["gs"].astype(np.float16),
                ns["globals"].astype(np.float32),
                float(done),
            ),
        )

    def _dim_loss(self, dim):
        buf = self.memory.buffers[dim]
        batch, idxs, isw = buf.sample(self.batch_size)
        spec = self.spec(dim)
        f32 = lambda arrs: torch.as_tensor(
            np.stack(arrs).astype(np.float32), dtype=torch.float32, device=self.device
        )
        cos = f32([b[0] for b in batch])
        gs = f32([b[1] for b in batch])
        glob = f32([b[2] for b in batch])
        ncos = f32([b[5] for b in batch])
        ngs = f32([b[6] for b in batch])
        nglob = f32([b[7] for b in batch])
        a = torch.as_tensor(
            [b[3] for b in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        r = torch.as_tensor(
            [b[4] for b in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        d = torch.as_tensor(
            [b[8] for b in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        isw = isw.to(self.device).unsqueeze(1)

        with torch.no_grad():
            na = self.q_net(ncos, ngs, nglob, spec).argmax(1, keepdim=True)
            nq = self.target_net(ncos, ngs, nglob, spec).gather(1, na)
            target = r + (1.0 - d) * self.gamma * nq

        curr = self.q_net(cos, gs, glob, spec).gather(1, a)
        td = (curr - target).detach().abs().cpu().numpy().flatten()
        for i, idx in enumerate(idxs):
            buf.update(idx, td[i])
        return (isw * F.smooth_l1_loss(curr, target, reduction="none")).mean()

    def learn(self):
        ready = self.memory.ready_dims(self.batch_size)
        if not ready:
            return 0.0
        chosen = random.sample(ready, min(self.dims_per_update, len(ready)))
        self.q_net.train()
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        self.optimizer.zero_grad()
        total = 0.0
        for dim in chosen:
            loss = self._dim_loss(dim)
            (loss / len(chosen)).backward()
            total += float(loss.item())
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.75)
        self.optimizer.step()
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return total / len(chosen)

    def step_scheduler(self):
        self.scheduler.step()

    def save(self, path, extra=None):
        extra = dict(extra or {})
        extra["rng"] = {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        tmp = path + ".tmp"
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "extra": extra,
            },
            tmp,
        )
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return {}
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(payload["q_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        extra = payload.get("extra", {})
        rng = extra.pop("rng", None)
        if rng:
            try:
                random.setstate(rng["py"])
                np.random.set_state(rng["np"])
                torch.set_rng_state(rng["torch"])
            except Exception:
                pass
        return extra


# ============================================================
# G6K sieve 服务进程（一张物理 GPU，一次 CUDA context，逐请求复用）
# ============================================================
def sieve_worker(req_q, resp_conns, helper_dir, worker_id):
    import faulthandler

    faulthandler.enable()

    # chdir 只改工作目录，不可靠替代模块搜索路径 -> G6K_ROOT 与 helper_dir 都进 sys.path
    for pth in (G6K_ROOT, helper_dir):
        if pth and pth not in sys.path:
            sys.path.insert(0, pth)

    try:
        import g6k_env  # noqa: F401  MUST be first: chdir 到 g6k_env 期望的工作目录
    except Exception as e:
        print(f"[sieve{worker_id}] g6k_env import failed: {e}", flush=True)

    g6k_ok = True
    IntegerMatrix = Siever = SieverParams = None
    try:
        from fpylll import IntegerMatrix
        from g6k import Siever, SieverParams
        import g6k

        print(
            f"[sieve{worker_id}] g6k={getattr(g6k, '__file__', '?')} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
    except Exception as e:
        g6k_ok = False
        print(
            f"[sieve{worker_id}] g6k unavailable: {e}; all requests -> None", flush=True
        )

    if not g6k_ok:
        while True:
            item = req_q.get()
            if item is None:
                return
            env_id, seq, rows = item
            try:
                resp_conns[env_id].send((seq, None))
            except Exception:
                pass
        return

    print(f"[sieve{worker_id}] ready", flush=True)
    n_done, t_busy = 0, 0.0
    while True:
        item = req_q.get()
        if item is None:
            break
        env_id, seq, rows = item
        t0 = time.time()
        coeffs = None
        try:
            A = IntegerMatrix.from_matrix([[int(x) for x in r] for r in rows])
            try:
                params = SieverParams(
                    threads=SIEVE_THREADS,
                    gpus=1,
                    gpu_bucketer=b"bdgl",
                    gpu_triple=False,
                )
            except Exception:
                try:
                    params = SieverParams(threads=SIEVE_THREADS)
                except Exception:
                    params = None
            g = Siever(A, params) if params is not None else Siever(A)
            g.initialize_local(0, 0, A.nrows)
            if hasattr(g, "gpu_sieve"):
                g.gpu_sieve()
            else:
                g(alg="bgj1")
            lifts = g.best_lifts()
            if lifts:
                best = min(lifts, key=lambda t: t[1])
                if best[2] is not None and len(best[2]) > 0:
                    coeffs = [int(x) for x in best[2]]
        except Exception as e:
            print(
                f"[sieve{worker_id}] fail (env{env_id}, beta={len(rows)}): {e}",
                flush=True,
            )
            coeffs = None

        try:
            resp_conns[env_id].send((seq, coeffs))
        except Exception:
            pass
        n_done += 1
        t_busy += time.time() - t0
        if n_done % 50 == 0:
            print(
                f"[sieve{worker_id}] served {n_done}, avg {t_busy / n_done:.2f}s/req",
                flush=True,
            )


def _g6k_probe(conn, helper_dir):
    """启动前自检子进程：sieve 角色下导入 g6k，报告 __file__ / GPU 能力。"""
    info = {"ok": False, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        for pth in (G6K_ROOT, helper_dir):
            if pth and pth not in sys.path:
                sys.path.insert(0, pth)
        try:
            import g6k_env  # noqa
        except Exception as e:
            info["g6k_env_warn"] = repr(e)
        import g6k
        from g6k import Siever, SieverParams

        info["g6k_file"] = getattr(g6k, "__file__", "?")
        try:
            SieverParams(threads=1, gpus=1, gpu_bucketer=b"bdgl", gpu_triple=False)
            info["gpu_params_ok"] = True
        except Exception as e:
            info["gpu_params_ok"] = False
            info["gpu_params_warn"] = repr(e)
        info["has_gpu_sieve"] = hasattr(Siever, "gpu_sieve")
        info["ok"] = True
    except Exception as e:
        info["error"] = repr(e)
    try:
        conn.send(info)
    except Exception:
        pass
    conn.close()


def validate_g6k(helper_dir, gpu, timeout=180):
    """在启动 62 个 env 之前单独验证 g6k 可用性（含 GPU 能力）。"""
    parent_conn, child_conn = mp.Pipe()
    os.environ["A10_ROLE"] = "sieve"
    os.environ["A10_SIEVE_GPU"] = str(gpu)
    p = mp.Process(target=_g6k_probe, args=(child_conn, helper_dir), daemon=True)
    p.start()
    os.environ["A10_ROLE"] = "main"
    os.environ.pop("A10_SIEVE_GPU", None)
    child_conn.close()
    info = None
    if parent_conn.poll(timeout):
        try:
            info = parent_conn.recv()
        except Exception:
            info = None
    p.join(timeout=10)
    if p.is_alive():
        p.terminate()
    return info


class SieveClient:
    """env worker 侧同步客户端：dump_block -> 请求 -> 按 seq 对齐等结果。
    seq 对齐：超时返回 None 后，那次请求的迟到响应会被下次调用识别并丢弃，
    避免把上一个 block 的系数错插到当前 block（会静默破坏矩阵）。"""

    def __init__(self, env_id, req_q, resp_conn):
        self.env_id = env_id
        self.req_q = req_q
        self.resp_conn = resp_conn
        self.seq = 0

    def sieve_block(self, rows):
        self.seq += 1
        my_seq = self.seq
        self.req_q.put((self.env_id, my_seq, rows))
        deadline = time.time() + SIEVE_TIMEOUT_S
        while True:
            remaining = deadline - time.time()
            if remaining <= 0 or not self.resp_conn.poll(remaining):
                return None  # 超时 -> 上层回退 ENUM
            seq, coeffs = self.resp_conn.recv()
            if seq == my_seq:
                return coeffs
            # 否则是此前超时请求的迟到响应，丢弃继续等


# ============================================================
# 环境（CPU-only 进程内运行）
# ============================================================
class LatticeEnv:
    def __init__(self, filepath, env_id=0, sieve_client=None):
        self.filepath = filepath
        self.env_id = env_id
        self.sieve = sieve_client
        self.dim, self.seed_id = parse_dim_seed(filepath)
        self.lll_frequency = LLL_FREQ

        self.action_list = build_action_list(self.dim)
        self.num_actions = len(self.action_list)
        self.max_steps = max(4 * self.dim, 4 * self.num_actions)

        self.ratio_w, self.alpha, self.gamma_r, self.cost_w = 30.0, 2.0, 1.0, 0.15
        self.repeat_window, self.repeat_penalty_base = 8, 0.3

        self._preload()

        self.best_ratio = float("inf")
        self.best_defect = self.best_max_cos = self.best_min_cos = None
        self.best_vector = self.best_basis = None
        self.best_episode = 0
        self.episode_count = 0

    def _preload(self):
        mat = parse_challenge_file(self.filepath)
        self.initial_pool_id = my_project_backend.create_matrix_lll(
            matrix_to_string(mat)
        )
        ev = my_project_backend.evaluate_matrix(self.initial_pool_id)
        gs = np.array(ev["gs_log_norms"], dtype=np.float32)
        self.log_vol = float(np.sum(gs))
        self.log_GH = self.log_vol / self.dim + 0.5 * math.log(
            self.dim / (2 * math.pi * math.e)
        )
        info = my_project_backend.reduce(self.initial_pool_id, "LLL", 2, 0)
        self.defect_scale = max(abs(float(info["log_prod"] - self.log_vol)), 1.0)
        self.ratio_scale = max(abs(float(gs[0] - self.log_GH)), 1.0)

    def reset(self):
        self.current_step = 0
        self.action_history = []
        self.episode_count += 1
        if getattr(self, "current_pool_id", -1) >= 0:
            try:
                my_project_backend.free_matrix(self.current_pool_id)
            except Exception:
                pass
        self.current_pool_id = my_project_backend.clone_matrix(self.initial_pool_id)
        if self.current_pool_id < 0:
            raise RuntimeError(f"clone_matrix failed: {self.filepath}")

        info = my_project_backend.reduce(self.current_pool_id, "LLL", 2, 0)
        state, log_b1, ratio, mx, mn, ld = self._build_state(info)
        self._c_logb1, self._c_maxcos, self._c_logdef = log_b1, mx, ld
        self.initial_ep_ratio = self.current_ep_best_ratio = ratio
        self.last_info = info
        return state

    def _build_state(self, info):
        C = np.array(info["cos_matrix"], dtype=np.float32)
        lower = C[np.tril_indices(self.dim, -1)]
        max_cos = float(np.clip(np.max(lower) if lower.size else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size else 0.0, 0.0, 1.0))
        C = C + C.T

        ev = my_project_backend.evaluate_matrix(self.current_pool_id)
        gs = np.array(ev["gs_log_norms"], dtype=np.float32)
        log_b1 = float(gs[0])
        log_def = float(info["log_prod"] - self.log_vol)
        log_ratio = float(log_b1 - self.log_GH)

        gs_norm = np.tanh((gs - self.log_GH) / self.ratio_scale).astype(np.float32)
        globals_vec = np.array(
            [
                max_cos,
                float(np.tanh(log_def / self.defect_scale)),
                float(np.tanh(log_ratio / self.ratio_scale)),
                float(np.tanh((gs[0] - gs[-1]) / self.ratio_scale)),
                float(self.current_step / self.max_steps),
                float(math.log(self.dim) - math.log(DIM_REF)),
                float((self.current_step % self.lll_frequency) / self.lll_frequency),
            ],
            dtype=np.float32,
        )
        state = {"cos": C, "gs": gs_norm, "globals": globals_vec, "dim": self.dim}

        true_ratio = float(math.exp(log_ratio))
        self._maybe_update_best(true_ratio, log_def, max_cos, min_cos)
        return state, log_b1, true_ratio, max_cos, min_cos, log_def

    def _maybe_update_best(self, ratio, log_def, mx, mn):
        if ratio < self.best_ratio:
            self.best_ratio = ratio
            self.best_defect, self.best_max_cos, self.best_min_cos = log_def, mx, mn
            self.best_episode = self.episode_count
            mat = parse_fplll(my_project_backend.dump_matrix(self.current_pool_id))
            if mat:
                self.best_vector, self.best_basis = mat[0], mat

    # ---- 三档动作执行：LOCAL_BKZ / ENUM / GPU sieve（回退 ENUM）----
    def _exec_action(self, beta, pos):
        mid = self.current_pool_id
        if beta < ENUM_MIN_BETA:
            return my_project_backend.reduce(mid, "LOCAL_BKZ", beta, pos)
        if beta < SIEVE_MIN_BETA or self.sieve is None:
            try:
                return my_project_backend.reduce(mid, "ORACLE_ENUM_BLOCK", beta, pos)
            except Exception:
                return my_project_backend.reduce(mid, "LOCAL_BKZ", beta, pos)

        # GPU sieve：dump 原始块行（不做 LLL —— coeffs 必须相对原始行）
        rows = parse_fplll(my_project_backend.dump_block(mid, pos, beta))
        coeffs = self.sieve.sieve_block(rows)
        if coeffs:
            try:
                my_project_backend.insert_coeff_vector(
                    mid, pos, beta, [str(int(c)) for c in coeffs]
                )
            except Exception:
                pass
        else:  # sieve 失败/超时 -> ENUM，绝不静默跳过
            try:
                return my_project_backend.reduce(mid, "ORACLE_ENUM_BLOCK", beta, pos)
            except Exception:
                return my_project_backend.reduce(mid, "LOCAL_BKZ", beta, pos)
        # 插入后必须 LLL 恢复 GS，顺便拿 info
        return my_project_backend.reduce(mid, "LLL", 2, 0)

    def step(self, action_idx):
        beta, pos = self.action_list[action_idx]
        old_logb1, old_maxcos, old_logdef = (
            self._c_logb1,
            self._c_maxcos,
            self._c_logdef,
        )

        act_info = self._exec_action(beta, pos)
        do_lll = (
            self.current_step % self.lll_frequency == self.lll_frequency - 1
            or self.current_step >= self.max_steps - 1
        )
        self.last_info = (
            my_project_backend.reduce(self.current_pool_id, "LLL", 2, 0)
            if do_lll
            else act_info
        )

        self.current_step += 1
        done = self.current_step >= self.max_steps
        if done:
            my_project_backend.reduce(
                self.current_pool_id, "LOCAL_BKZ", min(self.dim, 40), 0
            )
            self.last_info = my_project_backend.reduce(
                self.current_pool_id, "LLL", 2, 0
            )

        old_best, old_ep_best = self.best_ratio, self.current_ep_best_ratio
        state, new_logb1, new_ratio, new_maxcos, _, new_logdef = self._build_state(
            self.last_info
        )
        self._c_logb1, self._c_maxcos, self._c_logdef = (
            new_logb1,
            new_maxcos,
            new_logdef,
        )

        R_ratio = old_logb1 - new_logb1
        R_orth = old_maxcos - new_maxcos
        R_def = old_logdef - new_logdef

        if self.best_ratio < 1.08:
            w, al, gr, cw = 15.0, 8.0, 5.0, 0.08
        elif self.best_ratio < 1.15:
            w, al, gr, cw = 25.0, 3.0, 2.0, 0.12
        else:
            w, al, gr, cw = self.ratio_w, self.alpha, self.gamma_r, self.cost_w

        reward = (
            w * R_ratio
            + al * R_orth
            + gr * R_def
            - cw * (beta / max(b for b, _ in self.action_list))
        )

        if new_ratio < old_best:
            reward += 5.0
        elif new_ratio < old_ep_best:
            reward += 2.0
            self.current_ep_best_ratio = new_ratio
        if R_ratio > 1e-3 and pos <= 2 and beta >= 20:
            reward += 0.1
        if done:
            if new_ratio < old_ep_best:
                reward += 3.0 * (old_ep_best - new_ratio) / self.ratio_scale
                self.current_ep_best_ratio = new_ratio
            if self.current_ep_best_ratio >= self.initial_ep_ratio:
                reward -= 2.0

        self.action_history.append(action_idx)
        if len(self.action_history) > self.repeat_window:
            self.action_history.pop(0)
        rc = self.action_history.count(action_idx)
        if rc >= 2:
            reward -= self.repeat_penalty_base * (rc - 1) ** 1.5

        reward = float(np.clip(reward, -5.0, 50.0))
        info = {
            "beta": beta,
            "pos": pos,
            "b1_GH_ratio": new_ratio,
            "step": self.current_step,
        }
        return state, reward, done, info

    def get_best_payload(self):
        return {
            "dim": self.dim,
            "seed_id": self.seed_id,
            "seed_file": os.path.basename(self.filepath),
            "ratio": self.best_ratio,
            "defect": self.best_defect,
            "max_cos": self.best_max_cos,
            "min_cos": self.best_min_cos,
            "vector": self.best_vector,
            "basis": self.best_basis,
            "episode": self.best_episode,
        }


# ============================================================
# env worker（CPU-only；sieve 走队列）
# ============================================================
def env_worker(remote, parent_remote, filepath, env_id, sieve_req_q, sieve_resp_conn):
    parent_remote.close()
    client = (
        SieveClient(env_id, sieve_req_q, sieve_resp_conn)
        if sieve_req_q is not None
        else None
    )
    env = LatticeEnv(filepath, env_id=env_id, sieve_client=client)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                s, r, d, info = env.step(data)
                if d:
                    s = env.reset()
                remote.send((s, r, d, info))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "get_best":
                remote.send(env.get_best_payload())
            elif cmd == "close":
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, files, envs_per_seed=1, use_sieve=True, helper_dir=""):
        self.files = [f for f in files for _ in range(envs_per_seed)]
        self.num_envs = len(self.files)
        self.env_dims = [parse_dim_seed(f)[0] for f in self.files]
        self.env_seed_ids = [parse_dim_seed(f)[1] for f in self.files]

        # ---- sieve 基础设施：一个共享请求队列 + 每 env 一条应答 Pipe ----
        self.sieve_req_q = mp.Queue() if use_sieve else None
        self.sieve_procs = []
        resp_recv, resp_send = [], []
        if use_sieve:
            for _ in range(self.num_envs):
                r, s = mp.Pipe(duplex=False)
                resp_recv.append(r)
                resp_send.append(s)
            for wid in range(NUM_SIEVE_WORKERS):
                os.environ["A10_ROLE"] = "sieve"
                os.environ["A10_SIEVE_GPU"] = SIEVE_GPUS[wid % len(SIEVE_GPUS)]
                p = mp.Process(
                    target=sieve_worker,
                    args=(self.sieve_req_q, resp_send, helper_dir, wid),
                    daemon=True,
                )
                p.start()
                self.sieve_procs.append(p)
            os.environ["A10_ROLE"] = "main"
            os.environ.pop("A10_SIEVE_GPU", None)

        # ---- env workers（CPU-only）----
        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_envs)]
        )
        os.environ["A10_ROLE"] = "env"
        self.processes = []
        for eid, (wr, r, f) in enumerate(
            zip(self.work_remotes, self.remotes, self.files)
        ):
            p = mp.Process(
                target=env_worker,
                args=(
                    wr,
                    r,
                    f,
                    eid,
                    self.sieve_req_q,
                    resp_recv[eid] if use_sieve else None,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
        os.environ["A10_ROLE"] = "main"
        for wr in self.work_remotes:
            wr.close()

    def reset_all(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def send_one(self, eid, action):
        self.remotes[eid].send(("step", action))

    def recv_one(self, eid):
        return self.remotes[eid].recv()

    def poll_ready(self, eids):
        return [i for i in eids if self.remotes[i].poll(timeout=0)]

    def get_bests(self):
        for r in self.remotes:
            r.send(("get_best", None))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        if self.sieve_req_q is not None:
            for _ in self.sieve_procs:
                self.sieve_req_q.put(None)
        for p in self.processes:
            p.join(timeout=5)
        for p in self.sieve_procs:
            p.join(timeout=5)


# ============================================================
# 结果保存
# ============================================================
def save_seed_result(results_dir, info, is_update):
    fp = os.path.join(results_dir, f"dim{info['dim']}_seed{info['seed_id']}.txt")
    with open(fp, "a" if is_update else "w") as f:
        if not is_update:
            f.write(
                "=" * 60 + f"\n Dim={info['dim']} Seed={info['seed_id']} "
                f"File={info['seed_file']}\n" + "=" * 60 + "\n--- Initial ---\n"
            )
        else:
            f.write(f"\n--- New Best (Episode {info['episode']}) ---\n")
        f.write(f"  Ratio: {info['ratio']:.8f}\n")
        if info.get("defect") is not None:
            f.write(
                f"  Defect: {info['defect']:.8f}  MaxCos: {info['max_cos']:.6f}  "
                f"MinCos: {info['min_cos']:.6f}\n"
            )
        f.write(f"  b1 = {info.get('vector')}\n")


def save_final_summary(results_dir, all_infos, goal=1.05):
    fp = os.path.join(results_dir, "summary.txt")
    with open(fp, "w") as f:
        reached = [i for i in all_infos if i["ratio"] < goal]
        f.write("=" * 60 + f"\n FINAL SUMMARY (goal<{goal})\n" + "=" * 60 + "\n")
        f.write(f"Reached: {len(reached)}/{len(all_infos)}\n\n")
        for info in sorted(all_infos, key=lambda x: (x["dim"], x["ratio"])):
            st = "✓" if info["ratio"] < goal else " "
            f.write(
                f"  [{st}] dim{info['dim']:3d} seed{info['seed_id']:2d}: "
                f"{info['ratio']:.6f} (ep {info.get('episode', '?')})\n"
            )


# ============================================================
# 训练主循环
# ============================================================
def train_all(
    vec_env,
    agent,
    results_dir,
    total_updates=200000,
    train_every=4,
    log_every=4000,
    save_every=8000,
    goal_threshold=1.05,
    resume_extra=None,
):
    os.makedirs(results_dir, exist_ok=True)
    num_envs = vec_env.num_envs
    dims = vec_env.env_dims

    global_best, global_info = {}, {}
    history = {"loss": [], "best_min": []}
    updates = int((resume_extra or {}).get("updates", 0))
    env_steps = int((resume_extra or {}).get("env_steps", 0))

    if resume_extra:
        global_best.update(resume_extra.get("global_best", {}))
        global_info.update(resume_extra.get("global_info", {}))
        history = resume_extra.get("history", history)

    states = vec_env.reset_all()
    state_by_eid = {e: states[e] for e in range(num_envs)}
    prev_s = [None] * num_envs
    prev_a = [None] * num_envs

    def eps_now():
        return max(0.05, 0.3 * (1.0 - updates / max(1, total_updates)))

    a0 = agent.act_envs(state_by_eid, eps_now())
    for e in range(num_envs):
        prev_s[e], prev_a[e] = states[e], a0[e]
        vec_env.send_one(e, a0[e])
    pending = set(range(num_envs))

    t_start = time.time()
    while updates < total_updates:
        ready = vec_env.poll_ready(list(pending))
        if not ready:
            time.sleep(0.0005)
            continue

        newly = {}
        for e in ready:
            obs, rew, done, info = vec_env.recv_one(e)
            pending.discard(e)
            agent.remember(dims[e], prev_s[e], prev_a[e], rew, obs, done)
            states[e] = obs
            newly[e] = obs
            env_steps += 1

            if env_steps % train_every == 0:
                loss = agent.learn()
                if loss > 0:
                    updates += 1
                    history["loss"].append(loss)
                    if updates % 500 == 0:
                        agent.step_scheduler()

        acts = agent.act_envs(newly, eps_now())
        for e in newly:
            prev_s[e], prev_a[e] = states[e], acts[e]
            vec_env.send_one(e, acts[e])
            pending.add(e)

        if env_steps % log_every < len(ready):
            bests = vec_env.get_bests()
            reached = 0
            for b in bests:
                key = (b["dim"], b["seed_id"])
                if b["ratio"] < global_best.get(key, float("inf")):
                    first = key not in global_info
                    global_best[key] = b["ratio"]
                    global_info[key] = b
                    save_seed_result(results_dir, b, is_update=not first)
                    _log(f"  ★ dim{b['dim']} seed{b['seed_id']} best={b['ratio']:.8f}")
                if global_best.get(key, 9) < goal_threshold:
                    reached += 1
            best_min = min(global_best.values()) if global_best else float("inf")
            history["best_min"].append(best_min)
            rate = env_steps / max(1e-6, time.time() - t_start)
            _status(
                f"upd {updates}/{total_updates} | ε{eps_now():.3f} | "
                f"loss{history['loss'][-1] if history['loss'] else 0:.4f} | "
                f"bestmin {best_min:.6f} | reached {reached}/{len(global_best)} | "
                f"{rate:.0f} env-steps/s"
            )

        if env_steps % save_every < len(ready):
            agent.save(
                os.path.join(results_dir, "shared_resume.pth"),
                extra={
                    "updates": updates,
                    "env_steps": env_steps,
                    "global_best": global_best,
                    "global_info": global_info,
                    "history": history,
                },
            )

    for b in vec_env.get_bests():
        key = (b["dim"], b["seed_id"])
        if b["ratio"] < global_best.get(key, float("inf")):
            global_best[key] = b["ratio"]
            global_info[key] = b
    save_final_summary(results_dir, list(global_info.values()), goal_threshold)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"])
    plt.title("loss")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history["best_min"])
    plt.axhline(goal_threshold, color="r", ls="--")
    plt.title("global best ratio")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "training.png"))
    plt.close()
    print("\nDone. Summary ->", os.path.join(results_dir, "summary.txt"))
    return history


# ============================================================
# main
# ============================================================
def gather_files(dataset_dir, dims, seeds_per_dim=2):
    files = []
    for dim in dims:
        fs = sorted(
            glob.glob(os.path.join(dataset_dir, f"svpchallengedim{dim}seed*.txt")),
            key=lambda p: parse_dim_seed(p)[1],
        )
        if seeds_per_dim:
            fs = fs[:seeds_per_dim]
        files.extend(fs)
    return files


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    print("Learner device:", DEVICE)
    n_gpu = torch.cuda.device_count()
    print(
        f"Visible GPUs (learner): {n_gpu} | learner -> physical GPU0 | "
        f"sieve {NUM_SIEVE_WORKERS} workers spread over GPUs {sorted(set(SIEVE_GPUS))}"
    )
    print(f"backend: {getattr(my_project_backend, '__file__', '?')}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}  G6K_ROOT={G6K_ROOT}")

    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(
        PROJECT_ROOT, "results", "a10_shared"
    )  # a11 checkpoint 不会被自动加载
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 确定 g6k_env.py 目录（不猜测：显式 > 搜索 PROJECT_ROOT）----
    helper_dir = discover_helper_dir(G6K_HELPER_DIR, PROJECT_ROOT)
    print(f"G6K_HELPER_DIR={helper_dir or '(未找到 g6k_env.py)'}")

    # ---- 启动前单独验证 g6k（GPU 能力），避免起 62 个 env 后才发现不可用 ----
    probe = validate_g6k(helper_dir, SIEVE_GPUS[0])
    print(f"[g6k probe] {probe}")
    if not probe or not probe.get("ok"):
        if os.environ.get("A10_ALLOW_NO_SIEVE") == "1":
            print(
                "[warn] g6k 不可用，A10_ALLOW_NO_SIEVE=1 -> sieve 关闭，beta>=40 回退 ENUM"
            )
            USE_SIEVE = False
        else:
            print(
                "[fatal] g6k 自检失败。修好 g6k / 路径后重试；"
                "或设 A10_ALLOW_NO_SIEVE=1 用纯 CPU-ENUM 跑。"
            )
            sys.exit(1)
    else:
        USE_SIEVE = True
        if not probe.get("gpu_params_ok"):
            print(
                "[warn] SieverParams(gpus=1,bdgl) 不可用 -> 将回退 CPU bgj1（sieve 会很慢）"
            )

    DIMS_TO_RUN = list(range(50, 81))
    files = gather_files(DATASET_DIR, DIMS_TO_RUN, seeds_per_dim=2)
    if not files:
        print("No dataset files found in", DATASET_DIR)
        sys.exit(1)

    vec_env = SubprocVecEnv(
        files, envs_per_seed=1, use_sieve=USE_SIEVE, helper_dir=helper_dir
    )
    print(
        f"Total envs: {vec_env.num_envs} | dims: {sorted(set(vec_env.env_dims))} | "
        f"sieve workers: {len(vec_env.sieve_procs)}"
    )

    agent = DQNAgent(
        num_globals=NUM_GLOBALS,
        batch_size=128,
        dims_per_update=3,
        capacity_per_dim=12000,
    )

    resume_extra = agent.load(os.path.join(RESULTS_DIR, "shared_resume.pth"))
    if resume_extra:
        print("Resumed a10 shared checkpoint.")

    train_all(
        vec_env,
        agent,
        RESULTS_DIR,
        total_updates=200000,
        train_every=4,
        log_every=4000,
        save_every=8000,
        goal_threshold=0.85,
        resume_extra=resume_extra,
    )
    vec_env.close()

# ============================================================
# 调优备注（4×48G）：
#  1. watch -n1 nvidia-smi 看 4 卡 GPU-Util。都没打满 -> 加 sieve context：
#     NUM_SIEVE_WORKERS=12, SIEVE_GPUS=["0","1","2","3"]*3。
#  2. 若 learner 训练明显被拖慢（loss 更新变稀）-> 给 GPU0 减负：
#     NUM_SIEVE_WORKERS=7, SIEVE_GPUS=["1","2","3","0","1","2","3"]（GPU0 只 1 个陪 learner）。
#  3. CPU 别超订：62 env(OMP=1) + 8 sieve(各 SIEVE_THREADS=2)=78 线程 + main(4)。
#     核数不够就降 seeds_per_dim 或 SIEVE_THREADS。
# ============================================================
