# 改进：1.
import my_project_backend
import os, sys, math, time, random, re, pickle
import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"  # 控制 OpenMP 线程数
os.environ["MKL_NUM_THREADS"] = "4"  # 控制 Intel 数学库线程数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(4)
import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp

# ------------------------------
# Config
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)


def matrix_to_string(basis):
    lines = [" ".join(str(x) for x in row) for row in basis]
    return "[" + "\n".join(f"[{l}]" for l in lines) + "]"


def string_to_matrix_fast(mat_str):
    content = mat_str.strip()[1:-1]
    if not content:
        return []
    rows = content.split("\n")
    return [
        [int(x) for x in r.replace("[", "").replace("]", "").split()]
        for r in rows
        if r.strip()
    ]


def parse_challenge_file(filepath):
    """
    读取格基数据集文件 (如 svpchallengedim68seed0.txt)，
    并将其解析为 Python 的 2D 整数列表。
    自动处理可能包含的 '[' 或 ']' 符号。
    """
    matrix = []
    with open(filepath, "r") as f:
        # 读取整个文件，去掉所有括号
        content = f.read().replace("[", "").replace("]", "")
        # 按行分割
        for line in content.strip().split("\n"):
            if line.strip():  # 如果不是空行
                # 将每一行的数字用空格分割并转换为整数
                row = [int(x) for x in line.split()]
                matrix.append(row)
    return matrix


# ------------------------------
# Neural Network (Transformer + NoisyNet)
# ------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
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
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(x, self.weight_mu, self.bias_mu)


class AxialCNN_DuelingDDQN(nn.Module):
    def __init__(self, max_dim, action_dim):
        super().__init__()
        self.max_dim = max_dim
        self.token_dim = max_dim  # 移除 GS，只剩 max_dim (余弦矩阵行维度)

        # 分支 A: 沿列扫描 (引入空洞卷积，感受野扩大至 33x1)
        self.col_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(4, 0), dilation=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(8, 0), dilation=(4, 1)),
            nn.LeakyReLU(0.01),
        )

        # 分支 B: 沿行扫描 (引入空洞卷积，感受野扩大至 1x33)
        self.row_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 4), dilation=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 8), dilation=(1, 4)),
            nn.LeakyReLU(0.01),
        )

        # 融合层：因为上面输出了 64+64=128 通道，所以这里输入改为 128
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.LeakyReLU(0.01)
        )
        self.scalar_mlp = nn.Sequential(nn.Linear(4, 32), nn.LeakyReLU(0.01))

        self.grid_size = min(8, max_dim - 1)
        cnn_flat_size = 64 * self.grid_size * self.grid_size

        # 融合尺寸 = 4096 (网格特征) + 32 (标量特征) = 4128
        self.fusion = nn.Sequential(
            nn.Linear(cnn_flat_size + 32, 256), nn.LeakyReLU(0.01)
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(256, 128), nn.LeakyReLU(0.01), NoisyLinear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(256, 128), nn.LeakyReLU(0.01), NoisyLinear(128, action_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        tokens_flat_size = (self.max_dim - 1) * self.token_dim
        tokens_flat = x[:, :tokens_flat_size]
        scalars = x[:, tokens_flat_size:]

        # 直接将 tokens 转为图像张量 (Batch, 1, max_dim-1, max_dim)
        cos_matrix = tokens_flat.view(batch_size, 1, self.max_dim - 1, self.token_dim)

        col_feat = self.col_conv(cos_matrix)
        row_feat = self.row_conv(cos_matrix)
        concat_feat = torch.cat([col_feat, row_feat], dim=1)
        fused_matrix = self.fuse_conv(concat_feat)

        # 【核心修改 2】：使用网格池化，保留空间坐标信息！
        # 计算 Max 和 Avg
        pool_max = F.adaptive_max_pool2d(fused_matrix, (self.grid_size, self.grid_size))
        pool_avg = F.adaptive_avg_pool2d(fused_matrix, (self.grid_size, self.grid_size))
        # 混合 (各占一半权重)
        grid_out = 0.5 * pool_max + 0.5 * pool_avg

        # 展平后大小永远是 Batch x 4096
        cnn_out = grid_out.view(batch_size, -1)

        scalar_out = self.scalar_mlp(scalars)

        fused = torch.cat([cnn_out, scalar_out], dim=1)
        feat = self.fusion(fused)

        v = self.value_stream(feat)
        a = self.adv_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ------------------------------
# Agent
# ------------------------------
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
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

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
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    PER_e = 1e-5  # 防止优先级为零
    PER_a = 0.6  # 优先级指数：越大越偏向高 TD error 样本
    PER_b = 0.4  # 重要性采样权重初始值
    PER_b_increment = 0.001  # 每次采样后 b 递增，逐渐趋近均匀采样

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.PER_e) ** self.PER_a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / n
        self.PER_b = min(1.0, self.PER_b + self.PER_b_increment)

        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is None:
                s = random.uniform(0, self.tree.total())
                idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probs = np.array(priorities) / (self.tree.total() + 1e-10)
        is_weights = (self.tree.n_entries * sampling_probs + 1e-10) ** (-self.PER_b)
        is_weights /= is_weights.max()
        return batch, idxs, torch.FloatTensor(is_weights)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


class DQNAgent:
    def __init__(self, max_dim, state_dim, action_dim, batch_size=64):
        self.device = device
        self.batch_size = batch_size
        self.q_net = AxialCNN_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net = AxialCNN_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(
            self.q_net.parameters(), lr=3e-4, weight_decay=1e-4
        )
        # 学习率余弦退火：开始快，后期精细
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500, eta_min=1e-5
        )
        # PER 替换普通 deque
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.tau = 0.005

    def save_checkpoint(self, model_path, memory_path=None):
        torch.save(self.q_net.state_dict(), model_path)

    def load_checkpoint(self, model_path, memory_path=None):
        if os.path.exists(model_path):
            self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())

    def act_batch(self, states, is_training=True):
        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        if is_training:
            self.q_net.train()
            self.q_net.reset_noise()
        else:
            self.q_net.eval()
        with torch.no_grad():
            return self.q_net(s).argmax(dim=1).cpu().numpy().tolist()

    def remember(self, s, a, r, ns, d):
        s_fp16 = s.astype(np.float16)
        ns_fp16 = ns.astype(np.float16)
        # 新样本给最大 TD error = 1.0，确保至少被采样一次
        self.memory.add(1.0, (s_fp16, a, r, ns_fp16, float(d)))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        batch, tree_idxs, is_weights = self.memory.sample(self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(np.array(ns), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        is_weights = is_weights.to(self.device).unsqueeze(1)

        self.q_net.train()
        self.q_net.reset_noise()

        with torch.no_grad():
            next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
            self.target_net.reset_noise()
            next_q = self.target_net(ns).gather(1, next_actions)
            target_q = r + (1.0 - d) * self.gamma * next_q

        self.optimizer.zero_grad()
        curr_q = self.q_net(s).gather(1, a)

        # 计算每个样本的 TD error，用于更新优先级
        td_errors = (curr_q - target_q).detach().abs().cpu().numpy().flatten()
        for i, idx in enumerate(tree_idxs):
            self.memory.update(idx, td_errors[i])

        # IS 加权损失：补偿非均匀采样带来的偏差
        element_loss = F.smooth_l1_loss(curr_q, target_q, reduction="none")
        loss = (is_weights * element_loss).mean()
        loss.backward()

        max_grad = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                max_grad = max(max_grad, p.grad.abs().max().item())

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.75)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        return float(loss.item()), float(max_grad)

    def step_scheduler(self):
        self.scheduler.step()


# ------------------------------
# Environment
# ------------------------------
class LatticeEnv:
    def __init__(self, matrix_path, max_dim=250):
        self.initial_matrix_list = parse_challenge_file(matrix_path)
        self.dim = len(self.initial_matrix_list)
        self.max_dim = max_dim  # 用于网络输入的固定最大维度
        raw_matrix_str = matrix_to_string(self.initial_matrix_list)

        self.max_steps = self.dim * 3

        self.alpha = 0.3
        self.ratio_w = 40.0
        self.gamma = 0.1
        self.cost_w = 0.005
        self.last_pos = None

        # 动作空间：包含 beta 和 pos
        self.betas = [10 + 3 * i for i in range(9)]
        self.action_list = [
            (b, p) for b in self.betas if b <= self.dim for p in range(self.dim - b + 1)
        ]
        self.num_actions = len(self.action_list)

        init_res = my_project_backend.run_reduction_rust(raw_matrix_str, "LLL", 2, 0)
        self.initial_matrix_str = init_res["matrix_str"]

        rust_eval = my_project_backend.evaluate_state_rust(
            self.initial_matrix_str, 0, 0
        )
        self.initial_gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)
        self.log_vol = np.sum(self.initial_gs_logs)
        self.log_GH = (self.log_vol / self.dim) + 0.5 * math.log(
            self.dim / (2 * math.pi * math.e)
        )

        # 【动态归一化因子】：根据初始状态动态设定
        initial_log_defect = float(init_res["log_prod"] - self.log_vol)
        initial_log_ratio = float(self.initial_gs_logs[0] - self.log_GH)
        self.defect_scale = max(abs(initial_log_defect), 1.0)
        self.ratio_scale = max(abs(initial_log_ratio), 1.0)

        # 状态维度计算: (max_dim * (1 + max_dim)) + 4
        self.state_dim = ((self.max_dim - 1) * self.max_dim) + 4

        self.best_ratio = float("inf")
        self.best_vector = None
        self.best_max_cos = None
        self.best_min_cos = None
        self.best_defect = None

    def reset(self):
        self.current_step = 0
        self.useless_act_set = set()
        res = my_project_backend.run_reduction_rust(
            self.initial_matrix_str, "LLL", 2, 0
        )
        self.current_matrix_str = res["matrix_str"]
        self.last_rust_info = res

        state, _, current_ratio, _, _, _ = self._get_state_and_update_best(
            self.current_matrix_str, res
        )
        self.current_ep_best_ratio = current_ratio
        return state

    def _get_state_and_update_best(self, mat_str, rust_info):
        C = np.array(rust_info["cos_matrix"], dtype=np.float32)
        lower = C[np.tril_indices(self.dim, -1)].astype(np.float32)
        max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))

        C = C + C.T
        rust_eval = my_project_backend.evaluate_state_rust(mat_str, 0, 0)
        gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)

        log_b1 = gs_logs[0]
        log_defect = float(rust_info["log_prod"] - self.log_vol)
        log_ratio = float(log_b1 - self.log_GH)

        norm_log_defect = float(np.tanh(log_defect / self.defect_scale))
        norm_log_ratio = float(np.tanh(log_ratio / self.ratio_scale))

        tokens = np.zeros((self.max_dim - 1, self.max_dim), dtype=np.float32)
        for i in range(self.dim - 1):
            # 左半边：取第 i+1 行的下三角部分
            tokens[i, : i + 1] = C[i + 1, : i + 1]
            # 右半边：取第 i 行的上三角部分
            tokens[i, i + 1 : self.dim] = C[i, i + 1 : self.dim]

        tokens_flat = tokens.flatten()
        scalars = np.array(
            [max_cos, min_cos, norm_log_defect, norm_log_ratio], dtype=np.float32
        )
        state_vec = np.concatenate([tokens_flat, scalars], axis=0)

        true_b1_GH_ratio = float(math.exp(log_ratio))
        if true_b1_GH_ratio < self.best_ratio:
            self.best_ratio = true_b1_GH_ratio
            self.best_max_cos = max_cos
            self.best_min_cos = min_cos
            self.best_defect = log_defect
            mat_list = string_to_matrix_fast(mat_str)
            if mat_list:
                self.best_vector = mat_list[0]

        return state_vec, log_b1, true_b1_GH_ratio, max_cos, min_cos, log_defect

    def step(self, action_idx):
        beta, pos = self.action_list[action_idx]
        _, old_log_b1, _, old_max_cos, _, old_log_def = self._get_state_and_update_best(
            self.current_matrix_str, self.last_rust_info
        )

        # 1. 执行局部 BKZ
        new_rust_info = my_project_backend.run_reduction_rust(
            self.current_matrix_str, "LOCAL_BKZ", beta, pos
        )
        self.current_matrix_str = new_rust_info["matrix_str"]

        # 2. 全局 LLL
        lll_info = my_project_backend.run_reduction_rust(
            self.current_matrix_str, "LLL", 2, 0
        )
        self.current_matrix_str = lll_info["matrix_str"]
        self.last_rust_info = lll_info

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # ★ 关键修复：先保存旧的 best_ratio，再调用 _get_state_and_update_best
        old_best_ratio = self.best_ratio
        old_ep_best_ratio = self.current_ep_best_ratio

        state, new_log_b1, new_ratio, new_max_cos, _, new_log_def = (
            self._get_state_and_update_best(
                self.current_matrix_str, self.last_rust_info
            )
        )

        R_global = old_log_def - new_log_def
        R_ratio = old_log_b1 - new_log_b1
        R_cos = old_max_cos - new_max_cos

        # ★ 修复：用保存的旧值比较
        breakthrough_bonus = 0.0
        if new_ratio < old_best_ratio:
            breakthrough_bonus = 5.0
        elif new_ratio < old_ep_best_ratio:
            breakthrough_bonus = 2.0
            self.current_ep_best_ratio = new_ratio

        gain = self.alpha * R_global + self.ratio_w * R_ratio + self.gamma * R_cos
        reward = gain - self.cost_w * beta

        if pos <= 2 and beta >= 20:
            reward += 0.1

        # ★ 重设惩罚体系：温和且有信息量
        if gain <= 1e-5 and breakthrough_bonus == 0.0:
            # 无效动作只给小惩罚，不累加
            # 重复的 (beta, pos) 稍微多罚一点点
            action_key = (beta, pos)
            if action_key in self.useless_act_set:
                reward -= 1.0  # 重复无效：-1（原来是 -17！）
            else:
                reward -= 0.3  # 首次无效：-0.3（原来是 -2）
                self.useless_act_set.add(action_key)
        else:
            # 有增益时只清除当前 (beta, pos)，不清除全部
            action_key = (beta, pos)
            self.useless_act_set.discard(action_key)

        reward += breakthrough_bonus
        reward = float(np.clip(reward, -5.0, 50.0))  # 缩小负向裁剪范围

        info = {
            "beta": beta,
            "pos": pos,
            "b1_GH_ratio": new_ratio,
            "step": self.current_step,
        }
        return state, float(reward), done, info


# ------------------------------
# Multiprocessing Environment Workers
# ------------------------------
def env_worker(remote, parent_remote, matrix_path, max_dim):
    parent_remote.close()
    env = LatticeEnv(matrix_path, max_dim)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                state, reward, done, info = env.step(data)
                if done:
                    state = env.reset()
                remote.send((state, reward, done, info))
            elif cmd == "reset":
                state = env.reset()
                remote.send(state)
            elif cmd == "get_best":
                remote.send(
                    (
                        env.best_ratio,
                        env.best_defect,
                        env.best_max_cos,
                        env.best_min_cos,
                        env.best_vector,
                    )
                )
            elif cmd == "close":
                remote.close()
                break
    # 【修改 4】：增加 EOFError 捕获。当主进程被杀或意外退出时，子进程安静退出
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # 确保资源被释放
        remote.close()


class SubprocVecEnv:
    def __init__(self, num_envs, matrix_path, max_dim=250):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [
            mp.Process(
                target=env_worker, args=(work_remote, remote, matrix_path, max_dim)
            )
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)
        ]
        for p in self.processes:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_bests(self):
        for remote in self.remotes:
            remote.send(("get_best", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()


# ------------------------------
# Main & Train
# ------------------------------
def save_best_results(
    filepath, dim, ratio, defect, max_cos, min_cos, vector, is_initial=False
):
    """辅助函数：将结果写入本地 txt 文件"""
    mode = "w" if is_initial else "a"  # 初始覆盖，后续追加
    with open(filepath, mode) as f:
        if is_initial:
            f.write(f"=== Lattice Reduction Results (Dim {dim}) ===\n")
            f.write("--- Initial LLL State ---\n")
        else:
            f.write("\n--- New Best State Found ---\n")

        f.write(f"Best norm(b_1)/GH Ratio: {ratio:.8f}\n")
        f.write(f"Global Orthogonality Defect: {defect:.8f}\n")
        f.write(f"Max Cosine (Min Angle): {max_cos:.8f}\n")
        f.write(f"Min Cosine (Max Angle): {min_cos:.8f}\n")  # 修正变量名
        f.write("Best Vector (b_1):\n")
        if vector:
            f.write(" ".join(str(x) for x in vector) + "\n")


def train(
    vec_env,
    agent,
    num_envs,
    max_steps,
    dim,
    best_file_path,  # <--- 确保这里有这个参数
    episodes=200,
    print_every=10,
    save_dir="results",
    train_freq=4,
    utd=1.0,
):
    history = {"reward": [], "loss": [], "ratio_min": []}
    best_known_ratio = float("inf")

    # 1. 先重置环境，此时环境内部的 None 会被真实的数值覆盖
    states = vec_env.reset()

    # 2. 【修复】：在这里获取初始状态并保存
    initial_bests = vec_env.get_bests()
    init_ratio, init_defect, init_max_cos, init_min_cos, init_vector = initial_bests[0]
    save_best_results(
        best_file_path,
        dim,
        init_ratio,
        init_defect,
        init_max_cos,
        init_min_cos,
        init_vector,
        is_initial=True,
    )

    # 【新增】：跨 Episode 累积日志列表
    accumulated_ep_logs = []
    total_steps = 0

    for ep in range(1, episodes + 1):
        ep_rewards = np.zeros(num_envs)
        ep_ratios = []
        losses = []

        # 记录当前 Episode 某一个环境的动作与梯度轨迹
        ep_action_logs = []

        for step in range(max_steps):
            total_steps += 1  # 步数累加
            actions = agent.act_batch(states, is_training=True)
            next_states, rewards, batch_dones, infos = vec_env.step(actions)

            step_max_grad = 0.0

            # --- 步骤 A: 收集经验 (严格遵守 num_envs 长度) ---
            for i in range(num_envs):
                agent.remember(
                    states[i], actions[i], rewards[i], next_states[i], batch_dones[i]
                )
                ep_rewards[i] += rewards[i]
                ep_ratios.append(infos[i]["b1_GH_ratio"])

            # --- 步骤 B: 集中火力训练 ---
            # 不每步都更新。当攒够 4 步的数据后，一口气更新 8 次。
            # 平均下来依然是每步更新 2 次 (UTD 没变)，但是大大减少了主进程在收发数据和算梯度之间的来回切换。
            if total_steps % train_freq == 0:
                for _ in range(8):
                    loss, max_grad = agent.replay()
                    if loss != 0.0:
                        losses.append(loss)
                        step_max_grad = max(step_max_grad, max_grad)

            # 【提取动作数据】：为了避免日志刷屏，我们只提取 Env 0 的数据展示
            pos = infos[0]["pos"]
            beta = infos[0]["beta"]
            # 只有在触发训练的那一步，g(梯度)才有数值，平时为 0.0
            ep_action_logs.append(f"(p:{pos:2d}, b:{beta:2d}, g:{step_max_grad:.2f})")

            states = next_states

        avg_ep_reward = np.mean(ep_rewards)
        history["reward"].append(avg_ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        ep_min_ratio = min(ep_ratios) if ep_ratios else float("inf")
        history["ratio_min"].append(ep_min_ratio)

        bests = vec_env.get_bests()
        global_best_ratio = min([b[0] for b in bests])

        if global_best_ratio < best_known_ratio:
            best_known_ratio = global_best_ratio
            model_path = os.path.join(save_dir, f"agent6L_best_model_dim{dim}.pth")
            memory_path = os.path.join(
                save_dir, f"agent6L_best_model_dim_best_memory_dim{dim}.pkl"
            )
            agent.save_checkpoint(model_path, memory_path)

            # 找出是哪个环境找到了最优解，提取详细信息并立即保存到本地
            best_idx = np.argmin([b[0] for b in bests])
            b_ratio, b_defect, b_max_cos, b_min_cos, b_vector = bests[best_idx]
            save_best_results(
                best_file_path,
                dim,
                b_ratio,
                b_defect,
                b_max_cos,
                b_min_cos,
                b_vector,
                is_initial=False,
            )

            print(
                f"agent6LHisBest find {global_best_ratio:.4f}! weight, memory and txt backed up"
            )

        if best_known_ratio < 1.05:
            print(
                f"\n [Dim {dim}] Goal reached! Current best ratio {best_known_ratio:.4f} < 1.05 (At ep {ep})! The training of this dimension is stopped early."
            )
            break

        if ep % print_every == 0:
            # 1. 打印动作轨迹
            print(f"\n[{'=' * 15} Ep {ep} Trajectory (Env 0) {'=' * 15}]")
            for idx in range(
                0, len(ep_action_logs), 4
            ):  # 每行打印 4 个动作步骤防止超出屏幕
                print(" -> ".join(ep_action_logs[idx : idx + 4]))
            print("-" * 55)

            # 2. 生成当前 Ep 日志并加入累积列表
            current_log = f"Ep {ep:4d} | Avg R: {avg_ep_reward:9.3f} | Loss: {history['loss'][-1]:7.4f} | Ep min ratio: {ep_min_ratio:.4f} | historical best: {global_best_ratio:.4f}"
            accumulated_ep_logs.append(current_log)

            # 3. 集中打印全部历史日志
            print("\n=== Training History Log ===")
            for log in accumulated_ep_logs:
                print(log)
            print("============================\n")
        agent.step_scheduler()

    return history


def run_experiment(dim, dataset_dir, results_dir, num_envs=4):
    max_dim = ((dim + 7) // 8) * 8
    max_dim = max(max_dim, 16)

    print(f"[Dim {dim}] max_dim={max_dim}, {num_envs} envs...")
    train_file = os.path.join(dataset_dir, f"svpchallengedim{dim}seed0.txt")
    best_file_path = os.path.join(results_dir, f"A6_L_best_results_low_dim{dim}.txt")

    vec_env = SubprocVecEnv(num_envs, train_file, max_dim=max_dim)
    temp_env = LatticeEnv(train_file, max_dim=max_dim)
    agent = DQNAgent(
        max_dim=max_dim, state_dim=temp_env.state_dim, action_dim=temp_env.num_actions
    )

    model_path = os.path.join(results_dir, f"agent6_2UP_best_model_low_dim{dim}.pth")
    # 注意：PER 版不再保存/加载记忆
    agent.load_checkpoint(model_path)

    history = train(
        vec_env,
        agent,
        num_envs,
        max_steps=temp_env.max_steps,
        dim=dim,
        episodes=500,
        best_file_path=best_file_path,
        print_every=10,
        save_dir=results_dir,
    )  # 获取最佳结果
    bests = vec_env.get_bests()
    best_idx = np.argmin([b[0] for b in bests])
    best_ratio = bests[best_idx][0]

    plt.figure(figsize=(14, 6))

    # 左图：奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["reward"], label="Average Reward")
    plt.title(f"Dim {dim} - Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # 右图：最短向量比率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["ratio_min"], label="Ep Min Ratio", color="orange")
    # 画一条红色的虚线，标出全局找到的最优值作为参考底线
    plt.axhline(
        y=best_ratio, color="r", linestyle="--", label=f"Global Best ({best_ratio:.4f})"
    )
    plt.title(f"Dim {dim} - Approximation Factor (norm(b1)/GH)")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)

    # 拼装保存路径：直接存入 results 文件夹
    plot_path = os.path.join(results_dir, f"training_evolution_dim{dim}.png")
    plt.savefig(plot_path)
    plt.close()

    vec_env.close()
    print(f"[Dim {dim}] Training Complete! Results saved.")
    return dim


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    DIMS_TO_RUN = [40, 43, 44, 47, 50]
    for dim in DIMS_TO_RUN:
        run_experiment(dim, DATASET_DIR, RESULTS_DIR, num_envs=9)
