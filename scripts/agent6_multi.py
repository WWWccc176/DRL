import my_project_backend
import os, sys, math, time, random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "4"  # 控制 OpenMP 线程数
os.environ["MKL_NUM_THREADS"] = "4"  # 控制 Intel 数学库线程数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(4)


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
print("Using device:", device)


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


class SE_Block(nn.Module):
    """Squeeze-and-Excitation: 全局通道注意力"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AxialCNN_DuelingDDQN(nn.Module):
    def __init__(self, max_dim, action_dim):
        super().__init__()
        self.max_dim = max_dim
        self.token_dim = max_dim

        # ---- 维度自适应空洞率 ----
        if max_dim <= 32:
            dilations = [1, 2, 4]  # 感受野 29
        elif max_dim <= 64:
            dilations = [1, 3, 9]  # 感受野 53
        elif max_dim <= 128:
            dilations = [1, 4, 16]  # 感受野 85
        else:
            dilations = [1, 6, 24]  # 感受野 125

        # ---- 分支 A: 沿列（垂直）----
        self.col_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                32,
                64,
                kernel_size=(5, 1),
                padding=(dilations[1] * 2, 0),
                dilation=(dilations[1], 1),
            ),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                64,
                128,
                kernel_size=(5, 1),
                padding=(dilations[2] * 2, 0),
                dilation=(dilations[2], 1),
            ),
            nn.LeakyReLU(0.01),
        )

        # ---- 分支 B: 沿行（水平）----
        self.row_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                32,
                64,
                kernel_size=(1, 5),
                padding=(0, dilations[1] * 2),
                dilation=(1, dilations[1]),
            ),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                64,
                128,
                kernel_size=(1, 5),
                padding=(0, dilations[2] * 2),
                dilation=(1, dilations[2]),
            ),
            nn.LeakyReLU(0.01),
        )

        # ---- 融合层 ----
        # 128 + 128 = 256 通道输入（修复原来的 128 bug）
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),  # 2D 交互
            nn.LeakyReLU(0.01),
        )

        # ---- SE 注意力 ----
        self.se = SE_Block(64, reduction=4)

        # ---- 标量分支 ----
        # 输入: gs_profile(max_dim) + scalars(5) = max_dim + 5
        scalar_input_dim = max_dim + 5
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.LeakyReLU(0.01),
        )

        # ---- 池化 + 展平 ----
        self.grid_size = min(8, max_dim - 1)
        cnn_flat_size = 64 * self.grid_size * self.grid_size

        # ---- 融合 MLP ----
        # CNN 输出 cnn_flat_size + scalar_mlp 输出 64
        self.fusion = nn.Sequential(
            nn.Linear(cnn_flat_size + 64, 256),
            nn.LeakyReLU(0.01),
        )

        # ---- Dueling 头 ----
        self.value_stream = nn.Sequential(
            NoisyLinear(256, 128), nn.LeakyReLU(0.01), NoisyLinear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(256, 128), nn.LeakyReLU(0.01), NoisyLinear(128, action_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # ---- 拆分 state 向量 ----
        tokens_flat_size = (self.max_dim - 1) * self.token_dim
        gs_size = self.max_dim
        scalar_size = 5

        tokens_flat = x[:, :tokens_flat_size]
        gs_and_scalars = x[:, tokens_flat_size:]  # max_dim + 5

        # ---- CNN 路径 ----
        cos_matrix = tokens_flat.view(batch_size, 1, self.max_dim - 1, self.token_dim)

        col_feat = self.col_conv(cos_matrix)  # (B, 128, H, W)
        row_feat = self.row_conv(cos_matrix)  # (B, 128, H, W)
        concat_feat = torch.cat([col_feat, row_feat], dim=1)  # (B, 256, H, W)
        fused_matrix = self.fuse_conv(concat_feat)  # (B, 64, H, W)
        fused_matrix = self.se(fused_matrix)  # SE 注意力

        pool_max = F.adaptive_max_pool2d(fused_matrix, (self.grid_size, self.grid_size))
        pool_avg = F.adaptive_avg_pool2d(fused_matrix, (self.grid_size, self.grid_size))
        grid_out = 0.5 * pool_max + 0.5 * pool_avg
        cnn_out = grid_out.view(batch_size, -1)  # (B, 64*8*8)

        # ---- 标量路径 ----
        scalar_out = self.scalar_mlp(gs_and_scalars)  # (B, 64)

        # ---- 融合 + Dueling ----
        fused = torch.cat([cnn_out, scalar_out], dim=1)
        feat = self.fusion(fused)

        v = self.value_stream(feat)
        a = self.adv_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


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
    def __init__(self, max_dim, state_dim, action_dim, batch_size=128):
        self.device = device
        self.batch_size = batch_size
        self.q_net = AxialCNN_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net = AxialCNN_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=6e-5,
            weight_decay=1e-4,  # 原来 3e-4 → 5e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=500,
            eta_min=1e-6,  # eta_min 降低
        )
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.tau = 0.0025  # 原来 0.005 → 0.002，target 网络更慢更新

    def save_checkpoint(self, model_path, memory_path=None):
        torch.save(self.q_net.state_dict(), model_path)

    def load_checkpoint(self, model_path, memory_path=None):
        if os.path.exists(model_path):
            self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())

    def act_batch(self, states, is_training=True, epsilon=0.0):
        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        if is_training:
            self.q_net.train()
            self.q_net.reset_noise()
        else:
            self.q_net.eval()
        with torch.no_grad():
            greedy_actions = self.q_net(s).argmax(dim=1).cpu().numpy().tolist()

        if is_training and epsilon > 0:
            actions = []
            for a in greedy_actions:
                if random.random() < epsilon:
                    actions.append(
                        random.randint(0, self.q_net.adv_stream[-1].out_features - 1)
                    )
                else:
                    actions.append(a)
            return actions
        return greedy_actions

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
    def __init__(self, matrix_path, max_dim=250, seed_idx=None):
        """
        matrix_path: str — 单个文件路径（固定种子分配模式）
                     list — 多个路径（向后兼容，但建议用单个）
        seed_idx: int — 该环境负责的种子编号（用于日志）
        """
        # ---- 固定种子模式 ----
        if isinstance(matrix_path, list):
            # 向后兼容：如果传入列表但指定了 seed_idx，取对应的一个
            if seed_idx is not None and seed_idx < len(matrix_path):
                matrix_path = matrix_path[seed_idx]
            else:
                matrix_path = matrix_path[0]

        self.matrix_path = matrix_path  # 现在是单个 str
        self.seed_idx = seed_idx if seed_idx is not None else 0

        # 从文件名提取种子号（如 svpchallengedim55seed3.txt → 3）
        import re

        match = re.search(r"seed(\d+)", os.path.basename(self.matrix_path))
        self.seed_id = int(match.group(1)) if match else self.seed_idx

        # ---- 反重复机制 ----
        self.action_history = []
        self.repeat_window = 8
        self.repeat_penalty_base = 0.3

        # ---- 加载 ----
        self._load_lattice(self.matrix_path)

        # ---- 动作空间（维度自适应）----
        beta_max = min(int(0.8 * self.dim), 50)
        beta_min = max(8, int(0.15 * self.dim))
        n_betas = 7
        raw = np.geomspace(beta_min, beta_max, n_betas)
        self.betas = sorted(set(max(2, int(round(x))) for x in raw))

        self.action_list = []
        for b in self.betas:
            if b > self.dim:
                continue
            pos_step = max(1, b // 4)
            for p in range(0, self.dim - b + 1, pos_step):
                self.action_list.append((b, p))
            if self.dim - b >= 0 and (b, 0) not in self.action_list:
                self.action_list.insert(0, (b, 0))
        self.num_actions = len(self.action_list)

        # ---- 状态维度 ----
        self.state_dim = (self.max_dim - 1) * self.max_dim + self.max_dim + 5

        # ---- 全局最优追踪 ----
        self.best_ratio = float("inf")
        self.best_vector = None
        self.best_max_cos = None
        self.best_min_cos = None
        self.best_defect = None
        self.best_episode = 0  # ★ 新增：记录在哪个 episode 找到的
        self.best_basis = None  # ★ 新增：保存完整 basis 矩阵
        self.episode_count = 0  # ★ 新增：累计 episode 数
        # ★ 缓存 old state，避免 step() 开头重复调用 evaluate_matrix_rust
        self._cached_log_b1 = 0.0
        self._cached_max_cos = 0.0
        self._cached_log_def = 0.0

    def _load_lattice(self, filepath):
        """改为使用矩阵池"""
        self.initial_matrix_list = parse_challenge_file(filepath)
        self.dim = len(self.initial_matrix_list)
        self.max_dim = ((self.dim + 7) // 8) * 8
        self.max_dim = max(self.max_dim, 16)

        raw_matrix_str = matrix_to_string(self.initial_matrix_list)
        self.max_steps = self.dim * 3

        # 奖励参数（同前）...
        self.ratio_w = 30.0
        self.alpha = 2.0
        self.gamma_r = 1.0
        self.cost_w = 0.15

        # ★ 创建初始矩阵池句柄（LLL 预处理一步完成）
        self.initial_pool_id = my_project_backend.create_matrix_lll_rust(raw_matrix_str)

        # 获取 GSO 信息
        init_eval = my_project_backend.evaluate_matrix_rust(self.initial_pool_id)
        self.initial_gs_logs = np.array(init_eval["gs_log_norms"], dtype=np.float32)
        self.log_vol = np.sum(self.initial_gs_logs)
        self.log_GH = (self.log_vol / self.dim) + 0.5 * math.log(
            self.dim / (2 * math.pi * math.e)
        )

        # 获取 log_prod（做一次 reduce 获取 cos_matrix 等信息）
        init_info = my_project_backend.reduce_rust(
            self.initial_pool_id,
            "LLL",
            2,
            0,  # LLL 已做过，此处幂等
        )

        initial_log_defect = float(init_info["log_prod"] - self.log_vol)
        initial_log_ratio = float(self.initial_gs_logs[0] - self.log_GH)
        self.defect_scale = max(abs(initial_log_defect), 1.0)
        self.ratio_scale = max(abs(initial_log_ratio), 1.0)

        self.current_filepath = filepath

    def reset(self):
        """固定种子：每次 reset 都用同一个文件"""
        self.current_step = 0
        self.action_history = []
        self.episode_count += 1  # ★ 记录 episode

        # ★ 从初始矩阵克隆一份工作副本（不再随机换种子）
        if hasattr(self, "current_pool_id"):
            my_project_backend.free_matrix_rust(self.current_pool_id)
        self.current_pool_id = my_project_backend.clone_matrix_rust(
            self.initial_pool_id
        )

        # LLL 一次
        self.last_rust_info = my_project_backend.reduce_rust(
            self.current_pool_id, "LLL", 2, 0
        )

        state, log_b1, current_ratio, max_cos, _, log_def = (
            self._get_state_and_update_best(self.last_rust_info)
        )
        self.current_ep_best_ratio = current_ratio
        self.initial_ep_ratio = current_ratio
        # ★ 缓存，供下一次 step() 直接取用
        self._cached_log_b1 = log_b1
        self._cached_max_cos = max_cos
        self._cached_log_def = log_def
        return state

    def _get_state_and_update_best(self, rust_info):
        """不再需要 mat_str 参数！"""
        C = np.array(rust_info["cos_matrix"], dtype=np.float32)
        lower = C[np.tril_indices(self.dim, -1)].astype(np.float32)
        max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        C = C + C.T

        # ★ 通过句柄获取 GSO
        rust_eval = my_project_backend.evaluate_matrix_rust(self.current_pool_id)
        gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)

        log_b1 = gs_logs[0]
        log_defect = float(rust_info["log_prod"] - self.log_vol)
        log_ratio = float(log_b1 - self.log_GH)

        norm_log_defect = float(np.tanh(log_defect / self.defect_scale))
        norm_log_ratio = float(np.tanh(log_ratio / self.ratio_scale))

        tokens = np.zeros((self.max_dim - 1, self.max_dim), dtype=np.float32)
        for i in range(self.dim - 1):
            tokens[i, : i + 1] = C[i + 1, : i + 1]
            tokens[i, i + 1 : self.dim] = C[i, i + 1 : self.dim]
        tokens_flat = tokens.flatten()

        gs_profile = np.zeros(self.max_dim, dtype=np.float32)
        gs_normalized = (gs_logs - self.log_GH) / self.ratio_scale
        gs_profile[: self.dim] = np.tanh(gs_normalized)

        scalars = np.array(
            [
                max_cos,
                norm_log_defect,
                norm_log_ratio,
                float(np.tanh((gs_logs[0] - gs_logs[-1]) / self.ratio_scale)),
                float(self.current_step / self.max_steps),
            ],
            dtype=np.float32,
        )

        state_vec = np.concatenate([tokens_flat, gs_profile, scalars], axis=0)

        true_b1_GH_ratio = float(math.exp(log_ratio))
        if true_b1_GH_ratio < self.best_ratio:
            self.best_ratio = true_b1_GH_ratio
            self.best_max_cos = max_cos
            self.best_min_cos = min_cos
            self.best_defect = log_defect
            self.best_episode = self.episode_count  # ★ 记录 episode
            mat_str = my_project_backend.dump_matrix_rust(self.current_pool_id)
            mat_list = string_to_matrix_fast(mat_str)
            if mat_list:
                self.best_vector = mat_list[0]
                self.best_basis = mat_list  # ★ 保存完整 basis
        return state_vec, log_b1, true_b1_GH_ratio, max_cos, min_cos, log_defect

    def step(self, action_idx):
        beta, pos = self.action_list[action_idx]

        old_log_b1 = self._cached_log_b1
        old_max_cos = self._cached_max_cos
        old_log_def = self._cached_log_def

        # ★ 直接用句柄约化，零序列化
        bkz_info = my_project_backend.reduce_rust(
            self.current_pool_id, "LOCAL_BKZ", beta, pos
        )

        # LLL 调度
        lll_frequency = 3
        do_lll = (
            self.current_step % lll_frequency == lll_frequency - 1
            or self.current_step >= self.max_steps - 1
        )
        if do_lll:
            lll_info = my_project_backend.reduce_rust(self.current_pool_id, "LLL", 2, 0)
            self.last_rust_info = lll_info
        else:
            self.last_rust_info = bkz_info

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # 终局
        if done:
            final_beta = min(self.dim, 40)
            my_project_backend.reduce_rust(
                self.current_pool_id, "LOCAL_BKZ", final_beta, 0
            )
            self.last_rust_info = my_project_backend.reduce_rust(
                self.current_pool_id, "LLL", 2, 0
            )

        # =============== 4. 计算新状态 ===============
        old_best_ratio = self.best_ratio
        old_ep_best_ratio = self.current_ep_best_ratio

        state, new_log_b1, new_ratio, new_max_cos, _, new_log_def = (
            self._get_state_and_update_best(self.last_rust_info)
        )
        # ★ 更新缓存，供下一次 step() 使用
        self._cached_log_b1 = new_log_b1
        self._cached_max_cos = new_max_cos
        self._cached_log_def = new_log_def

        # =============== 5. 奖励计算（分阶段）===============
        R_ratio = old_log_b1 - new_log_b1
        R_orth = old_max_cos - new_max_cos
        R_def = old_log_def - new_log_def

        # 阶段动态权重
        if self.best_ratio < 1.08:
            eff_ratio_w = 15.0
            eff_alpha = 8.0
            eff_gamma_r = 5.0
            eff_cost_w = 0.08
        elif self.best_ratio < 1.15:
            eff_ratio_w = 25.0
            eff_alpha = 3.0
            eff_gamma_r = 2.0
            eff_cost_w = 0.12
        else:
            eff_ratio_w = self.ratio_w
            eff_alpha = self.alpha
            eff_gamma_r = self.gamma_r
            eff_cost_w = self.cost_w

        reward = (
            eff_ratio_w * R_ratio
            + eff_alpha * R_orth
            + eff_gamma_r * R_def
            - eff_cost_w * (beta / max(self.betas))
        )

        # 里程碑奖励
        if new_ratio < old_best_ratio:
            reward += 5.0
        elif new_ratio < old_ep_best_ratio:
            reward += 2.0
            self.current_ep_best_ratio = new_ratio

        # 位置引导
        if R_ratio > 1e-3 and pos <= 2 and beta >= 20:
            reward += 0.1

        # 终局 bonus
        if done:
            if new_ratio < old_ep_best_ratio:
                reward += 3.0 * (old_ep_best_ratio - new_ratio) / self.ratio_scale
                self.current_ep_best_ratio = new_ratio
            if self.current_ep_best_ratio >= self.initial_ep_ratio:
                reward -= 2.0

        # =============== 6. 重复动作惩罚 ===============
        self.action_history.append(action_idx)
        if len(self.action_history) > self.repeat_window:
            self.action_history.pop(0)

        recent_count = self.action_history.count(action_idx)
        if recent_count >= 2:
            repeat_penalty = self.repeat_penalty_base * (recent_count - 1) ** 1.5
            reward -= repeat_penalty

        reward = float(np.clip(reward, -5.0, 50.0))

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
def env_worker(remote, parent_remote, matrix_path, max_dim, seed_idx=0):
    parent_remote.close()
    env = LatticeEnv(matrix_path, max_dim=max_dim, seed_idx=seed_idx)
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
                    {
                        "ratio": env.best_ratio,
                        "defect": env.best_defect,
                        "max_cos": env.best_max_cos,
                        "min_cos": env.best_min_cos,
                        "vector": env.best_vector,
                        "basis": env.best_basis,  # ★ 完整基矩阵
                        "seed_id": env.seed_id,  # ★ 种子编号
                        "seed_file": os.path.basename(env.matrix_path),  # ★ 文件名
                        "episode": env.best_episode,  # ★ 发现时的 episode
                        "total_episodes": env.episode_count,  # ★ 总 episode 数
                    }
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
    def __init__(self, num_envs, matrix_paths, max_dim=250):
        """
        matrix_paths: list[str] — 所有种子文件路径
        num_envs: int — 环境数（应 == len(matrix_paths)，一对一分配）
        """
        assert num_envs == len(matrix_paths), (
            f"num_envs({num_envs}) must equal to the length of matrix_path({len(matrix_paths)})."
        )
        self.num_envs = num_envs
        self.seed_files = matrix_paths

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [
            mp.Process(
                target=env_worker,
                args=(
                    work_remote,
                    remote,
                    matrix_paths[i],  # ★ 每个 env 分配一个固定种子
                    max_dim,
                    i,  # ★ seed_idx
                ),
            )
            for i, (work_remote, remote) in enumerate(
                zip(self.work_remotes, self.remotes)
            )
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
def save_seed_result(results_dir, dim, seed_info, is_update=False):
    """
    为每个种子保存独立结果文件
    文件名: dim55_seed0.txt, dim55_seed3.txt, ...
    """
    seed_id = seed_info["seed_id"]
    filepath = os.path.join(results_dir, f"dim{dim}_seed{seed_id}.txt")

    mode = "a" if is_update else "w"
    with open(filepath, mode) as f:
        if not is_update:
            f.write(f"{'=' * 60}\n")
            f.write(f" Lattice Reduction Results | Dim={dim} | Seed={seed_id}\n")
            f.write(f" File: {seed_info['seed_file']}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write("--- Initial State ---\n")
        else:
            f.write(f"\n--- New Best Found (Episode {seed_info['episode']}) ---\n")

        f.write(f"  Ratio (‖b₁‖/GH):  {seed_info['ratio']:.8f}\n")
        f.write(f"  Orthog. Defect:    {seed_info['defect']:.8f}\n")
        f.write(f"  Max Cosine:        {seed_info['max_cos']:.8f}\n")
        f.write(f"  Min Cosine:        {seed_info['min_cos']:.8f}\n")
        f.write(f"  b₁ = {seed_info['vector']}\n")


def save_final_summary(results_dir, dim, all_seed_infos):
    """
    训练结束后保存总结文件 + 每个达标种子的完整 basis
    """
    filepath = os.path.join(results_dir, f"dim{dim}_summary.txt")
    with open(filepath, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f" FINAL SUMMARY | Dim={dim}\n")
        f.write(f"{'=' * 60}\n\n")

        reached = [s for s in all_seed_infos if s["ratio"] < 1.05]
        f.write(
            f"Seeds reached goal (<1.05): {len(reached)} / {len(all_seed_infos)}\n\n"
        )

        # 按 ratio 排序
        sorted_infos = sorted(all_seed_infos, key=lambda x: x["ratio"])
        for info in sorted_infos:
            status = "✓" if info["ratio"] < 1.05 else "✗"
            f.write(
                f"  [{status}] Seed {info['seed_id']:2d} | "
                f"Ratio={info['ratio']:.6f} | "
                f"Found at Ep {info['episode']:4d} / {info['total_episodes']:4d} | "
                f"File: {info['seed_file']}\n"
            )

        # 输出达标种子的完整 basis
        f.write(f"\n{'=' * 60}\n")
        f.write("FULL BASIS MATRICES (ratio < 1.05)\n")
        f.write(f"{'=' * 60}\n")

        for info in sorted_infos:
            if info["ratio"] >= 1.05 or info["basis"] is None:
                continue
            f.write(f"\n--- Seed {info['seed_id']} (ratio={info['ratio']:.6f}) ---\n")
            f.write("[\n")
            for row in info["basis"]:
                f.write("  [" + " ".join(str(x) for x in row) + "]\n")
            f.write("]\n")


def train(
    vec_env,
    agent,
    num_envs,
    max_steps,
    dim,
    episodes=800,  # ★ 上限放大，实际由早停决定
    print_every=10,
    save_dir="results",
    patience=80,  # ★ 连续多少 ep 无改善就停
    goal_count_needed=6,  # ★ 可调的种子达标数
):
    history = {"reward": [], "loss": [], "ratio_min": []}

    seed_best_ratios = {}
    seed_best_infos = {}
    goal_threshold = 1.05
    # goal_count_needed 现在从参数传入

    # ★ patience 早停追踪
    best_global_ratio = float("inf")
    no_improve_count = 0

    states = vec_env.reset()

    # ---- 保存初始状态 ----
    initial_bests = vec_env.get_bests()  # 现在返回 list[dict]
    for info in initial_bests:
        sid = info["seed_id"]
        seed_best_ratios[sid] = info["ratio"]
        seed_best_infos[sid] = info
        save_seed_result(save_dir, dim, info, is_update=False)

    accumulated_ep_logs = []
    total_steps = 0

    for ep in range(1, episodes + 1):
        ep_rewards = np.zeros(num_envs)
        ep_ratios = []
        losses = []
        ep_action_logs = []

        epsilon = max(0.05, 0.3 * (1.0 - ep / episodes))

        for step in range(max_steps):
            total_steps += 1
            actions = agent.act_batch(states, is_training=True, epsilon=epsilon)
            next_states, rewards, batch_dones, infos = vec_env.step(actions)

            for i in range(num_envs):
                agent.remember(
                    states[i], actions[i], rewards[i], next_states[i], batch_dones[i]
                )
                ep_rewards[i] += rewards[i]
                ep_ratios.append(infos[i]["b1_GH_ratio"])

            if total_steps % 2 == 0:
                for _ in range(2):
                    loss, max_grad = agent.replay()
                    if loss != 0.0:
                        losses.append(loss)

            ep_action_logs.append(f"(p:{infos[0]['pos']:2d}, b:{infos[0]['beta']:2d})")
            states = next_states

        # ---- Episode 统计 ----
        avg_ep_reward = np.mean(ep_rewards)
        history["reward"].append(avg_ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        ep_min_ratio = min(ep_ratios) if ep_ratios else float("inf")
        history["ratio_min"].append(ep_min_ratio)

        # ---- ★ 按种子更新最优 ----
        bests = vec_env.get_bests()  # list[dict]
        updated_seeds = []

        for info in bests:
            sid = info["seed_id"]
            if info["ratio"] < seed_best_ratios.get(sid, float("inf")):
                seed_best_ratios[sid] = info["ratio"]
                seed_best_infos[sid] = info
                save_seed_result(save_dir, dim, info, is_update=True)
                updated_seeds.append(sid)

        if updated_seeds:
            # 保存模型
            model_path = os.path.join(save_dir, f"agent6UP_best_model_dim{dim}.pth")
            agent.save_checkpoint(model_path)
            for sid in updated_seeds:
                print(f"  ★ Seed {sid} new best: {seed_best_ratios[sid]:.6f} (ep {ep})")

        # ---- ★ 检查 6/9 停止条件 ----
        reached_count = sum(1 for r in seed_best_ratios.values() if r < goal_threshold)
        # ---- ★ 检查 6/9 停止条件 ----
        reached_count = sum(1 for r in seed_best_ratios.values() if r < goal_threshold)
        if reached_count >= goal_count_needed:
            print(
                f"\n🎉 [Dim {dim}] Goal reached! "
                f"{reached_count}/{len(seed_best_ratios)} seeds < {goal_threshold} "
                f"at episode {ep}"
            )
            break

        # ---- ★ patience 早停 ----
        current_global_best = min(seed_best_ratios.values())
        if current_global_best < best_global_ratio - 1e-6:
            best_global_ratio = current_global_best
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(
                f"\n⏹ [Dim {dim}] Patience exhausted: no improvement for "
                f"{patience} episodes. Stopping at ep {ep}. "
                f"Best global ratio: {best_global_ratio:.6f}, "
                f"Reached: {reached_count}/{goal_count_needed}"
            )
            break
        # ---- 打印 ----
        if ep % print_every == 0:
            print(f"\n[{'=' * 15} Ep {ep} (Dim {dim}) {'=' * 15}]")

            # 每个种子的进度一览
            print("  Seed Progress:")
            for sid in sorted(seed_best_ratios.keys()):
                r = seed_best_ratios[sid]
                status = "✓" if r < goal_threshold else " "
                print(f"    [{status}] Seed {sid:2d}: {r:.6f}")
            print(f"  Reached: {reached_count}/{goal_count_needed} needed")

            # Env 0 轨迹
            print("Env0 trajectory (last 10 actions):")
            for idx in range(max(0, len(ep_action_logs) - 10), len(ep_action_logs), 5):
                print("    " + " -> ".join(ep_action_logs[idx : idx + 5]))

            current_log = (
                f"Ep {ep:4d} | ε:{epsilon:.3f} | R:{avg_ep_reward:8.2f} "
                f"| Loss:{history['loss'][-1]:.4f} | "
                f"Reached:{reached_count}/{goal_count_needed}"
            )
            accumulated_ep_logs.append(current_log)
            print("\n  === History ===")
            for log in accumulated_ep_logs[-10:]:  # 只显示最近 10 条
                print(f"  {log}")
            print()

        agent.step_scheduler()

    # ---- ★ 训练结束：保存总结 ----
    final_bests = vec_env.get_bests()
    for info in final_bests:
        sid = info["seed_id"]
        if info["ratio"] < seed_best_ratios.get(sid, float("inf")):
            seed_best_infos[sid] = info

    save_final_summary(save_dir, dim, list(seed_best_infos.values()))
    print(f"\n[Dim {dim}] Summary saved to {save_dir}/dim{dim}_summary.txt")

    return history


def run_experiment(dim, dataset_dir, results_dir, num_envs=9):
    max_dim = ((dim + 7) // 8) * 8
    max_dim = max(max_dim, 16)

    import glob

    pattern = os.path.join(dataset_dir, f"svpchallengedim{dim}seed*.txt")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print(f"[Dim {dim}] No files found matching {pattern}")
        return

    # ★ env 数量 = 种子文件数量（一对一）
    envs_per_seed = 2
    # 每个种子文件重复2次：[seed0, seed0, seed1, seed1, ...]
    all_files_expanded = [f for f in all_files for _ in range(envs_per_seed)]
    num_envs = len(all_files_expanded)
    print(
        f"[Dim {dim}] {len(all_files)} seeds × {envs_per_seed} envs = {num_envs} envs, max_dim={max_dim}"
    )
    # ★ 创建维度专属结果目录
    dim_results_dir = os.path.join(results_dir, f"a6up_dim{dim}")
    os.makedirs(dim_results_dir, exist_ok=True)

    vec_env = SubprocVecEnv(num_envs, all_files_expanded, max_dim=max_dim)

    # 临时 env 获取动作空间大小
    temp_env = LatticeEnv(all_files[0], max_dim=max_dim, seed_idx=0)

    agent = DQNAgent(
        max_dim=max_dim,
        state_dim=temp_env.state_dim,
        action_dim=temp_env.num_actions,
    )

    model_path = os.path.join(dim_results_dir, f"agent6UP_best_model_dim{dim}.pth")
    agent.load_checkpoint(model_path)

    history = train(
        vec_env,
        agent,
        num_envs,
        max_steps=temp_env.max_steps,
        dim=dim,
        episodes=800,  # ★ 上限兜底
        print_every=10,
        save_dir=dim_results_dir,
        patience=80,  # ★ 无改善早停
        goal_count_needed=6,  # ★ 种子达标数
    )

    # ---- 画图 ----
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["reward"], label="Average Reward")
    plt.title(f"Dim {dim} - Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["ratio_min"], label="Ep Min Ratio", color="orange")
    plt.axhline(y=1.05, color="r", linestyle="--", label="Goal (1.05)")
    plt.title(f"Dim {dim} - Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(dim_results_dir, f"training_dim{dim}.png"))
    plt.close()
    vec_env.close()
    print(f"[Dim {dim}] Complete!\n{'=' * 60}\n")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    DIMS_TO_RUN = [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    for dim in DIMS_TO_RUN:
        run_experiment(dim, DATASET_DIR, RESULTS_DIR)
