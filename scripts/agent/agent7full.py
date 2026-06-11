import my_project_backend

import os, sys, math, time, random

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
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
    matrix = []
    with open(filepath, "r") as f:
        content = f.read().replace("[", "").replace("]", "")
        for line in content.strip().split("\n"):
            if line.strip():
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

        if max_dim <= 32:
            dilations = [1, 2, 4]
        elif max_dim <= 64:
            dilations = [1, 3, 9]
        elif max_dim <= 128:
            dilations = [1, 4, 16]
        else:
            dilations = [1, 6, 24]

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

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.01),
        )

        self.se = SE_Block(64, reduction=4)

        scalar_input_dim = max_dim + 5
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.LeakyReLU(0.01),
        )

        self.grid_size = min(8, max_dim - 1)
        cnn_flat_size = 64 * self.grid_size * self.grid_size

        self.fusion = nn.Sequential(
            nn.Linear(cnn_flat_size + 64, 256),
            nn.LeakyReLU(0.01),
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
        gs_and_scalars = x[:, tokens_flat_size:]

        cos_matrix = tokens_flat.view(batch_size, 1, self.max_dim - 1, self.token_dim)

        col_feat = self.col_conv(cos_matrix)
        row_feat = self.row_conv(cos_matrix)
        concat_feat = torch.cat([col_feat, row_feat], dim=1)
        fused_matrix = self.fuse_conv(concat_feat)
        fused_matrix = self.se(fused_matrix)

        pool_max = F.adaptive_max_pool2d(fused_matrix, (self.grid_size, self.grid_size))
        pool_avg = F.adaptive_avg_pool2d(fused_matrix, (self.grid_size, self.grid_size))
        grid_out = 0.5 * pool_max + 0.5 * pool_avg
        cnn_out = grid_out.view(batch_size, -1)

        scalar_out = self.scalar_mlp(gs_and_scalars)

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
# Replay Buffer
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
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

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
    PER_e = 1e-5
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment = 0.001

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


# ------------------------------
# Agent
# ------------------------------
class DQNAgent:
    def __init__(self, max_dim, action_dim, batch_size=128):
        self.device = device
        self.batch_size = batch_size
        self.q_net = AxialCNN_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net = AxialCNN_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=6e-5,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=500,
            eta_min=1e-6,
        )
        self.memory = PrioritizedReplayBuffer(50000)
        self.gamma = 0.99
        self.tau = 0.0025

    def save_checkpoint(self, model_path):
        torch.save(self.q_net.state_dict(), model_path)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save_training_state(self, path, extra=None):
        """保存可续训状态；不保存 replay buffer，避免 pth 过大"""
        payload = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "extra": extra or {},
        }

        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def load_training_state(self, path):
        """读取可续训状态；返回 extra 训练信息"""
        if not os.path.exists(path):
            return {}

        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        self.q_net.load_state_dict(payload["q_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])

        return payload.get("extra", {})

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
        td_errors = (curr_q - target_q).detach().abs().cpu().numpy().flatten()
        for i, idx in enumerate(tree_idxs):
            self.memory.update(idx, td_errors[i])

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
# Environment（保留随机种子分配）
# ------------------------------
class LatticeEnv:
    def __init__(self, matrix_paths, max_dim=250):
        if isinstance(matrix_paths, str):
            matrix_paths = [matrix_paths]
        self.matrix_paths = matrix_paths

        self.action_history = []
        self.repeat_window = 8
        self.repeat_penalty_base = 0.3

        self._load_lattice(self.matrix_paths[0])

        # 动作空间
        beta_max = min(int(0.8 * self.dim), 50)
        beta_min = max(8, int(0.15 * self.dim))
        raw = np.geomspace(beta_min, beta_max, 7)
        self.betas = sorted(set(max(2, int(round(x))) for x in raw))
        self.action_list = []
        for b in self.betas:
            if b > self.dim:
                continue
            pos_step = max(1, b // 2)
            for p in range(0, self.dim - b + 1, pos_step):
                self.action_list.append((b, p))
            if self.dim - b >= 0 and (b, 0) not in self.action_list:
                self.action_list.insert(0, (b, 0))
        self.num_actions = len(self.action_list)

        self.max_steps = max(4 * self.dim, 4 * self.num_actions)
        self.state_dim = (self.max_dim - 1) * self.max_dim + self.max_dim + 5

        # 全局最优
        self.best_ratio = float("inf")
        self.best_vector = None
        self.best_basis = None
        self.best_defect = None
        self.best_max_cos = None
        self.best_min_cos = None
        self.best_seed_file = None
        self.best_episode = 0
        self.episode_count = 0

        # ★ per-seed 追踪：ratio + 详细信息
        self.seed_best_ratios = {}
        self.seed_best_infos = {}
        self.seed_data = {}
        self._preload_all_seeds()

        # 缓存
        self._cached_log_b1 = 0.0
        self._cached_max_cos = 0.0
        self._cached_log_def = 0.0

    def _load_lattice(self, filepath):
        self.initial_matrix_list = parse_challenge_file(filepath)
        self.dim = len(self.initial_matrix_list)
        self.max_dim = self.dim

        raw_matrix_str = matrix_to_string(self.initial_matrix_list)
        self.max_steps = None

        self.ratio_w = 30.0
        self.alpha = 2.0
        self.gamma_r = 1.0
        self.cost_w = 0.15

        self.initial_pool_id = my_project_backend.create_matrix_lll_rust(raw_matrix_str)

        init_eval = my_project_backend.evaluate_matrix_rust(self.initial_pool_id)
        self.initial_gs_logs = np.array(init_eval["gs_log_norms"], dtype=np.float32)
        self.log_vol = np.sum(self.initial_gs_logs)
        self.log_GH = (self.log_vol / self.dim) + 0.5 * math.log(
            self.dim / (2 * math.pi * math.e)
        )

        init_info = my_project_backend.reduce_rust(
            self.initial_pool_id,
            "LLL",
            2,
            0,
        )

        initial_log_defect = float(init_info["log_prod"] - self.log_vol)
        initial_log_ratio = float(self.initial_gs_logs[0] - self.log_GH)
        self.defect_scale = max(abs(initial_log_defect), 1.0)
        self.ratio_scale = max(abs(initial_log_ratio), 1.0)

        self.current_filepath = filepath

    def _preload_all_seeds(self):
        """训练前主动为所有种子做 LLL 约化，存储 pool_id 和初始信息"""
        import re

        for filepath in self.matrix_paths:
            match = re.search(r"seed(\d+)", os.path.basename(filepath))
            sid = int(match.group(1)) if match else 0

            mat_list = parse_challenge_file(filepath)
            raw_str = matrix_to_string(mat_list)
            pool_id = my_project_backend.create_matrix_lll_rust(raw_str)

            eval_info = my_project_backend.evaluate_matrix_rust(pool_id)
            gs_logs = np.array(eval_info["gs_log_norms"], dtype=np.float32)
            log_vol = float(np.sum(gs_logs))
            log_GH = (log_vol / self.dim) + 0.5 * math.log(
                self.dim / (2 * math.pi * math.e)
            )

            lll_info = my_project_backend.reduce_rust(pool_id, "LLL", 2, 0)
            initial_log_defect = float(lll_info["log_prod"] - log_vol)
            initial_log_ratio = float(gs_logs[0] - log_GH)
            defect_scale = max(abs(initial_log_defect), 1.0)
            ratio_scale = max(abs(initial_log_ratio), 1.0)

            self.seed_data[sid] = {
                "pool_id": pool_id,
                "log_vol": log_vol,
                "log_GH": log_GH,
                "defect_scale": defect_scale,
                "ratio_scale": ratio_scale,
                "initial_gs_logs": gs_logs,
                "filepath": filepath,
            }

            # ★ 计算初始 ratio 并填充 seed_best_ratios / seed_best_infos
            true_ratio = float(math.exp(gs_logs[0] - log_GH))

            C = np.array(lll_info["cos_matrix"], dtype=np.float32)
            lower = C[np.tril_indices(self.dim, -1)]
            max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
            min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))

            mat_str = my_project_backend.dump_matrix_rust(pool_id)
            mat_list_reduced = string_to_matrix_fast(mat_str)

            self.seed_best_ratios[sid] = true_ratio
            self.seed_best_infos[sid] = {
                "ratio": true_ratio,
                "defect": initial_log_defect,
                "max_cos": max_cos,
                "min_cos": min_cos,
                "vector": mat_list_reduced[0] if mat_list_reduced else None,
                "basis": mat_list_reduced,
                "seed_id": sid,
                "seed_file": os.path.basename(filepath),
                "episode": 0,
            }

            # 更新全局最优
            if true_ratio < self.best_ratio:
                self.best_ratio = true_ratio
                self.best_max_cos = max_cos
                self.best_min_cos = min_cos
                self.best_defect = initial_log_defect
                self.best_vector = mat_list_reduced[0] if mat_list_reduced else None
                self.best_basis = mat_list_reduced
                self.best_seed_file = filepath
                self.best_episode = 0

    def reset(self):
        # 每次 reset 随机选种子
        sid_list = list(self.seed_data.keys())
        chosen_sid = random.choice(sid_list)
        sd = self.seed_data[chosen_sid]

        self.current_seed_id = chosen_sid
        self.current_filepath = sd["filepath"]
        self.initial_pool_id = sd["pool_id"]
        self.log_vol = sd["log_vol"]
        self.log_GH = sd["log_GH"]
        self.defect_scale = sd["defect_scale"]
        self.ratio_scale = sd["ratio_scale"]
        self.initial_gs_logs = sd["initial_gs_logs"]

        self.current_step = 0
        self.action_history = []
        self.episode_count += 1

        if hasattr(self, "current_pool_id") and self.current_pool_id >= 0:
            try:
                my_project_backend.free_matrix_rust(self.current_pool_id)
            except Exception:
                pass

        self.current_pool_id = my_project_backend.clone_matrix_rust(
            self.initial_pool_id
        )

        if self.current_pool_id < 0:
            raise RuntimeError(
                f"clone_matrix_rust failed: "
                f"initial_pool_id={self.initial_pool_id}, "
                f"seed={chosen_sid}, "
                f"file={self.current_filepath}"
            )

        init_info = my_project_backend.reduce_rust(
            self.current_pool_id,
            "LLL",
            2,
            0,
        )

        state, log_b1, ratio, max_cos, min_cos, log_def = (
            self._get_state_and_update_best(init_info)
        )

        self._cached_log_b1 = log_b1
        self._cached_max_cos = max_cos
        self._cached_log_def = log_def

        self.initial_ep_ratio = ratio
        self.current_ep_best_ratio = ratio

        self.last_rust_info = init_info

        state = np.asarray(state, dtype=np.float32)

        expected_dim = self.state_dim
        if state.ndim != 1 or state.shape[0] != expected_dim:
            raise RuntimeError(
                f"bad reset state: shape={state.shape}, "
                f"dtype={state.dtype}, expected=({expected_dim},)"
            )

        return state

    def _get_state_and_update_best(self, rust_info):
        C = np.array(rust_info["cos_matrix"], dtype=np.float32)
        lower = C[np.tril_indices(self.dim, -1)].astype(np.float32)
        max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        C = C + C.T

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

        # ★ per-seed 最优检查（先于全局，因为需要 dump 一次即可）
        sid = getattr(self, "current_seed_id", 0)
        is_seed_best = true_b1_GH_ratio < self.seed_best_ratios.get(sid, float("inf"))
        is_global_best = true_b1_GH_ratio < self.best_ratio

        if is_seed_best:
            self.seed_best_ratios[sid] = true_b1_GH_ratio
            mat_str = my_project_backend.dump_matrix_rust(self.current_pool_id)
            mat_list = string_to_matrix_fast(mat_str)
            if mat_list:
                self.seed_best_infos[sid] = {
                    "ratio": true_b1_GH_ratio,
                    "defect": log_defect,
                    "max_cos": max_cos,
                    "min_cos": min_cos,
                    "vector": mat_list[0],
                    "basis": mat_list,
                    "seed_id": sid,
                    "seed_file": os.path.basename(self.current_filepath),
                    "episode": self.episode_count,
                }
                if is_global_best:
                    self.best_vector = mat_list[0]
                    self.best_basis = mat_list

        if is_global_best:
            self.best_ratio = true_b1_GH_ratio
            self.best_max_cos = max_cos
            self.best_min_cos = min_cos
            self.best_defect = log_defect
            self.best_episode = self.episode_count
            self.best_seed_file = self.current_filepath

        return state_vec, log_b1, true_b1_GH_ratio, max_cos, min_cos, log_defect

    def step(self, action_idx):
        beta, pos = self.action_list[action_idx]

        old_log_b1 = self._cached_log_b1
        old_max_cos = self._cached_max_cos
        old_log_def = self._cached_log_def

        bkz_info = my_project_backend.reduce_rust(
            self.current_pool_id, "LOCAL_BKZ", beta, pos
        )

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

        if done:
            final_beta = min(self.dim, 40)
            my_project_backend.reduce_rust(
                self.current_pool_id, "LOCAL_BKZ", final_beta, 0
            )
            self.last_rust_info = my_project_backend.reduce_rust(
                self.current_pool_id, "LLL", 2, 0
            )

        old_best_ratio = self.best_ratio
        old_ep_best_ratio = self.current_ep_best_ratio

        state, new_log_b1, new_ratio, new_max_cos, _, new_log_def = (
            self._get_state_and_update_best(self.last_rust_info)
        )
        self._cached_log_b1 = new_log_b1
        self._cached_max_cos = new_max_cos
        self._cached_log_def = new_log_def

        R_ratio = old_log_b1 - new_log_b1
        R_orth = old_max_cos - new_max_cos
        R_def = old_log_def - new_log_def

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

        if new_ratio < old_best_ratio:
            reward += 5.0
        elif new_ratio < old_ep_best_ratio:
            reward += 2.0
            self.current_ep_best_ratio = new_ratio

        if R_ratio > 1e-3 and pos <= 2 and beta >= 20:
            reward += 0.1

        if done:
            if new_ratio < old_ep_best_ratio:
                reward += 3.0 * (old_ep_best_ratio - new_ratio) / self.ratio_scale
                self.current_ep_best_ratio = new_ratio
            if self.current_ep_best_ratio >= self.initial_ep_ratio:
                reward -= 2.0

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
def env_worker(remote, parent_remote, matrix_paths, max_dim):
    parent_remote.close()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    env = LatticeEnv(matrix_paths, max_dim)
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
                        "basis": env.best_basis,
                        "seed_file": os.path.basename(env.best_seed_file)
                        if env.best_seed_file
                        else "unknown",
                        "episode": env.best_episode,
                        "seed_ratios": dict(env.seed_best_ratios),
                        "seed_infos": dict(env.seed_best_infos),  # ★ 每个种子的详细信息
                    }
                )
            elif cmd == "close":
                remote.close()
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


class SubprocVecEnv:
    def __init__(self, num_envs, matrix_paths, max_dim=250):
        """
        matrix_paths: list[str] — 所有种子文件路径（每个 env 都拿到完整列表，随机选）
        num_envs: int — 环境数（与种子数无关）
        """
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [
            mp.Process(
                target=env_worker,
                args=(work_remote, remote, matrix_paths, max_dim),
            )
            for work_remote, remote in zip(self.work_remotes, self.remotes)
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

    def send_one(self, env_id, action):
        self.remotes[env_id].send(("step", action))

    def recv_one(self, env_id):
        return self.remotes[env_id].recv()

    def poll_ready(self, env_ids, timeout=0.002):
        return [i for i in env_ids if self.remotes[i].poll(timeout=0)]


# ------------------------------
# 结果保存函数
# ------------------------------
def save_seed_result(results_dir, dim, seed_info, is_update=False):
    """为每个种子保存独立结果文件"""
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
        if seed_info.get("defect") is not None:
            f.write(f"  Orthog. Defect:    {seed_info['defect']:.8f}\n")
        if seed_info.get("max_cos") is not None:
            f.write(f"  Max Cosine:        {seed_info['max_cos']:.8f}\n")
        if seed_info.get("min_cos") is not None:
            f.write(f"  Min Cosine:        {seed_info['min_cos']:.8f}\n")
        f.write(f"  b₁ = {seed_info.get('vector')}\n")


def save_final_summary(results_dir, dim, all_seed_infos, goal_threshold=1.05):
    """训练结束后保存总结文件 + 每个达标种子的完整 basis"""
    filepath = os.path.join(results_dir, f"dim{dim}_summary.txt")
    with open(filepath, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f" FINAL SUMMARY | Dim={dim}\n")
        f.write(f"{'=' * 60}\n\n")

        reached = [s for s in all_seed_infos if s["ratio"] < goal_threshold]
        f.write(
            f"Seeds reached goal (<{goal_threshold}): "
            f"{len(reached)} / {len(all_seed_infos)}\n\n"
        )

        sorted_infos = sorted(all_seed_infos, key=lambda x: x["ratio"])
        for info in sorted_infos:
            status = "✓" if info["ratio"] < goal_threshold else "✗"
            ep_str = info.get("episode", "?")
            f.write(
                f"  [{status}] Seed {info['seed_id']:2d} | "
                f"Ratio={info['ratio']:.6f} | "
                f"Found at Ep {ep_str} | "
                f"File: {info['seed_file']}\n"
            )

        f.write(f"\n{'=' * 60}\n")
        f.write(f"FULL BASIS MATRICES (ratio < {goal_threshold})\n")
        f.write(f"{'=' * 60}\n")

        for info in sorted_infos:
            if info["ratio"] >= goal_threshold or info.get("basis") is None:
                continue
            f.write(f"\n--- Seed {info['seed_id']} (ratio={info['ratio']:.6f}) ---\n")
            f.write("[\n")
            for row in info["basis"]:
                f.write("  [" + " ".join(str(x) for x in row) + "]\n")
            f.write("]\n")


# ------------------------------
# Train
# ------------------------------
def train(
    vec_env,
    agent,
    num_envs,
    max_steps,
    dim,
    episodes=800,
    print_every=10,
    save_dir="results",
    patience=70,
    goal_count_needed=6,
    goal_threshold=1.05,
    resume_extra=None,
    run_tag=None,
):
    history = {"reward": [], "loss": [], "ratio_min": []}

    # ★ 主进程侧的 per-seed 聚合追踪
    global_seed_best_ratios = {}  # sid -> float
    global_seed_best_infos = {}  # sid -> dict（详细信息）

    best_global_ratio = float("inf")
    no_improve_count = 0

    resume_extra = resume_extra or {}

    if resume_extra:
        history = resume_extra.get("history", history)
        global_seed_best_ratios.update(resume_extra.get("global_seed_best_ratios", {}))
        global_seed_best_infos.update(resume_extra.get("global_seed_best_infos", {}))
        best_global_ratio = float(
            resume_extra.get("best_global_ratio", best_global_ratio)
        )
        no_improve_count = int(resume_extra.get("no_improve_count", 0))

    start_ep = int(resume_extra.get("episode", 0)) + 1
    accumulated_ep_logs = resume_extra.get("accumulated_ep_logs", [])
    total_steps = int(resume_extra.get("total_steps", 0))

    states = vec_env.reset()

    # ---- 初始状态：从所有 env 收集初始 per-seed 信息并保存 ----
    initial_bests = vec_env.get_bests()
    for env_info in initial_bests:
        for sid, sinfo in env_info["seed_infos"].items():
            if (
                sid not in global_seed_best_infos
                or sinfo["ratio"] < global_seed_best_infos[sid]["ratio"]
            ):
                global_seed_best_ratios[sid] = sinfo["ratio"]
                global_seed_best_infos[sid] = sinfo
                save_seed_result(save_dir, dim, sinfo, is_update=False)

    for ep in range(start_ep, episodes + 1):
        ep_rewards = np.zeros(num_envs)
        ep_ratios = []
        losses = []
        ep_action_logs = [[] for _ in range(num_envs)]
        best_ep_ratio = float("inf")
        best_ep_eid = None
        epsilon = max(0.05, 0.3 * (1.0 - ep / episodes))

        env_steps = np.zeros(num_envs, dtype=int)
        prev_states = states.copy()

        actions = agent.act_batch(states, is_training=True, epsilon=epsilon)
        prev_actions = actions.copy()
        pending = set(range(num_envs))

        for eid in range(num_envs):
            vec_env.send_one(eid, actions[eid])

        collected = 0

        while True:
            ready = vec_env.poll_ready(pending)
            if not ready:
                time.sleep(0.001)
                continue

            for eid in ready:
                obs_i, rew_i, done_i, info_i = vec_env.recv_one(eid)
                pending.discard(eid)

                agent.remember(
                    prev_states[eid], prev_actions[eid], rew_i, obs_i, done_i
                )
                states[eid] = obs_i
                ep_rewards[eid] += rew_i
                env_steps[eid] += 1
                ep_ratios.append(info_i["b1_GH_ratio"])
                collected += 1

                ep_action_logs[eid].append(
                    f"a:{prev_actions[eid]:3d} "
                    f"(p:{info_i['pos']:2d}, b:{info_i['beta']:2d}, "
                    f"g:{info_i['b1_GH_ratio']:.8f})"
                )
                if info_i["b1_GH_ratio"] < best_ep_ratio:
                    best_ep_ratio = info_i["b1_GH_ratio"]
                    best_ep_eid = eid
            if collected % 4 == 0 and collected > 0:
                loss, max_grad = agent.replay()
                if loss != 0.0:
                    losses.append(loss)
                total_steps += 1

            to_send = [eid for eid in ready if env_steps[eid] < max_steps]
            if to_send:
                batch_s = np.stack([states[eid] for eid in to_send])
                batch_a = agent.act_batch(batch_s, is_training=True, epsilon=epsilon)
                for idx, eid in enumerate(to_send):
                    vec_env.send_one(eid, batch_a[idx])
                    prev_states[eid] = states[eid].copy()
                    prev_actions[eid] = batch_a[idx]
                    pending.add(eid)

            if min(env_steps) >= max_steps:
                break

        # ---- Episode 统计 ----
        avg_ep_reward = np.mean(ep_rewards)
        history["reward"].append(avg_ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        ep_min_ratio = min(ep_ratios) if ep_ratios else float("inf")
        history["ratio_min"].append(ep_min_ratio)

        # ---- ★ 合并所有 env 的 per-seed 信息 ----
        bests = vec_env.get_bests()
        updated_seeds = []

        for env_info in bests:
            for sid, sinfo in env_info["seed_infos"].items():
                old_ratio = global_seed_best_ratios.get(sid, float("inf"))
                if sinfo["ratio"] < old_ratio:
                    is_first_time = sid not in global_seed_best_infos
                    global_seed_best_ratios[sid] = sinfo["ratio"]
                    global_seed_best_infos[sid] = sinfo
                    save_seed_result(save_dir, dim, sinfo, is_update=not is_first_time)
                    updated_seeds.append(sid)
        # ---- 全局最优更新 ----
        current_global_best = (
            min(global_seed_best_ratios.values())
            if global_seed_best_ratios
            else float("inf")
        )

        if current_global_best < best_global_ratio - 1e-6:
            best_global_ratio = current_global_best
            no_improve_count = 0

            agent.save_checkpoint(
                os.path.join(save_dir, f"{run_tag}_best_model_dim{dim}.pth")
            )
            # 找到产出该 ratio 的种子
            best_sid = min(global_seed_best_ratios, key=global_seed_best_ratios.get)
            bi = global_seed_best_infos[best_sid]
            print(
                f"  ★ New global best {best_global_ratio:.11f} "
                f"from seed {best_sid} [{bi['seed_file']}] (ep {ep})"
            )
        else:
            no_improve_count += 1

        if updated_seeds:
            for sid in updated_seeds:
                print(
                    f"  ★ Seed {sid} new best: {global_seed_best_ratios[sid]:.11f} (ep {ep})"
                )

        # ---- 停止条件 ----
        reached_count = sum(
            1 for r in global_seed_best_ratios.values() if r < goal_threshold
        )

        if reached_count >= goal_count_needed:
            print(
                f"\n[Dim {dim}] Goal reached! "
                f"{reached_count}/{len(global_seed_best_ratios)} seeds < {goal_threshold} "
                f"at episode {ep}"
            )
            break

        # if no_improve_count >= patience:
        #    print(
        #        f"\n⏹ [Dim {dim}] No improvement for {patience} ep, stop at ep {ep}. "
        #        f"Best: {best_global_ratio:.8f}, Reached: {reached_count}/{goal_count_needed}"
        #    )
        #    break

        # ---- 打印 ----
        if ep % print_every == 0:
            print(f"\n[{'=' * 15} Ep {ep} (Dim {dim}) {'=' * 15}]")
            print("  Seed Progress:")
            for sid in sorted(global_seed_best_ratios.keys()):
                r = global_seed_best_ratios[sid]
                status = "✓" if r < goal_threshold else " "
                print(f"    [{status}] Seed {sid:2d}: {r:.6f}")
            print(f"  Reached: {reached_count}/{goal_count_needed}")

            print("  Best trajectory actions:")
            if best_ep_eid is not None:
                best_logs = ep_action_logs[best_ep_eid]
                print(
                    f"    Env{best_ep_eid} | best ratio in episode: {best_ep_ratio:.8f}"
                )
                for idx in range(0, len(best_logs), 5):
                    print("      " + " -> ".join(best_logs[idx : idx + 5]))
                del best_logs
            else:
                print("No actions recorded.")

            ep_action_logs.clear()

            current_log = (
                f"Ep {ep:4d} | ε:{epsilon:.3f} | R:{avg_ep_reward:8.2f} "
                f"| Loss:{history['loss'][-1]:.4f} | Best:{best_global_ratio:.6f} "
                f"| Seeds:{reached_count}/{goal_count_needed}"
            )
            accumulated_ep_logs.append(current_log)
            print("\n  === History ===")
            for log in accumulated_ep_logs[-10:]:
                print(f"  {log}")
            print()

        agent.step_scheduler()

        agent.save_training_state(
            os.path.join(save_dir, f"{run_tag}_latest_resume_dim{dim}.pth"),
            extra={
                "episode": ep,
                "best_global_ratio": best_global_ratio,
                "global_seed_best_ratios": global_seed_best_ratios,
                "global_seed_best_infos": global_seed_best_infos,
                "history": history,
                "no_improve_count": no_improve_count,
                "accumulated_ep_logs": accumulated_ep_logs[-10:],
                "total_steps": total_steps,
            },
        )

    # ---- ★ 终局：最终收集一次并保存 summary ----
    final_bests = vec_env.get_bests()
    for env_info in final_bests:
        for sid, sinfo in env_info["seed_infos"].items():
            old_ratio = global_seed_best_ratios.get(sid, float("inf"))
            if sinfo["ratio"] < old_ratio:
                global_seed_best_ratios[sid] = sinfo["ratio"]
                global_seed_best_infos[sid] = sinfo

    save_final_summary(
        save_dir, dim, list(global_seed_best_infos.values()), goal_threshold
    )
    print(f"\n[Dim {dim}] Summary saved to {save_dir}/dim{dim}_summary.txt")

    return history


# ------------------------------
# run_experiment
# ------------------------------
def run_experiment(dim, dataset_dir, results_dir, num_envs=12):
    max_dim = dim

    import glob

    pattern = os.path.join(dataset_dir, f"svpchallengedim{dim}seed*.txt")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print(f"[Dim {dim}] No files found matching {pattern}")
        return

    print(
        f"[Dim {dim}] {len(all_files)} seeds, {num_envs} envs (random assignment), max_dim={max_dim}"
    )

    version_name = "a7"
    version_results_dir = os.path.join(results_dir, version_name)

    dim_results_dir = os.path.join(version_results_dir, f"{version_name}_dim{dim}")
    os.makedirs(dim_results_dir, exist_ok=True)

    # ★ 所有 env 都拿到完整种子列表，每次 reset 随机选
    vec_env = SubprocVecEnv(num_envs, all_files, max_dim=max_dim)
    temp_env = LatticeEnv(all_files, max_dim=max_dim)

    agent = DQNAgent(
        max_dim=max_dim,
        action_dim=temp_env.num_actions,
    )

    resume_path = os.path.join(
        dim_results_dir, f"{version_name}_latest_resume_dim{dim}.pth"
    )
    model_path = os.path.join(
        dim_results_dir, f"{version_name}_best_model_dim{dim}.pth"
    )

    resume_extra = agent.load_training_state(resume_path)

    if resume_extra:
        print(f"[Dim {dim}] Loaded resume checkpoint: {resume_path}")
    else:
        agent.load_checkpoint(model_path)
        print(f"[Dim {dim}] Loaded best model checkpoint if exists: {model_path}")

    history = train(
        vec_env,
        agent,
        num_envs,
        max_steps=temp_env.max_steps,
        dim=dim,
        episodes=800,
        print_every=10,
        save_dir=dim_results_dir,
        patience=80,
        goal_count_needed=6,
        goal_threshold=0.85,
        resume_extra=resume_extra,
        run_tag=version_name,
    )

    # ---- 画图 ----
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["reward"], label="Avg Reward")
    plt.title(f"Dim {dim} - Reward")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["ratio_min"], label="Ep Min Ratio", color="orange")
    plt.axhline(y=1.05, color="r", linestyle="--", label="Goal")
    plt.title(f"Dim {dim} - Ratio")
    plt.grid(True)
    plt.legend()
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

    DIMS_TO_RUN = [
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
    ]
    for dim in DIMS_TO_RUN:
        run_experiment(dim, DATASET_DIR, RESULTS_DIR)
