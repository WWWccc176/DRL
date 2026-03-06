import my_project_backend
import os, sys, math, time, random, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mpmath
from mpmath import mp

import matplotlib.pyplot as plt
from collections import deque


# ------------------------------
# Seed (optional but recommended)
# ------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def matrix_to_string(basis):
    # 将二维列表转换为 C++ 期望的字符串格式: [[1 2][3 4]]
    lines = []
    for row in basis:
        # 将数字转为字符串并用空格连接
        row_str = " ".join(str(x) for x in row)
        lines.append(f"[{row_str}]")
    # 用换行符连接所有行，并包裹在最外层的 []
    return "[" + "\n".join(lines) + "]"


# ------------------------------
# Device
# ------------------------------
def get_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print("✅ Using device:", device)

# ------------------------------
# Robust metrics (for logging, slow)
# ------------------------------
mp.dps = 100


def compute_metrics_robust(basis, prec=200):
    """
    高精度 Gram-Schmidt，返回 (ortho_defect, gh_ratio)
    gh_ratio = ||b1|| / GH(L)
    """
    mp.dps = prec

    B = basis
    n = len(B)

    Bmp = [[mp.mpf(int(x)) for x in row] for row in B]

    bstar = [[] for _ in range(n)]
    bstar_norm2 = [mp.mpf("0")] * n

    for i in range(n):
        bstar[i] = Bmp[i].copy()
        for j in range(i):
            if bstar_norm2[j] == 0:
                continue
            dot_ij = mp.fdot(Bmp[i], bstar[j])
            mu_ij = dot_ij / bstar_norm2[j]
            for k in range(n):
                bstar[i][k] -= mu_ij * bstar[j][k]
        bstar_norm2[i] = mp.fdot(bstar[i], bstar[i])

    log_det = mp.mpf("0")
    for i in range(n):
        if bstar_norm2[i] <= 0:
            return float("inf"), float("inf")
        log_det += mp.log(bstar_norm2[i]) / 2

    log_gh = (log_det / n) + (mp.log(n) - mp.log(2 * mp.pi * mp.e)) / 2

    b1norm2 = mp.fdot(Bmp[0], Bmp[0])
    log_b1 = mp.log(b1norm2) / 2
    gh_ratio = mp.e ** (log_b1 - log_gh)

    sum_log_orig_norms = mp.mpf("0")
    for i in range(n):
        orig_norm2 = mp.fdot(Bmp[i], Bmp[i])
        sum_log_orig_norms += mp.log(orig_norm2) / 2

    ortho_defect = mp.e ** (sum_log_orig_norms - log_det)
    return float(ortho_defect), float(gh_ratio)


compute_metrics = compute_metrics_robust


def extract_features_from_rust(rust_info, dim):
    """
    直接从 Rust 返回的字典中提取 State 需要的特征
    rust_info keys: 'matrix_str', 'log_prod', 'min_norm', 'cos_matrix'
    """
    # 1. 处理 Cosine Matrix
    C = rust_info["cos_matrix"]

    # ⚠️ 逻辑修正：
    # C++ 代码生成的是下三角矩阵 (i > j 有值，其余为 0)
    # 所以必须使用 tril_indices (Lower Triangle) 提取特征
    # k=-1 表示不包含主对角线
    il = np.tril_indices(dim, -1)
    lower = C[il].astype(np.float32)

    # ⚠️ 拼写修复：
    # 原代码: if RESULTS_DIRpper.size ... (这是乱码)
    # 修正为: if lower.size ...
    max_cos = float(np.max(lower)) if lower.size > 0 else 0.0

    # 截断防止数值误差导致 arccos 报错
    max_cos = float(np.clip(max_cos, 0.0, 1.0))
    theta_min = float(np.arccos(max_cos))

    return lower, theta_min


# ------------------------------
# Parse SVP challenge
# ------------------------------
def parse_challenge_file(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    tokens = re.findall(r"[-+]?\d+", text)
    numbers = [int(t) for t in tokens]

    if len(numbers) < 4:
        raise ValueError(f"parse_challenge_file: not enough numbers in {filepath}")

    total = len(numbers)

    if total >= 2 and int(math.isqrt(total - 2)) ** 2 == (total - 2):
        dim = int(math.isqrt(total - 2))
        matrix_data = numbers[2:]
    elif total >= 1 and int(math.isqrt(total - 1)) ** 2 == (total - 1):
        dim = int(math.isqrt(total - 1))
        matrix_data = numbers[1:]
    else:
        dim = int(math.isqrt(total))
        if dim * dim != total:
            raise ValueError(
                f"parse_challenge_file: cannot infer square matrix, total={total}"
            )
        matrix_data = numbers

    if len(matrix_data) != dim * dim:
        raise ValueError(
            f"parse_challenge_file: size mismatch, dim={dim}, got={len(matrix_data)}"
        )

    mat = np.array(matrix_data, dtype=object).reshape(dim, dim)
    return mat.tolist()


# ------------------------------
# Run reducers (whole matrix or block matrix)
# ------------------------------
def run_reduction_wrapper(matrix_data, beta, step_id, debug=False):
    # 1. 序列化矩阵
    mat_str = matrix_to_string(matrix_data)

    # 2. 确定方法
    method = "LLL" if beta <= 2 else "BKZ"

    # 3. 调用 Rust 接口 (内存级交互)
    # result 是一个字典，包含: 'matrix_str', 'log_prod', 'min_norm', 'cos_matrix'
    result = my_project_backend.run_reduction_rust(mat_str, method, int(beta))

    # 4. 解析返回的矩阵字符串 (格式: [[1 2][3 4]])
    # C++ dump_matrix_core 返回的格式需要解析回 Python list
    new_matrix_str = result["matrix_str"]

    # 简单的解析逻辑 (比正则快)
    new_matrix = []
    # 去掉首尾的 [ ]
    content = new_matrix_str.strip()[1:-1]
    if content:
        # 按行分割 (假设 C++ 用 \n 分隔行)
        rows = content.split("\n")  # 或者 split('][') 取决于 C++ dump 实现
        for r in rows:
            # 清理行内的括号
            r_clean = r.replace("[", "").replace("]", "").strip()
            if r_clean:
                new_matrix.append([int(x) for x in r_clean.split()])

    metrics = {
        "log_prod": result["log_prod"],  # Sum log ||b_i||
        "cos_matrix": np.array(result["cos_matrix"], dtype=np.float32),
        # 如果 C++ 没算 theta_min，这里可以用 cos_matrix 快速算
    }
    return new_matrix, metrics  # 返回 result 以便利用 C++ 算好的 Metrics


# ------------------------------
# Safe float conversion (avoid overflow)
# ------------------------------
def _shift_int_toward_zero(x: int, shift: int) -> int:
    if shift <= 0:
        return int(x)
    if x >= 0:
        return int(x >> shift)
    return -int((-x) >> shift)


def safe_float_basis(basis, target_bits: int = 50) -> np.ndarray:
    """
    支持方阵/矩形矩阵：
      - basis: m 行向量，每行长度 n（m 可以 != n）
    逐行按 2^shift 缩放到 float64 安全范围，保持方向信息用于点积/GS/角度。
    """
    m = len(basis)
    if m == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n = len(basis[0])
    for i in range(1, m):
        if len(basis[i]) != n:
            raise ValueError("safe_float_basis: inconsistent row lengths")

    out = np.zeros((m, n), dtype=np.float64)

    for i, row in enumerate(basis):
        max_abs = 0
        for v in row:
            av = abs(int(v))
            if av > max_abs:
                max_abs = av
        shift = max(0, int(max_abs.bit_length()) - int(target_bits)) if max_abs else 0

        for j, v in enumerate(row):
            out[i, j] = float(_shift_int_toward_zero(int(v), shift))
    return out


# ------------------------------
# Features: cosine matrix upper triangle + theta_min
# ------------------------------
def cosine_upper_triangle_features(basis):
    X = safe_float_basis(basis)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1) + 1e-30
    C = (X @ X.T) / (np.outer(norms, norms) + 1e-30)
    C = np.clip(C, -1.0, 1.0)

    iu = np.triu_indices(n, 1)
    upper = C[iu].astype(np.float32)

    max_cos = float(np.max(upper)) if upper.size else -1.0
    max_cos = float(np.clip(max_cos, -1.0, 1.0))
    theta_min = float(np.arccos(max_cos))
    return upper, theta_min


# ------------------------------
# GS log norms (log||b_i*||)
# ------------------------------
def get_gs_log_norms(basis) -> np.ndarray:
    """
    支持 m×n（m个向量，每个在n维），返回长度 m 的 log||b*_i||。
    """
    B = safe_float_basis(basis)
    m, n = B.shape
    if m == 0:
        return np.zeros(0, dtype=np.float64)

    b_star = np.zeros((m, n), dtype=np.float64)
    gs_log_norms = np.zeros(m, dtype=np.float64)

    for i in range(m):
        v = B[i].copy()
        for j in range(i):
            denom = float(np.dot(b_star[j], b_star[j]))
            if denom <= 1e-300:
                continue
            mu = float(np.dot(B[i], b_star[j]) / denom)
            v -= mu * b_star[j]
        b_star[i] = v
        norm_sq = float(np.dot(v, v))
        gs_log_norms[i] = 0.5 * np.log(norm_sq) if norm_sq > 1e-300 else -690.0
    return gs_log_norms


# ------------------------------
# log orthogonality defect (fast approx)
# log δ = Σlog||b_i|| - Σlog||b_i*||
# ------------------------------
def compute_log_ortho_defect_fast(basis) -> float:
    """
    支持 m×n：
      log δ = Σ log||b_i|| - Σ log||b*_i||
    这个对“局部 block（beta×dim）”也有意义：衡量这组向量的“非正交程度”。
    """
    X = safe_float_basis(basis)
    m = X.shape[0]
    if m == 0:
        return 0.0

    orig_norms = np.linalg.norm(X, axis=1) + 1e-300
    log_prod_norms = float(np.sum(np.log(orig_norms)))

    log_det_like = float(np.sum(get_gs_log_norms(basis)))  # Σ log||b*_i||
    return float(log_prod_norms - log_det_like)


# ------------------------------
# Fast metrics for plotting (optional)
# ------------------------------
def compute_metrics_float(basis):
    X = safe_float_basis(basis)
    n = X.shape[0]

    gs_logs = get_gs_log_norms(basis)
    log_det = float(np.sum(gs_logs))

    orig_norms = np.linalg.norm(X, axis=1) + 1e-300
    log_prod_norms = float(np.sum(np.log(orig_norms)))
    log_defect = log_prod_norms - log_det
    ortho_defect = float(np.exp(log_defect))

    log_gh = (1.0 / n) * log_det + 0.5 * (np.log(n) - np.log(2 * np.pi * np.e))
    b1_norm = float(orig_norms[0])
    gh_ratio = float(np.exp(np.log(b1_norm) - log_gh))
    return ortho_defect, gh_ratio


# ------------------------------
# NoisyNet Dueling DDQN
# ------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
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


class DuelingDDQN_Noisy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1),
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_stream(feat)
        a = self.adv_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DQNAgent:
    def __init__(
        self, state_dim, action_dim, use_amp=True, batch_size=256, updates_per_step=2
    ):
        self.device = device
        self.use_amp = use_amp and self.device.type == "cuda"
        self.batch_size = int(batch_size)
        self.updates_per_step = int(updates_per_step)

        self.q_net = DuelingDDQN_Noisy(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDDQN_Noisy(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=200000)

        self.gamma = 0.99
        self.tau = 0.005

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def act(self, state, is_training=True):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if is_training:
            self.q_net.train()
            self.q_net.reset_noise()
        else:
            self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, float(d)))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        losses = []
        for _ in range(self.updates_per_step):
            batch = random.sample(self.memory, self.batch_size)
            s, a, r, ns, d = zip(*batch)

            s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
            ns = torch.as_tensor(np.array(ns), dtype=torch.float32, device=self.device)
            a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
            r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
            d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

            self.q_net.train()
            self.q_net.reset_noise()

            with torch.no_grad():
                next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
                self.target_net.reset_noise()
                next_q = self.target_net(ns).gather(1, next_actions)
                target_q = r + (1.0 - d) * self.gamma * next_q

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                self.q_net.reset_noise()
                curr_q = self.q_net(s).gather(1, a)
                loss = F.mse_loss(curr_q, target_q)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

            losses.append(float(loss.item()))
        return float(np.mean(losses))


# ------------------------------
# Environment (改进版：state/action/reward/step pipeline)
# ------------------------------
class LatticeEnv:
    def __init__(
        self,
        matrix_path,
        max_steps=20,
        metrics_every=5,
        debug_step=False,
        alpha=1.0,
        beta_w=0.3,
        gamma=0.1,
        cost_w=0.002,
    ):
        self.initial_matrix = parse_challenge_file(matrix_path)
        self.dim = len(self.initial_matrix)

        self.max_steps = int(max_steps)
        self.metrics_every = int(metrics_every)
        self.debug_step = bool(debug_step)

        # reward weights
        self.alpha = float(alpha)
        self.beta_w = float(beta_w)
        self.gamma = float(gamma)
        self.cost_w = float(cost_w)

        # action a=(beta,pos)
        self.betas = [2 + 4 * i for i in range(10)]  # 2..38
        self.action_list = []
        for beta in self.betas:
            if beta > self.dim:
                continue
            for pos in range(0, self.dim - beta + 1):
                self.action_list.append((beta, pos))

        self.num_actions = len(self.action_list)
        self.action_map = {i: self.action_list[i] for i in range(self.num_actions)}

        # state dim: cos upper triangle + gs_logs + (theta_min, log_defect)
        self.state_dim = (self.dim * (self.dim - 1)) // 2 + self.dim + 2

        self._cached_metrics = (float("inf"), float("inf"))
        self._step_global = 0
        self.initial_log_vol = np.sum(get_gs_log_norms(self.initial_matrix))
        self.reset()

    def reset(self):
        self.basis = [row[:] for row in self.initial_matrix]
        self.current_step = 0
        self._step_global += 1

        # init: LLL
        self.basis, self.last_rust_info = run_reduction_wrapper(
            self.basis, 2, f"init_{self._step_global}"
        )
        # robust metrics for logging
        self._cached_metrics = compute_metrics(self.basis)

        self.state = self._get_state(self.last_rust_info)
        return self.state

    def _get_state(self, rust_info):

        # 1. Cosine 特征 (直接从 Rust 拿)
        cos_feat, theta_min = extract_features_from_rust(rust_info, self.dim)

        # 2. GS 特征 (目前还得 Python 算，除非升级 C++)
        gs_logs = get_gs_log_norms(self.basis)
        gs_feat = (gs_logs - gs_logs.mean()) / (gs_logs.std() + 1e-6)
        gs_feat = gs_feat.astype(np.float32)

        # 3. Defect (利用 Rust 算的 log_prod)
        # log_defect = log_prod_norms - log_vol
        log_prod = rust_info["log_prod"]
        log_defect = float(log_prod - self.initial_log_vol)

        tail = np.array([theta_min, log_defect], dtype=np.float32)
        s = np.concatenate([cos_feat, gs_feat, tail], axis=0).astype(np.float32)
        return s

    def step(self, action_idx):
        beta, pos = self.action_map[int(action_idx)]

        # ---------------------------------------------------------
        # 1. 获取旧状态指标 (从 self.last_rust_info 快速获取)
        # ---------------------------------------------------------
        old_cos_feat, old_theta = extract_features_from_rust(
            self.last_rust_info, self.dim
        )

        # 利用 Rust 算的 log_prod 快速算 defect
        old_log_def = float(self.last_rust_info["log_prod"] - self.initial_log_vol)

        # 计算局部 Block 的 defect (目前 Rust 没返回局部信息，只能 Python 算，这是性能瓶颈)
        # 如果为了极致速度，可以暂时去掉 R_local
        old_block = [row[:] for row in self.basis[pos : pos + beta]]
        old_local_log_def = float(compute_log_ortho_defect_fast(old_block))

        # ---------------------------------------------------------
        # 2. 执行动作 (调用 Rust)
        # ---------------------------------------------------------
        # 🛠️ 修复: 去掉非法的 ... 语法，传入正确的 step_id
        self.basis, new_rust_info = run_reduction_wrapper(
            self.basis, beta, self._step_global
        )

        # ---------------------------------------------------------
        # 3. 获取新状态指标
        # ---------------------------------------------------------
        new_cos_feat, new_theta = extract_features_from_rust(new_rust_info, self.dim)
        new_log_def = float(new_rust_info["log_prod"] - self.initial_log_vol)

        new_block = [row[:] for row in self.basis[pos : pos + beta]]
        new_local_log_def = float(compute_log_ortho_defect_fast(new_block))

        # 更新全局步数和缓存
        self.current_step += 1
        self._step_global += 1
        self.last_rust_info = new_rust_info

        # ---------------------------------------------------------
        # 4. 计算 Reward
        # ---------------------------------------------------------
        R_global = old_log_def - new_log_def
        R_local = old_local_log_def - new_local_log_def
        R_theta = new_theta - old_theta  # 假设 theta 越大越好(越正交)

        reward = self.alpha * R_global + self.beta_w * R_local + self.gamma * R_theta
        reward -= self.cost_w * float(beta)

        done = self.current_step >= self.max_steps

        # ---------------------------------------------------------
        # 5. Logging & State Update
        # ---------------------------------------------------------
        # Robust metrics (仅在需要时计算)
        defect_r, gh_ratio_r = 0.0, 0.0
        if (self.current_step % self.metrics_every) == 0 or done:
            defect_r, gh_ratio_r = compute_metrics(self.basis)

        # 🛠️ 修复: 传入 new_rust_info
        self.state = self._get_state(new_rust_info)

        info = {
            "beta": beta,
            "pos": pos,
            "R_global": float(R_global),
            "R_local": float(R_local),
            "R_theta": float(R_theta),
            "log_def_old": float(old_log_def),
            "log_def_new": float(new_log_def),
            "theta_old": float(old_theta),
            "theta_new": float(new_theta),
            "ortho_defect_robust": float(defect_r),
            "gh_ratio_robust": float(gh_ratio_r),
            "step": self.current_step,
        }

        if self.debug_step:
            print(
                f"  step={self.current_step:02d} a=(beta={beta},pos={pos}) "
                f"R=[g:{R_global:.4f} l:{R_local:.4f} th:{R_theta:.4f}] "
                f"logδ {old_log_def:.4f}->{new_log_def:.4f} "
                f"θ {old_theta:.4f}->{new_theta:.4f}"
            )

        return self.state, float(reward), bool(done), info


# ------------------------------
# Train
# ------------------------------
def train(env, agent, episodes=200, print_every=10, debug_each_step=False):
    history = {
        "reward": [],
        "loss": [],
        "updates": [],
        "mem": [],
        "gh_min_robust": [],
        "gh_last_robust": [],
        "gh_min_fast": [],
        "gh_last_fast": [],
    }

    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        losses = []
        updates = 0

        gh_robust_list = []
        gh_fast_list = []

        done = False
        while not done:
            a = agent.act(s, is_training=True)
            ns, r, done, info = env.step(a)

            agent.remember(s, a, r, ns, done)
            loss = agent.replay()
            if loss != 0.0:
                losses.append(loss)
                updates += 1

            s = ns
            ep_reward += r

            current_gh_r = float(info["gh_ratio_robust"])
            if current_gh_r > 1e-6:  # 只有当它被计算时(非0)才记录
                gh_robust_list.append(float(info["gh_ratio_robust"]))

            _, gh_fast = compute_metrics_float(env.basis)
            gh_fast_list.append(float(gh_fast))

            if debug_each_step:
                print(
                    f"    step={info['step']:02d} beta={info['beta']} pos={info['pos']} r={r:.4f} ghR={info['gh_ratio_robust']:.6f}"
                )

        history["reward"].append(ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["updates"].append(updates)
        history["mem"].append(len(agent.memory))

        gh_min_r = min(gh_robust_list) if gh_robust_list else float("inf")
        gh_last_r = gh_robust_list[-1] if gh_robust_list else float("inf")
        gh_min_f = min(gh_fast_list) if gh_fast_list else float("inf")
        gh_last_f = gh_fast_list[-1] if gh_fast_list else float("inf")

        history["gh_min_robust"].append(gh_min_r)
        history["gh_last_robust"].append(gh_last_r)
        history["gh_min_fast"].append(gh_min_f)
        history["gh_last_fast"].append(gh_last_f)

        if ep % print_every == 0:
            print(
                f"Ep {ep:4d} | R: {ep_reward:9.3f} | Loss: {history['loss'][-1]:.4f} | "
                f"upd: {updates:3d} | mem: {len(agent.memory):6d} | "
                f"robust[min/last]={gh_min_r:.12f}/{gh_last_r:.12f} | "
                f"fast[min/last]={gh_min_f:.12f}/{gh_last_f:.12f}"
            )

    return history


# ------------------------------
# Dataset path
# ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 确保输出目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. 定义你要使用的维度和可能的种子 (根据截图推测种子通常是 0)
DIM = 70
SEED = 0


# 2. 修改函数：接收 dim 参数，根据截图中的文件名格式构建路径
def get_challenge_file(dim, seed=0):
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    # ⚠️【关键修改】根据截图 image_f2f21e.png 的文件名格式进行拼接
    # 格式为: svpchallengedim{维度}seed{种子}.txt
    filename = f"svpchallengedim{dim}seed{seed}.txt"

    selected_file = os.path.join(DATASET_DIR, filename)

    # 检查文件是否存在，防止拼写错误或文件缺失
    if not os.path.exists(selected_file):
        raise FileNotFoundError(
            f"Target file not found: {selected_file}\n请检查 dataset 目录下是否有维度为 {dim} 的文件。"
        )

    print(f"📂 Using challenge file: {selected_file}")
    return selected_file


# 3. 调用函数
try:
    TRAIN_FILE = get_challenge_file(DIM, SEED)
    print("📂 Training on:", TRAIN_FILE)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")

TRAIN_FILE = get_challenge_file(DIM)
print("📂 Training on:", TRAIN_FILE)

# ------------------------------
# Run
# ------------------------------
env = LatticeEnv(
    TRAIN_FILE,
    max_steps=25,
    metrics_every=5,
    debug_step=False,
    alpha=1.0,
    beta_w=0.3,
    gamma=0.1,
    cost_w=0.002,
)

print(
    f"🤖 Action space size = {env.num_actions} (example first 10): {env.action_list[:10]}"
)
print(f"🧠 State dim = {env.state_dim}")

agent = DQNAgent(
    state_dim=env.state_dim,
    action_dim=env.num_actions,
    batch_size=256,
    updates_per_step=2,
)

# 你要 500 次就改这里
history = train(env, agent, episodes=1000, print_every=5, debug_each_step=False)

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history["reward"], label="Total Reward")
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["gh_min_fast"], label="GH_min_fast")
plt.title("Approximation Factor (Lower is Better)")
plt.xlabel("Episode")
plt.grid(True)

plot_path = os.path.join(RESULTS_DIR, "training_evolution.png")
plt.savefig(plot_path)
print(f"📊 Plot saved to {plot_path}")

print(f"Final Best GH_min_robust: {min(history['gh_min_robust'])}")
print(f"Final Best GH_min_fast:   {min(history['gh_min_fast'])}")
