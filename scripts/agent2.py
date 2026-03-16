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
# Config
# ------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)


# ------------------------------
# 极速解析工具
# ------------------------------
def matrix_to_string(basis):
    lines = [" ".join(str(x) for x in row) for row in basis]
    return "[" + "\n".join(f"[{l}]" for l in lines) + "]"


def string_to_matrix_fast(mat_str):
    """仅在需要 mpmath 高精度计算时调用，平时 RL 循环中不再使用"""
    content = mat_str.strip()[1:-1]
    if not content:
        return []
    rows = content.split("\n")
    return [
        [int(x) for x in r.replace("[", "").replace("]", "").split()]
        for r in rows
        if r.strip()
    ]


# ------------------------------
# 高精度 Metrics
# ------------------------------
mp.dps = 100


def compute_metrics_robust(basis, prec=200):
    mp.dps = prec
    Bmp = [[mp.mpf(int(x)) for x in row] for row in basis]
    n = len(Bmp)
    if n == 0:
        return float("inf"), float("inf")

    bstar = [[] for _ in range(n)]
    bstar_norm2 = [mp.mpf("0")] * n

    for i in range(n):
        bstar[i] = Bmp[i].copy()
        for j in range(i):
            if bstar_norm2[j] == 0:
                continue
            mu_ij = mp.fdot(Bmp[i], bstar[j]) / bstar_norm2[j]
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
    gh_ratio = float(mp.e ** (log_b1 - log_gh))

    # 简化 ortho_defect 的计算，重点关注 gh_ratio
    return float("inf"), gh_ratio


# ------------------------------
# NoisyNet Dueling DDQN (保持原样，很棒)
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
        self, state_dim, action_dim, use_amp=False, batch_size=256, updates_per_step=2
    ):
        self.device = device
        self.use_amp = use_amp and self.device.type == "cuda"
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
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
            return int(self.q_net(s).argmax(dim=1).item())

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
                loss = F.smooth_l1_loss(curr_q, target_q)

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
# Environment
# ------------------------------
def parse_challenge_file(filepath):
    with open(filepath, "r") as f:
        tokens = re.findall(r"[-+]?\d+", f.read())
    numbers = [int(t) for t in tokens]
    total = len(numbers)
    dim = (
        int(math.isqrt(total))
        if int(math.isqrt(total)) ** 2 == total
        else int(math.isqrt(total - 1))
    )
    return np.array(numbers[-dim * dim :], dtype=object).reshape(dim, dim).tolist()


class LatticeEnv:
    def __init__(
        self,
        matrix_path,
        max_steps=20,
        metrics_every=5,
        alpha=1.0,
        beta_w=0.3,
        gamma=0.1,
        cost_w=0.002,
    ):
        self.initial_matrix_list = parse_challenge_file(matrix_path)
        self.initial_matrix_str = matrix_to_string(self.initial_matrix_list)
        self.dim = len(self.initial_matrix_list)
        self.max_steps = max_steps
        self.metrics_every = metrics_every

        self.alpha, self.beta_w, self.gamma, self.cost_w = alpha, beta_w, gamma, cost_w
        self.betas = [2 + 4 * i for i in range(10)]
        self.action_list = [
            (b, p) for b in self.betas if b <= self.dim for p in range(self.dim - b + 1)
        ]
        self.num_actions = len(self.action_list)

        # state dim: cos lower triangle + gs_logs + (theta_min, log_defect)
        self.state_dim = (self.dim * (self.dim - 1)) // 2 + self.dim + 2

        # 预计算全局初始体积
        rust_eval = my_project_backend.evaluate_state_rust(
            self.initial_matrix_str, 0, 0
        )
        self.initial_log_vol = np.sum(rust_eval["gs_log_norms"])

    def reset(self):
        self.current_step = 0
        # 初始规约使用 LLL
        res = my_project_backend.run_reduction_rust(self.initial_matrix_str, "LLL", 2)
        self.current_matrix_str = res["matrix_str"]
        self.last_rust_info = res
        return self._get_state(self.current_matrix_str, res)

    def _get_state(self, mat_str, rust_info):
        # 1. Cosine 特征
        C = np.array(rust_info["cos_matrix"], dtype=np.float32)
        lower = C[np.tril_indices(self.dim, -1)].astype(np.float32)
        max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        theta_min = float(np.arccos(max_cos))

        # 2. 调用 Rust 获取 GS 特征 (极速)
        rust_eval = my_project_backend.evaluate_state_rust(mat_str, 0, 0)
        gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)
        gs_feat = (gs_logs - gs_logs.mean()) / (gs_logs.std() + 1e-6)

        # 3. Defect
        log_defect = float(rust_info["log_prod"] - self.initial_log_vol)

        return np.concatenate([lower, gs_feat, [theta_min, log_defect]], axis=0).astype(
            np.float32
        )

    def step(self, action_idx):
        beta, pos = self.action_list[action_idx]

        # 1. 旧特征提取 (利用 Rust 快速评估旧块缺陷)
        old_eval = my_project_backend.evaluate_state_rust(
            self.current_matrix_str, pos, beta
        )
        old_local_log_def = old_eval["local_log_defect"]
        old_log_def = float(self.last_rust_info["log_prod"] - self.initial_log_vol)

        C_old = np.array(self.last_rust_info["cos_matrix"], dtype=np.float32)
        lower_old = C_old[np.tril_indices(self.dim, -1)]
        old_theta = float(
            np.arccos(np.clip(np.max(lower_old) if lower_old.size else 0, 0, 1))
        )

        # 2. 执行动作 (全权交给 C++)
        method = "LLL" if beta <= 2 else "BKZ"
        new_rust_info = my_project_backend.run_reduction_rust(  # type: ignore
            self.current_matrix_str, method, beta
        )
        self.current_matrix_str = new_rust_info["matrix_str"]

        # 3. 新特征提取
        new_eval = my_project_backend.evaluate_state_rust(  # type: ignore
            self.current_matrix_str, pos, beta
        )
        new_local_log_def = new_eval["local_log_defect"]
        new_log_def = float(new_rust_info["log_prod"] - self.initial_log_vol)

        C_new = np.array(new_rust_info["cos_matrix"], dtype=np.float32)
        lower_new = C_new[np.tril_indices(self.dim, -1)]
        new_theta = float(
            np.arccos(np.clip(np.max(lower_new) if lower_new.size else 0, 0, 1))
        )

        self.current_step += 1
        self.last_rust_info = new_rust_info

        # 4. 计算 Reward
        R_global = old_log_def - new_log_def
        R_local = old_local_log_def - new_local_log_def
        R_theta = new_theta - old_theta
        reward = (
            self.alpha * R_global
            + self.beta_w * R_local
            + self.gamma * R_theta
            - self.cost_w * beta
        )

        done = self.current_step >= self.max_steps

        # 5. 指标记录 (只在特定步数把 String 解码为 List 给 mpmath 计算)
        gh_ratio_r = 0.0
        if (self.current_step % self.metrics_every) == 0 or done:
            current_basis_list = string_to_matrix_fast(self.current_matrix_str)
            _, gh_ratio_r = compute_metrics_robust(current_basis_list)

        state = self._get_state(self.current_matrix_str, new_rust_info)
        info = {
            "beta": beta,
            "pos": pos,
            "gh_ratio_robust": gh_ratio_r,
            "step": self.current_step,
        }
        return state, float(reward), done, info


# ------------------------------
# Main & Train (与原版逻辑基本一致)
# ------------------------------
def train(env, agent, episodes=200, print_every=10):
    history = {"reward": [], "loss": [], "gh_min_robust": []}

    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        losses = []
        gh_robust_list = []

        done = False
        while not done:
            a = agent.act(s, is_training=True)
            ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done)
            loss = agent.replay()
            if loss != 0.0:
                losses.append(loss)
            s = ns
            ep_reward += r
            if info["gh_ratio_robust"] > 1e-6:
                gh_robust_list.append(info["gh_ratio_robust"])

        history["reward"].append(ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)

        gh_min = min(gh_robust_list) if gh_robust_list else float("inf")
        history["gh_min_robust"].append(gh_min)

        if ep % print_every == 0:
            print(
                f"Ep {ep:4d} | R: {ep_reward:9.3f} | Loss: {history['loss'][-1]:.4f} | GH_min: {gh_min:.8f}"
            )

    return history


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    DIM = 70
    TRAIN_FILE = os.path.join(DATASET_DIR, f"svpchallengedim{DIM}seed0.txt")
    print("📂 Training on:", TRAIN_FILE)

    env = LatticeEnv(TRAIN_FILE, max_steps=25, metrics_every=5)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.num_actions)

    history = train(env, agent, episodes=1000, print_every=5)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["reward"], label="Total Reward")
    plt.title("Reward per Episode")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["gh_min_robust"], label="GH Ratio (Robust)")
    plt.title("Approximation Factor (Lower is Better)")
    plt.grid(True)

    plot_path = os.path.join(RESULTS_DIR, "training_evolution.png")
    plt.savefig(plot_path)
    print(f"📊 Plot saved to {plot_path}")
