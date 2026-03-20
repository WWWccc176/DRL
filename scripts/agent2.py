import my_project_backend
import os, sys, math, time, random, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.weight_epsilon.copy_(eps_out.ger(eps_in))  # type: ignore
        self.bias_epsilon.copy_(eps_out)  # type: ignore

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,  # type: ignore
                self.bias_mu + self.bias_sigma * self.bias_epsilon,  # type: ignore
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
        self, state_dim, action_dim, use_amp=False, batch_size=256, updates_per_step=1
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
        # 修复了 AMP API 警告
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
            # 修复了 autocast API 警告
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
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
        alpha=1.0,
        ratio_w=5.0,
        gamma=0.1,
        cost_w=0.002,
    ):
        self.initial_matrix_list = parse_challenge_file(matrix_path)
        raw_matrix_str = matrix_to_string(self.initial_matrix_list)
        self.dim = len(self.initial_matrix_list)
        self.max_steps = max_steps

        self.alpha, self.ratio_w, self.gamma, self.cost_w = (
            alpha,
            ratio_w,
            gamma,
            cost_w,
        )
        self.betas = [2 + 4 * i for i in range(10)]
        self.action_list = [
            (b, p) for b in self.betas if b <= self.dim for p in range(self.dim - b + 1)
        ]
        self.num_actions = len(self.action_list)

        init_res = my_project_backend.run_reduction_rust(raw_matrix_str, "LLL", 2)
        self.initial_matrix_str = init_res["matrix_str"]

        rust_eval = my_project_backend.evaluate_state_rust(
            self.initial_matrix_str, 0, 0
        )
        self.initial_gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)
        self.log_vol = np.sum(self.initial_gs_logs)
        self.log_GH = (self.log_vol / self.dim) + 0.5 * math.log(
            self.dim / (2 * math.pi * math.e)
        )

        # ==========================================
        # 修复 1：修正神经网络的输入维度！
        # 下三角矩阵长度 + gs_feat长度(self.dim) + 3个标量(max_cos, defect, ratio)
        # ==========================================
        self.state_dim = (self.dim * (self.dim - 1)) // 2 + self.dim + 3

        self.best_ratio = float("inf")
        self.best_vector = None
        self.best_max_cos = None
        self.best_min_cos = None
        self.best_defect = None

    def reset(self):
        self.current_step = 0
        res = my_project_backend.run_reduction_rust(self.initial_matrix_str, "LLL", 2)
        self.current_matrix_str = res["matrix_str"]
        self.last_rust_info = res

        # 接收当前初始状态的 ratio
        state, _, current_ratio, _, _, _ = self._get_state_and_update_best(
            self.current_matrix_str, res
        )

        # 新增：初始化当前回合的最佳 ratio
        self.current_ep_best_ratio = current_ratio

        return state

    def _get_state_and_update_best(self, mat_str, rust_info):
        C = np.array(rust_info["cos_matrix"], dtype=np.float32)
        lower = C[np.tril_indices(self.dim, -1)].astype(np.float32)

        max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))

        rust_eval = my_project_backend.evaluate_state_rust(mat_str, 0, 0)
        gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)
        gs_feat = (gs_logs - gs_logs.mean()) / (gs_logs.std() + 1e-6)

        # ==========================================
        # 修复 2：把 log_b1, log_defect, log_ratio 的计算提前！
        # ==========================================
        log_b1 = gs_logs[0]
        log_defect = float(rust_info["log_prod"] - self.log_vol)
        log_ratio = float(log_b1 - self.log_GH)

        # 现在再做 tanh 归一化就不会报错了
        norm_log_defect = float(np.tanh(log_defect / 10.0))
        norm_log_ratio = float(np.tanh(log_ratio / 5.0))

        # 组装完整的状态向量
        state_vec = np.concatenate(
            [lower, gs_feat, [max_cos, norm_log_defect, norm_log_ratio]], axis=0
        ).astype(np.float32)

        # 算出 Ratio 真值供记录
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

        # 1. 获取旧状态指标
        _, old_log_b1, _, old_max_cos, _, old_log_def = self._get_state_and_update_best(
            self.current_matrix_str, self.last_rust_info
        )
        old_theta = float(np.arccos(old_max_cos))

        # 2. 执行动作
        method = "LLL" if beta <= 2 else "BKZ"
        new_rust_info = my_project_backend.run_reduction_rust(
            self.current_matrix_str, method, beta
        )  # type: ignore
        self.current_matrix_str = new_rust_info["matrix_str"]
        self.last_rust_info = new_rust_info

        self.current_step += 1
        done = self.current_step >= self.max_steps

        # --- 新增：每一轮结束前，强制来一次 LLL ---
        if done:
            # 注：由于后端接口限制，这里对整个矩阵执行 LLL。
            # LLL 速度极快，且数学上完全覆盖了“前20个向量做LLL”的需求。
            final_rust_info = my_project_backend.run_reduction_rust(
                self.current_matrix_str, "LLL", 2
            )  # type: ignore
            self.current_matrix_str = final_rust_info["matrix_str"]
            self.last_rust_info = final_rust_info

        # 3. 获取新状态指标 (并在内部自动更新 best 记录)
        state, new_log_b1, new_ratio, new_max_cos, _, new_log_def = (
            self._get_state_and_update_best(
                self.current_matrix_str, self.last_rust_info
            )
        )
        new_theta = float(np.arccos(new_max_cos))

        # 4. 计算 Reward
        R_global = old_log_def - new_log_def

        # 新增：用 log(b1) 的变化量代表 norm(b1)/GH 的变化量 (因为 GH 是常数)
        # 如果 b1 变短了，old_log_b1 > new_log_b1，R_ratio 为正奖励
        R_ratio = old_log_b1 - new_log_b1

        R_cos = old_max_cos - new_max_cos

        breakthrough_bonus = 0.0
        # 判断当前步的新 ratio 是否打破了这局的记录或历史记录
        if new_ratio < self.best_ratio:
            breakthrough_bonus = 100.0  # 打破全局历史记录，给极其夸张的奖励
        elif new_ratio < self.current_ep_best_ratio:
            breakthrough_bonus = 10.0  # 就算没破世界纪录，打破了本局记录也给鼓励奖

        # 新增：更新当前回合的最优记录
        if new_ratio < self.current_ep_best_ratio:
            self.current_ep_best_ratio = new_ratio

        # 修改权重：削弱 R_global，极大地增强 R_ratio
        self.alpha = 0.1  # 把底薪降下来
        self.ratio_w = 50.0  # 把提成拉上去

        reward = (
            self.alpha * R_global
            + self.ratio_w * R_ratio
            + self.gamma * R_cos
            - self.cost_w * beta
            + breakthrough_bonus  # 加上暴击奖励
        )
        info = {
            "beta": beta,
            "pos": pos,
            "b1_GH_ratio": new_ratio,
            "step": self.current_step,
        }
        return state, float(reward), done, info


# ------------------------------
# Main & Train
# ------------------------------
# 【修改这里】：新增一个 update_freq=4 的参数
def train(env, agent, episodes=200, print_every=10, update_freq=4):
    history = {"reward": [], "loss": [], "ratio_min": []}

    for ep in range(1, episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        losses = []
        ep_ratios = []

        done = False
        step_count = 0  # 新增：记录当前回合走了几步

        while not done:
            a = agent.act(s, is_training=True)
            ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done)

            step_count += 1

            # ==========================================
            # 【核心提速修改】：每走 update_freq 步才向 GPU 发送一次更新指令，
            # 或者是到了回合最后一步（done）也强制更新一次。
            # ==========================================
            if step_count % update_freq == 0 or done:
                loss = agent.replay()
                if loss != 0.0:
                    losses.append(loss)

            s = ns
            ep_reward += r
            ep_ratios.append(info["b1_GH_ratio"])

        history["reward"].append(ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)

        ep_min_ratio = min(ep_ratios) if ep_ratios else float("inf")
        history["ratio_min"].append(ep_min_ratio)

        if ep % print_every == 0:
            print(
                f"Ep {ep:4d} | R: {ep_reward:9.3f} | Loss: {history['loss'][-1]:.4f} | "
                f"Ep min ratio: {ep_min_ratio:.4f} | historical best ratio: {env.best_ratio:.4f}"
            )

    return history


def run_experiment(dim, dataset_dir, results_dir):
    """
    封装单个维度的完整训练流程，供多进程调用
    """
    print(f"🚀 [Dim {dim}] Process started...")

    # 1. 动态生成文件路径
    train_file = os.path.join(dataset_dir, f"svpchallengedim{dim}seed0.txt")
    if not os.path.exists(train_file):
        print(f"❌ [Dim {dim}] File not found: {train_file}")
        return

    # 2. 初始化环境和智能体
    env = LatticeEnv(train_file, max_steps=25)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.num_actions)

    # 3. 开始训练
    history = train(env, agent, episodes=1000, print_every=50)  # 减少打印频率防刷屏

    # 4. 保存该维度的最优结果 (文件名带上 dim)
    best_file_path = os.path.join(results_dir, f"best_results_dim{dim}.txt")
    with open(best_file_path, "w") as f:
        f.write(f"=== Lattice Reduction Best Results (Dim {dim}) ===\n")
        f.write(f"Best norm(b_1)/GH Ratio: {env.best_ratio:.8f}\n")
        f.write(f"Global Orthogonality Defect: {env.best_defect:.8f}\n")
        f.write(f"Max Cosine (Min Angle): {env.best_max_cos:.8f}\n")
        f.write(f"Min Cosine (Max Angle): {env.best_min_cos:.8f}\n")
        f.write("Best Vector (b_1):\n")
        if env.best_vector:
            f.write(" ".join(str(x) for x in env.best_vector) + "\n")
        else:
            f.write("None\n")

    # 5. 绘制并保存该维度的图表
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["reward"], label="Total Reward")
    plt.title(f"Dim {dim} - Reward per Episode")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["ratio_min"], label="norm(b1)/GH Ratio", color="orange")
    plt.axhline(y=env.best_ratio, color="r", linestyle="--", label="Global Best")
    plt.title(f"Dim {dim} - Approximation Factor")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(results_dir, f"training_evolution_dim{dim}.png")
    plt.savefig(plot_path)
    plt.close()  # 必须 close，防止多进程绘图内存泄漏

    print(f"✅ [Dim {dim}] Training Complete! Results saved.")
    return dim


if __name__ == "__main__":
    import concurrent.futures
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 定义需要并行的维度列表
    DIMS_TO_RUN = [68, 69, 70, 71]

    # 获取 CPU 核心数，决定最大并行进程数
    max_workers = min(len(DIMS_TO_RUN), 3)
    print(f"🔥 Starting parallel training with {max_workers} workers...")

    # 使用进程池并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(run_experiment, dim, DATASET_DIR, RESULTS_DIR)
            for dim in DIMS_TO_RUN
        ]

        # 等待并获取结果
        for future in concurrent.futures.as_completed(futures):
            try:
                finished_dim = future.result()
                print(f"🎉 Process for Dim {finished_dim} exited successfully.")
            except Exception as exc:
                print(f"⚠️ A process generated an exception: {exc}")

    print("🏆 All parallel training tasks finished!")
