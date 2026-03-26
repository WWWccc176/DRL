import my_project_backend
import os, sys, math, time, random, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

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


# ToString
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


# NoisyNet Dueling DDQN
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


class CNN_DuelingDDQN_Noisy(nn.Module):
    def __init__(self, dim, action_dim):
        super().__init__()
        self.dim = dim
        self.matrix_size = dim * dim
        self.global_size = dim + 3

        # -------------------------------------------
        # 分支 A: CNN 处理 N x N 余弦相似度矩阵
        # -------------------------------------------
        self.cnn_branch = nn.Sequential(
            # 输入: (Batch, 1, N, N)
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 尺寸减半
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 尺寸再减半
            # 使用自适应池化，强制输出固定大小 (4x4)，这样无论 dim 是 68 还是 72 都能兼容
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),  # 输出维度: 32 * 4 * 4 = 512
        )

        # -------------------------------------------
        # 分支 B: MLP 处理一维全局特征 (GS范数, Ratio等)
        # -------------------------------------------
        self.global_branch = nn.Sequential(
            nn.Linear(self.global_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        # -------------------------------------------
        # 融合层: 拼接 CNN 和 MLP 的特征
        # -------------------------------------------
        fusion_dim = 512 + 64  # 576
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_dim, 256), nn.ReLU())

        # -------------------------------------------
        # Dueling 架构 (保持 NoisyNet 探索)
        # -------------------------------------------
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
        batch_size = x.size(0)

        # 1. 拆分输入数据
        matrix_flat = x[:, : self.matrix_size]
        global_feat = x[:, self.matrix_size :]

        # 2. Reshape 矩阵并送入 CNN
        # 变成 (Batch, Channels, Height, Width)
        matrix_2d = matrix_flat.view(batch_size, 1, self.dim, self.dim)
        cnn_out = self.cnn_branch(matrix_2d)

        # 3. 全局特征送入 MLP
        global_out = self.global_branch(global_feat)

        # 4. 融合特征
        fused = torch.cat([cnn_out, global_out], dim=1)
        feat = self.fusion_layer(fused)

        # 5. Dueling 输出
        v = self.value_stream(feat)
        a = self.adv_stream(feat)
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class DQNAgent:
    def __init__(
        self,
        dim,
        state_dim,
        action_dim,
        use_amp=False,
        batch_size=256,
        updates_per_step=4,
    ):
        self.device = device
        self.use_amp = use_amp and self.device.type == "cuda"
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step

        # 【修改】：使用新的 CNN 网络，并传入 dim
        self.q_net = CNN_DuelingDDQN_Noisy(dim, action_dim).to(self.device)
        self.target_net = CNN_DuelingDDQN_Noisy(dim, action_dim).to(self.device)

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

    def act_batch(self, states, is_training=True):
        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        if is_training:
            self.q_net.train()
            self.q_net.reset_noise()
        else:
            self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(s)
            # 返回一个动作列表，对应每个环境的动作
            return q_values.argmax(dim=1).cpu().numpy().tolist()

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
    def __init__(self, matrix_path, max_steps=20):
        self.initial_matrix_list = parse_challenge_file(matrix_path)
        self.dim = len(self.initial_matrix_list)
        raw_matrix_str = matrix_to_string(self.initial_matrix_list)
        self.state_dim = self.dim * self.dim + self.dim + 3
        self.max_steps = max_steps

        # 【修复】：将超参数固定在 init 中，降低 cost_w 鼓励探索
        self.alpha = 0.1
        self.ratio_w = 20.0
        self.gamma = 0.1
        self.cost_w = 0.0005  # 降低算力惩罚，防止智能体“摆烂”

        # 【修复】：移除 beta=2, 6，逼迫智能体使用真正的 BKZ (从 10 开始)
        self.betas = [10 + 4 * i for i in range(8)]  # [10, 14, 18, 22, 26, 30, 34, 38]
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

        state, _, current_ratio, _, _, _ = self._get_state_and_update_best(
            self.current_matrix_str, res
        )
        self.current_ep_best_ratio = current_ratio
        return state

    def _get_state_and_update_best(self, mat_str, rust_info):
        C = np.array(rust_info["cos_matrix"], dtype=np.float32)
        C_flat = C.flatten()
        lower = C[np.tril_indices(self.dim, -1)].astype(np.float32)

        max_cos = float(np.clip(np.max(lower) if lower.size > 0 else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size > 0 else 0.0, 0.0, 1.0))

        rust_eval = my_project_backend.evaluate_state_rust(mat_str, 0, 0)
        gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)
        gs_feat = (gs_logs - gs_logs.mean()) / (gs_logs.std() + 1e-6)

        log_b1 = gs_logs[0]
        log_defect = float(rust_info["log_prod"] - self.log_vol)
        log_ratio = float(log_b1 - self.log_GH)

        norm_log_defect = float(np.tanh(log_defect / 50.0))
        norm_log_ratio = float(np.tanh(log_ratio / 10.0))

        state_vec = np.concatenate(
            [C_flat, gs_feat, [max_cos, norm_log_defect, norm_log_ratio]], axis=0
        ).astype(np.float32)

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

        # 强制使用 BKZ (因为动作空间已经移除了 LLL)
        new_rust_info = my_project_backend.run_reduction_rust(
            self.current_matrix_str, "BKZ", beta
        )  # type: ignore
        self.current_matrix_str = new_rust_info["matrix_str"]
        self.last_rust_info = new_rust_info

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            final_rust_info = my_project_backend.run_reduction_rust(
                self.current_matrix_str, "LLL", 2
            )  # type: ignore
            self.current_matrix_str = final_rust_info["matrix_str"]
            self.last_rust_info = final_rust_info

        state, new_log_b1, new_ratio, new_max_cos, _, new_log_def = (
            self._get_state_and_update_best(
                self.current_matrix_str, self.last_rust_info
            )
        )

        R_global = old_log_def - new_log_def
        R_ratio = old_log_b1 - new_log_b1
        R_cos = old_max_cos - new_max_cos

        breakthrough_bonus = 0.0
        if new_ratio < self.best_ratio:
            breakthrough_bonus = 50.0
        elif new_ratio < self.current_ep_best_ratio:
            breakthrough_bonus = 10.0
            self.current_ep_best_ratio = new_ratio

        reward = (
            self.alpha * R_global
            + self.ratio_w * R_ratio
            + self.gamma * R_cos
            - self.cost_w * beta
            + breakthrough_bonus
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
def train(envs, agent, episodes=200, print_every=10, update_freq=4):
    history = {"reward": [], "loss": [], "ratio_min": []}
    num_envs = len(envs)

    for ep in range(1, episodes + 1):
        # 并行重置所有环境
        states = [env.reset() for env in envs]
        ep_rewards = [0.0] * num_envs
        losses = []
        ep_ratios = []
        dones = [False] * num_envs
        step_count = 0

        while not all(dones):
            # 过滤出还没结束的环境
            active_indices = [i for i, d in enumerate(dones) if not d]
            active_states = [states[i] for i in active_indices]

            # Agent 批量输出动作
            actions = agent.act_batch(active_states, is_training=True)

            # 定义单步执行函数
            def step_env(args):
                env_idx, action = args
                return env_idx, envs[env_idx].step(action)

            # 【核心并行】：使用线程池同时执行多个环境的 BKZ 约化
            with ThreadPoolExecutor(max_workers=num_envs) as executor:
                results = list(executor.map(step_env, zip(active_indices, actions)))

            # 收集经验并存入回放池
            for env_idx, (ns, r, done, info) in results:
                action_taken = actions[active_indices.index(env_idx)]
                agent.remember(states[env_idx], action_taken, r, ns, done)
                states[env_idx] = ns
                ep_rewards[env_idx] += r
                ep_ratios.append(info["b1_GH_ratio"])
                dones[env_idx] = done

            step_count += 1
            # 只要有经验就按频率更新网络
            if step_count % update_freq == 0 or all(dones):
                loss = agent.replay()
                if loss != 0.0:
                    losses.append(loss)

        # 记录本轮统计数据 (取所有并行环境的平均或最优值)
        avg_ep_reward = np.mean(ep_rewards)
        history["reward"].append(avg_ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)

        ep_min_ratio = min(ep_ratios) if ep_ratios else float("inf")
        history["ratio_min"].append(ep_min_ratio)

        # 获取所有环境中最好的历史记录
        global_best_ratio = min([env.best_ratio for env in envs])

        if ep % print_every == 0:
            print(
                f"Ep {ep:4d} | Avg R: {avg_ep_reward:9.3f} | Loss: {history['loss'][-1]:.4f} | "
                f"Ep min ratio: {ep_min_ratio:.4f} | historical best ratio: {global_best_ratio:.4f}"
            )

    return history


def run_experiment(dim, dataset_dir, results_dir, num_envs=4):
    print(f"🚀 [Dim {dim}] Process started with {num_envs} parallel environments...")
    train_file = os.path.join(dataset_dir, f"svpchallengedim{dim}seed0.txt")
    if not os.path.exists(train_file):
        print(f"❌ [Dim {dim}] File not found: {train_file}")
        return

    # 【修改】：实例化多个独立的环境
    envs = [LatticeEnv(train_file, max_steps=25) for _ in range(num_envs)]

    # Agent 只需要一个，状态维度取第一个环境的即可
    agent = DQNAgent(
        dim=dim, state_dim=envs[0].state_dim, action_dim=envs[0].num_actions
    )

    # 传入环境列表
    history = train(envs, agent, episodes=1000, print_every=20)

    # 【修改】：从所有并行环境中找出破纪录最狠的那个环境
    best_env = min(envs, key=lambda e: e.best_ratio)

    best_file_path = os.path.join(results_dir, f"best_results_dim{dim}.txt")
    with open(best_file_path, "w") as f:
        f.write(f"=== Lattice Reduction Best Results (Dim {dim}) ===\n")
        f.write(f"Best norm(b_1)/GH Ratio: {best_env.best_ratio:.8f}\n")
        f.write(f"Global Orthogonality Defect: {best_env.best_defect:.8f}\n")
        f.write(f"Max Cosine (Min Angle): {best_env.best_max_cos:.8f}\n")
        f.write(f"Min Cosine (Max Angle): {best_env.best_min_cos:.8f}\n")
        f.write("Best Vector (b_1):\n")
        if best_env.best_vector:
            f.write(" ".join(str(x) for x in best_env.best_vector) + "\n")
        else:
            f.write("None\n")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["reward"], label="Total Reward")
    plt.title(f"Dim {dim} - Reward per Episode")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["ratio_min"], label="norm(b1)/GH Ratio", color="orange")
    plt.axhline(y=best_env.best_ratio, color="r", linestyle="--", label="Global Best")
    plt.title(f"Dim {dim} - Approximation Factor")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(results_dir, f"training_evolution_dim{dim}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ [Dim {dim}] Training Complete! Results saved.")
    return dim


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    DIMS_TO_RUN = [68, 69, 70, 71, 72]

    print("🔥 Starting sequential training (one model at a time)...")

    for dim in DIMS_TO_RUN:
        print(f"\n" + "=" * 50)
        print(f"▶️  Starting dimension: {dim}")
        print("=" * 50)
        try:
            finished_dim = run_experiment(dim, DATASET_DIR, RESULTS_DIR)
            print(f"🎉 Training for Dim {finished_dim} exited successfully.")
        except KeyboardInterrupt:
            print(f"\n🛑 User manually interrupted training for Dim {dim}. Exiting...")
            break
        except Exception as exc:
            print(f"⚠️ An exception occurred while training Dim {dim}: {exc}")

    print("\n🏆 All sequential training tasks finished!")
