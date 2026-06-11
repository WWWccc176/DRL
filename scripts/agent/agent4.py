import my_project_backend
import os, sys, math, time, random, re, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp

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


class Transformer_DuelingDDQN(nn.Module):
    def __init__(self, max_dim, action_dim):
        super().__init__()
        self.max_dim = max_dim
        # 每个 Token 的特征维度: 1 (GS范数) + max_dim (与其他向量的余弦夹角)
        self.token_dim = max_dim

        # Transformer 编码器
        self.embedding = nn.Linear(self.token_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 全局标量特征处理 (max_cos, min_cos, defect, ratio)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(nn.Linear(128 + 32, 256), nn.ReLU())

        # Dueling 头
        self.value_stream = nn.Sequential(
            NoisyLinear(256, 128), nn.ReLU(), NoisyLinear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(256, 128), nn.ReLU(), NoisyLinear(128, action_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 解析输入状态 (由环境打包好的)
        # x 结构: [Tokens (max_dim * token_dim)] + [Scalars (4)]
        tokens_flat_size = self.max_dim * self.token_dim
        tokens_flat = x[:, :tokens_flat_size]
        scalars = x[:, tokens_flat_size:]

        # Reshape 为序列: (Batch, Seq_len, Feature_dim)
        tokens = tokens_flat.view(batch_size, self.max_dim, self.token_dim)

        # Transformer 处理
        emb = self.embedding(tokens)
        out_seq = self.transformer(emb)

        # 全局平均池化 (聚合所有基向量的信息)
        seq_pooled = out_seq.mean(dim=1)

        # 标量处理
        scalar_feat = self.scalar_mlp(scalars)

        # 融合与输出
        fused = torch.cat([seq_pooled, scalar_feat], dim=1)
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
class DQNAgent:
    def __init__(self, max_dim, state_dim, action_dim, batch_size=256):
        self.device = device
        self.batch_size = batch_size
        self.q_net = Transformer_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net = Transformer_DuelingDDQN(max_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)

        # 【修改 1】：大幅减小回放池容量，防止内存溢出 (从 200000 降到 20000)
        self.memory = deque(maxlen=50000)

        self.gamma = 0.99
        self.tau = 0.005

    # 【新增】：保存和加载大脑与记忆
    def save_checkpoint(self, model_path, memory_path=None):
        torch.save(self.q_net.state_dict(), model_path)
        if memory_path:
            with open(memory_path, "wb") as f:
                pickle.dump(self.memory, f)

    def load_checkpoint(self, model_path, memory_path=None):
        if os.path.exists(model_path):
            self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, "rb") as f:
                self.memory.extend(pickle.load(f))

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
        # 【修改 2】：存入回放池时，强制转换为 float16，内存占用直接减半！
        s_fp16 = s.astype(np.float16)
        ns_fp16 = ns.astype(np.float16)
        self.memory.append((s_fp16, a, r, ns_fp16, float(d)))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        # 【修改 3】：从回放池取出时，转回 float32 供 PyTorch 计算
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

        self.optimizer.zero_grad()
        curr_q = self.q_net(s).gather(1, a)
        loss = F.smooth_l1_loss(curr_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.75)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        return float(loss.item())


# ------------------------------
# Environment
# ------------------------------
class LatticeEnv:
    def __init__(self, matrix_path, max_dim=250):
        self.initial_matrix_list = parse_challenge_file(matrix_path)
        self.dim = len(self.initial_matrix_list)
        self.max_dim = max_dim  # 用于网络输入的固定最大维度
        raw_matrix_str = matrix_to_string(self.initial_matrix_list)

        # 增大 max_steps，因为现在是局部微操
        self.max_steps = math.ceil(self.dim * 1.5)

        self.alpha = 1.0
        self.ratio_w = 20.0
        self.gamma = 0.5
        self.cost_w = 0.005

        # 动作空间：包含 beta 和 pos
        self.betas = [10 + 4 * i for i in range(8)]
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
        self.state_dim = (self.max_dim * self.max_dim) + 4

        self.best_ratio = float("inf")
        self.best_vector = None
        self.best_max_cos = None
        self.best_min_cos = None
        self.best_defect = None

    def reset(self):
        self.current_step = 0
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

        rust_eval = my_project_backend.evaluate_state_rust(mat_str, 0, 0)
        gs_logs = np.array(rust_eval["gs_log_norms"], dtype=np.float32)

        log_b1 = gs_logs[0]
        log_defect = float(rust_info["log_prod"] - self.log_vol)
        log_ratio = float(log_b1 - self.log_GH)

        # 使用动态因子归一化
        norm_log_defect = float(np.tanh(log_defect / self.defect_scale))
        norm_log_ratio = float(np.tanh(log_ratio / self.ratio_scale))

        # 【构建 Transformer 序列状态】
        tokens = np.zeros((self.max_dim, self.max_dim), dtype=np.float32)
        for i in range(self.dim):
            # 填入与之前向量的余弦夹角 (下三角部分)
            tokens[i, :i] = C[i, :i]

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

        # 1. 执行局部块 BKZ (传入 pos)
        new_rust_info = my_project_backend.run_reduction_rust(
            self.current_matrix_str, "LOCAL_BKZ", beta, pos
        )
        self.current_matrix_str = new_rust_info["matrix_str"]

        # 2. 强制执行一次全局 LLL 维护格基质量
        lll_info = my_project_backend.run_reduction_rust(
            self.current_matrix_str, "LLL", 2, 0
        )
        self.current_matrix_str = lll_info["matrix_str"]
        self.last_rust_info = lll_info

        self.current_step += 1
        done = self.current_step >= self.max_steps

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
            breakthrough_bonus = 10.0
        elif new_ratio < self.current_ep_best_ratio:
            breakthrough_bonus = 3.0  # 回合内突破，给高分
            self.current_ep_best_ratio = new_ratio

        # 2. 核心逻辑重构：按情况给分，杜绝混日子
        if R_ratio > 1e-6:
            base_reward = 15.0 * R_ratio 
        elif R_ratio < -1e-6:
            # 恶性动作：把原本的第一向量搞长了（质量变差），必须严惩！
            base_reward = -0.5 + 5.0 * R_ratio 
        else:
            # 潜伏期动作：第一向量没变，这时候才看结构有没有改善（铺垫）
            # 【关键】：加一个基础的 step penalty (-0.1)，防止模型原地疯狂刷无用步数
            base_reward = -0.1 + 0.3 * R_global + 0.3 * R_cos

        reward = base_reward - self.cost_w * beta + breakthrough_bonus
        
        reward = float(np.clip(reward, -20.0, 50.0))

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
# 【修改细节 1】：在参数列表里加上 max_steps
def train(
    vec_env,
    agent,
    num_envs,
    max_steps,
    dim,
    episodes=200,
    print_every=10,
    save_dir="results",
):
    history = {"reward": [], "loss": [], "ratio_min": []}

    # 初始化一个局部最优记录用于比较
    best_known_ratio = float("inf")

    # 回合开始前，统一获取所有环境的初始状态
    states = vec_env.reset()

    for ep in range(1, episodes + 1):
        ep_rewards = np.zeros(num_envs)
        ep_ratios = []
        losses = []

        # 【核心修改】：放弃 while 循环，直接使用 for 循环固定步数推进
        # 因为所有环境每局固定走 max_steps 步，这样写最安全，绝不会出现进程脱节
        for step in range(max_steps):
            # 1. 神经网络根据当前状态输出动作
            actions = agent.act_batch(states, is_training=True)

            # 2. 4个环境同时执行动作 (底层去调 C++ 的 pos, beta, LLL)
            next_states, rewards, batch_dones, infos = vec_env.step(actions)

            # 3. 收集经验并训练
            for i in range(num_envs):
                agent.remember(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    batch_dones[i],
                )
                ep_rewards[i] += rewards[i]
                ep_ratios.append(infos[i]["b1_GH_ratio"])

            loss = agent.replay()
            if loss != 0.0:
                losses.append(loss)

            # 4. 状态滚动更新
            states = next_states

        # 一个 Episode 结束后的日志统计
        avg_ep_reward = np.mean(ep_rewards)
        history["reward"].append(avg_ep_reward)
        history["loss"].append(float(np.mean(losses)) if losses else 0.0)
        ep_min_ratio = min(ep_ratios) if ep_ratios else float("inf")
        history["ratio_min"].append(ep_min_ratio)

        bests = vec_env.get_bests()
        global_best_ratio = min([b[0] for b in bests])

        if global_best_ratio < best_known_ratio:
            best_known_ratio = global_best_ratio
            # 【修改处】：文件名加上维度，防止多维度依次训练时互相覆盖
            model_path = os.path.join(save_dir, f"agent4best_model_dim{dim}.pth")
            memory_path = os.path.join(save_dir, f"agent4best_memory_dim{dim}.pkl")

            # 破纪录时，把模型和包含突破路径的优质经验池全存下来
            agent.save_checkpoint(model_path, memory_path)
            print(
                f"agent4HisBest find {global_best_ratio:.4f}! weight and memory backed up"
            )

        if ep % print_every == 0:
            print(
                f"Ep {ep:4d} | Avg R: {avg_ep_reward:9.3f} | Loss: {history['loss'][-1]:.4f} | "
                f"Ep min ratio: {ep_min_ratio:.4f} | historical best ratio: {global_best_ratio:.4f}"
            )
    return history


def run_experiment(dim, dataset_dir, results_dir, num_envs=4):
    print(f"🚀 [Dim {dim}] Process started with {num_envs} parallel environments...")
    train_file = os.path.join(dataset_dir, f"svpchallengedim{dim}seed0.txt")

    # 初始化多进程向量环境
    vec_env = SubprocVecEnv(num_envs, train_file, max_dim=250)

    # 临时创建一个 env 获取动作维度
    temp_env = LatticeEnv(train_file, max_dim=250)
    agent = DQNAgent(
        max_dim=250, state_dim=temp_env.state_dim, action_dim=temp_env.num_actions
    )

    model_path = os.path.join(results_dir, f"agent4best_model_dim{dim}.pth")
    memory_path = os.path.join(results_dir, f"agent4best_memory_dim{dim}.pkl")
    agent.load_checkpoint(model_path, memory_path)

    history = train(
        vec_env,
        agent,
        num_envs,
        max_steps=temp_env.max_steps,
        dim=dim,  # <--- 传给 train 函数
        episodes=500,
        print_every=10,
        save_dir=results_dir,  # <--- 传给 train 函数
    )  # 获取最佳结果
    bests = vec_env.get_bests()
    best_idx = np.argmin([b[0] for b in bests])
    best_ratio, best_defect, best_max_cos, best_min_cos, best_vector = bests[best_idx]

    best_file_path = os.path.join(results_dir, f"best_results_dim{dim}.txt")
    with open(best_file_path, "w") as f:
        f.write(f"=== Lattice Reduction Best Results (Dim {dim}) ===\n")
        f.write(f"Best norm(b_1)/GH Ratio: {best_ratio:.8f}\n")
        f.write(f"Global Orthogonality Defect: {best_defect:.8f}\n")
        f.write(f"Max Cosine (Min Angle): {best_max_cos:.8f}\n")
        f.write(f"Min Cosine (Max Angle): {best_min_cos:.8f}\n")
        f.write("Best Vector (b_1):\n")
        if best_vector:
            f.write(" ".join(str(x) for x in best_vector) + "\n")

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
    print(f"✅ [Dim {dim}] Training Complete! Results saved.")
    return dim


if __name__ == "__main__":
    mp.set_start_method("spawn")  # 强制使用 spawn，防止 CUDA 和 C++ 库在 fork 时死锁
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    DIMS_TO_RUN = [68, 69, 70, 71, 72]
    for dim in DIMS_TO_RUN:
        run_experiment(dim, DATASET_DIR, RESULTS_DIR, num_envs=8)
