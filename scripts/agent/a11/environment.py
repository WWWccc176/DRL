from __future__ import annotations

import math
import os

import numpy as np

from .action_space import build_action_list
from .backend import LatticeBackend
from .config import (
    DIM_REF,
    FINAL_POLISH_BETA,
    INITIAL_BKZ_BETA,
    RESULTS_DIR,
    STATE_PHASE_PERIOD,
)
from .io_utils import matrix_to_string, parse_challenge_file, parse_dim_seed, parse_fplll


class LatticeEnv:
    def __init__(
        self,
        filepath: str,
        env_id: int = 0,
        cpu_gate=None,
        gpu_gate=None,
        gpu_id: int | None = None,
    ):
        self.filepath = filepath
        self.env_id = env_id
        self.gpu_id = gpu_id
        self.backend = LatticeBackend(
            cpu_gate=cpu_gate,
            gpu_gate=gpu_gate,
            gpu_id=gpu_id,
        )
        self.dim, self.seed_id = parse_dim_seed(filepath)

        self.action_list = build_action_list(self.dim)
        self.num_actions = len(self.action_list)
        self.max_steps = math.ceil((self.dim + 3) * self.dim / 8)

        self.ratio_w, self.alpha, self.gamma_r, self.cost_w = 30.0, 2.0, 1.0, 0.15
        self.repeat_window, self.repeat_penalty_base = 8, 0.3

        self.best_ratio = float("inf")
        self.best_defect = self.best_max_cos = self.best_min_cos = None
        self.best_vector = self.best_basis = None
        self.best_episode = 0
        self.episode_count = 0
        self.best_dirty = False
        self.current_pool_id = -1

        self._preload()

    def _initial_feature_summary(self, info, log_vol=None):
        cos = np.asarray(info["cos_matrix"], dtype=np.float32)
        lower = cos[np.tril_indices(self.dim, -1)]
    
        max_cos = float(
            np.clip(
                np.max(lower) if lower.size else 0.0,
                0.0,
                1.0,
            )
        )
    
        min_cos = float(
            np.clip(
                np.min(lower) if lower.size else 0.0,
                0.0,
                1.0,
            )
        )
    
        gs = np.asarray(info["gs_log_norms"], dtype=np.float64)
    
        if log_vol is None:
            log_vol = float(np.sum(gs))
    
        log_gh = (
            log_vol / self.dim
            + 0.5 * math.log(
                self.dim / (2 * math.pi * math.e)
            )
        )
    
        log_b1 = float(gs[0])
        log_ratio = log_b1 - log_gh
        ratio = float(math.exp(log_ratio))
        log_defect = float(info["log_prod"] - log_vol)
    
        return {
            "log_vol": log_vol,
            "log_gh": log_gh,
            "log_b1": log_b1,
            "ratio": ratio,
            "log_defect": log_defect,
            "max_cos": max_cos,
            "min_cos": min_cos,
            "gs_head": float(gs[0]),
            "gs_tail": float(gs[-1]),
            "gs_length": int(len(gs)),
            "cos_shape": tuple(cos.shape),
        }


    def _log_initial_stage(self, stage, metrics):
        message = (
            f"\n"
            f"[A11 INIT][{stage}]\n"
            f"  env          = {self.env_id}\n"
            f"  dim          = {self.dim}\n"
            f"  seed         = {self.seed_id}\n"
            f"  file         = {os.path.basename(self.filepath)}\n"
            f"  Norm/GH      = {metrics['ratio']:.10f}\n"
            f"  log(b1)      = {metrics['log_b1']:.10f}\n"
            f"  log(GH)      = {metrics['log_gh']:.10f}\n"
            f"  log(volume)  = {metrics['log_vol']:.10f}\n"
            f"  log(defect)  = {metrics['log_defect']:.10f}\n"
            f"  max cosine   = {metrics['max_cos']:.10f}\n"
            f"  min cosine   = {metrics['min_cos']:.10f}\n"
            f"  GSO head     = {metrics['gs_head']:.10f}\n"
            f"  GSO tail     = {metrics['gs_tail']:.10f}\n"
            f"  GSO length   = {metrics['gs_length']}\n"
            f"  cosine shape = {metrics['cos_shape']}\n"
        )
    
        print(message, flush=True)
    
        init_dir = os.path.join(
            RESULTS_DIR,
            "initialization",
        )
        os.makedirs(init_dir, exist_ok=True)
    
        log_path = os.path.join(
            init_dir,
            f"env{self.env_id}_dim{self.dim}_seed{self.seed_id}.log",
        )
    
        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(message)
            f.flush()

    def _preload(self):
        print(
            "\n"
            + "=" * 80
            + "\n"
            + "[A11 INITIALIZATION START]\n"
            + f"  env  = {self.env_id}\n"
            + f"  dim  = {self.dim}\n"
            + f"  seed = {self.seed_id}\n"
            + f"  file = {os.path.basename(self.filepath)}\n"
            + "=" * 80,
            flush=True,
        )

        matrix = parse_challenge_file(self.filepath)

        print(
            f"[env{self.env_id}] "
            f"dim={self.dim} seed={self.seed_id} "
            f"-> Step 1/2: LLL preprocessing started",
            flush=True,
        )

        self.initial_pool_id = self.backend.create_matrix_lll(
            matrix_to_string(matrix)
        )

        lll_info = self.backend.evaluate(
            self.initial_pool_id
        )

        lll_metrics = self._initial_feature_summary(
            lll_info
        )

        self.log_vol = lll_metrics["log_vol"]
        self.log_GH = lll_metrics["log_gh"]

        self._log_initial_stage(
            "LLL",
            lll_metrics,
        )

        print(
            f"[env{self.env_id}] "
            f"dim={self.dim} seed={self.seed_id} "
            f"-> Step 1/2: LLL preprocessing completed",
            flush=True,
        )

        bkz_beta = min(
            INITIAL_BKZ_BETA,
            self.dim,
        )

        print(
            f"[env{self.env_id}] "
            f"dim={self.dim} seed={self.seed_id} "
            f"-> Step 2/2: Global BKZ2.0 beta={bkz_beta}, tours=1 started",
            flush=True,
        )

        bkz_info = self.backend.initial_bkz(
            self.initial_pool_id,
            bkz_beta,
        )

        bkz_metrics = self._initial_feature_summary(
            bkz_info,
            log_vol=self.log_vol,
        )

        self._log_initial_stage(
            f"BKZ{bkz_beta}",
            bkz_metrics,
        )

        print(
            f"[env{self.env_id}] "
            f"dim={self.dim} seed={self.seed_id} "
            f"-> Step 2/2: Global BKZ2.0 beta={bkz_beta}, tours=1 completed",
            flush=True,
        )

        gs = np.asarray(
            bkz_info["gs_log_norms"],
            dtype=np.float32,
        )

        self.defect_scale = max(
            abs(
                float(
                    bkz_info["log_prod"]
                    - self.log_vol
                )
            ),
            1.0,
        )

        self.ratio_scale = max(
            abs(
                float(
                    gs[0]
                    - self.log_GH
                )
            ),
            1.0,
        )

        beta_values = sorted(
            {
                beta
                for _, beta in self.action_list
            }
        )

        print(
            "\n"
            + "=" * 80
            + "\n"
            + "[A11 FORMAL TRAINING READY]\n"
            + f"  env                = {self.env_id}\n"
            + f"  dimension          = {self.dim}\n"
            + f"  seed               = {self.seed_id}\n"
            + f"  file               = {os.path.basename(self.filepath)}\n"
            + f"  initial state      = LLL + global BKZ{bkz_beta}\n"
            + f"  initial Norm/GH    = {bkz_metrics['ratio']:.10f}\n"
            + f"  initial log defect = {bkz_metrics['log_defect']:.10f}\n"
            + f"  initial max cosine = {bkz_metrics['max_cos']:.10f}\n"
            + f"  max steps          = {self.max_steps}\n"
            + f"  action count       = {self.num_actions}\n"
            + f"  beta min           = {beta_values[0]}\n"
            + f"  beta max           = {beta_values[-1]}\n"
            + f"  beta values        = {beta_values}\n"
            + "[A11] 开始正式训练\n"
            + "=" * 80,
            flush=True,
        )

    def reset(self):
        self.current_step = 0
        self.action_history = []
        self.episode_count += 1
        if self.current_pool_id >= 0:
            try:
                self.backend.free_matrix(self.current_pool_id)
            except Exception:
                pass

        self.current_pool_id = self.backend.clone_matrix(self.initial_pool_id)
        if self.current_pool_id < 0:
            raise RuntimeError(f"clone_matrix failed: {self.filepath}")

        info = self.backend.evaluate(self.current_pool_id)
        state, log_b1, ratio, max_cos, min_cos, log_def = self._build_state(info)
        self._c_logb1, self._c_maxcos, self._c_logdef = log_b1, max_cos, log_def
        self.initial_ep_ratio = self.current_ep_best_ratio = ratio
        self.last_info = info
        return state

    def _build_state(self, info):
        cos = np.asarray(info["cos_matrix"], dtype=np.float32)
        lower = cos[np.tril_indices(self.dim, -1)]
        max_cos = float(np.clip(np.max(lower) if lower.size else 0.0, 0.0, 1.0))
        min_cos = float(np.clip(np.min(lower) if lower.size else 0.0, 0.0, 1.0))
        cos = cos + cos.T

        gs = np.asarray(info["gs_log_norms"], dtype=np.float32)
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
                float((self.current_step % STATE_PHASE_PERIOD) / STATE_PHASE_PERIOD),
            ],
            dtype=np.float32,
        )
        state = {"cos": cos, "gs": gs_norm, "globals": globals_vec, "dim": self.dim}

        true_ratio = float(math.exp(log_ratio))
        self._maybe_update_best(true_ratio, log_def, max_cos, min_cos)
        return state, log_b1, true_ratio, max_cos, min_cos, log_def

    def _maybe_update_best(self, ratio, log_def, max_cos, min_cos):
        if ratio < self.best_ratio:
            self.best_ratio = ratio
            self.best_defect = log_def
            self.best_max_cos = max_cos
            self.best_min_cos = min_cos
            self.best_episode = self.episode_count
            matrix = parse_fplll(self.backend.dump_matrix(self.current_pool_id))
            if matrix:
                self.best_vector, self.best_basis = matrix[0], matrix
            self.best_dirty = True

    def pop_best_update(self):
        if not self.best_dirty:
            return None
        self.best_dirty = False
        payload = self.get_best_payload()
        payload.pop("basis", None)
        return payload

    def _exec_action(self, pos: int, beta: int):
        return self.backend.reduce(self.current_pool_id, pos, beta)

    def step(self, action_idx: int):
        pos, beta = self.action_list[action_idx]
        old_logb1, old_maxcos, old_logdef = (
            self._c_logb1,
            self._c_maxcos,
            self._c_logdef,
        )
        old_best, old_ep_best = self.best_ratio, self.current_ep_best_ratio

        self.last_info = self._exec_action(pos, beta)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            self.last_info = self.backend.final_polish(
                self.current_pool_id,
                min(FINAL_POLISH_BETA, self.dim),
            )

        state, new_logb1, new_ratio, new_maxcos, _, new_logdef = self._build_state(
            self.last_info
        )
        self._c_logb1, self._c_maxcos, self._c_logdef = (
            new_logb1,
            new_maxcos,
            new_logdef,
        )

        r_ratio = old_logb1 - new_logb1
        r_orth = old_maxcos - new_maxcos
        r_def = old_logdef - new_logdef

        if self.best_ratio < 1.08:
            w, alpha, gamma_r, cost_w = 15.0, 8.0, 5.0, 0.08
        elif self.best_ratio < 1.15:
            w, alpha, gamma_r, cost_w = 25.0, 3.0, 2.0, 0.12
        else:
            w, alpha, gamma_r, cost_w = (
                self.ratio_w,
                self.alpha,
                self.gamma_r,
                self.cost_w,
            )

        max_beta = max(b for _, b in self.action_list)
        reward = (
            w * r_ratio
            + alpha * r_orth
            + gamma_r * r_def
            - cost_w * (beta / max_beta)
        )

        if new_ratio < old_best:
            reward += 5.0
        elif new_ratio < old_ep_best:
            reward += 2.0
            self.current_ep_best_ratio = new_ratio
        if r_ratio > 1e-3 and pos <= 2 and beta >= 20:
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
        repeat_count = self.action_history.count(action_idx)
        if repeat_count >= 2:
            reward -= self.repeat_penalty_base * (repeat_count - 1) ** 1.5

        reward = float(np.clip(reward, -5.0, 50.0))
        info = {
            "beta": beta,
            "pos": pos,
            "b1_GH_ratio": new_ratio,
            "step": self.current_step,
            "backend": self.last_info.get("backend", "adaptive"),
            "accepted": bool(self.last_info.get("accepted", True)),
            "time_ms": float(self.last_info.get("time_ms", 0.0)),
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

    def close(self):
        for matrix_id in (
            getattr(self, "current_pool_id", -1),
            getattr(self, "initial_pool_id", -1),
        ):
            if matrix_id >= 0:
                try:
                    self.backend.free_matrix(matrix_id)
                except Exception:
                    pass
