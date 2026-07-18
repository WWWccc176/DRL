from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .action_space import build_action_spec
from .config import NUM_GLOBALS
from .network import DimAgnosticQNet
from .replay import MultiDimReplay
from .runtime import get_device

class DQNAgent:
    def __init__(
        self,
        num_globals=NUM_GLOBALS,
        batch_size=128,
        dims_per_update=3,
        capacity_per_dim=12000,
    ):
        self.device = get_device()
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
