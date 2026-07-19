from __future__ import annotations

import random

import numpy as np
import torch


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
        return (
            self._retrieve(left, s)
            if s <= self.tree[left]
            else self._retrieve(left + 1, s - self.tree[left])
        )

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
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


class PERBuffer:
    PER_e, PER_a, PER_b, PER_b_inc = 1e-5, 0.6, 0.4, 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _prio(self, err):
        return (np.abs(err) + self.PER_e) ** self.PER_a

    def add(self, err, sample):
        self.tree.add(self._prio(err), sample)

    def sample(self, n):
        batch, idxs, prios = [], [], []
        total = self.tree.total()
        seg = total / n
        self.PER_b = min(1.0, self.PER_b + self.PER_b_inc)
        for i in range(n):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                idx, p, data = self.tree.get(random.uniform(0, total))
            prios.append(p)
            idxs.append(idx)
            batch.append(data)
        probs = np.array(prios) / (self.tree.total() + 1e-10)
        w = (self.tree.n_entries * probs + 1e-10) ** (-self.PER_b)
        w /= w.max()
        return batch, idxs, torch.as_tensor(w, dtype=torch.float32)

    def update(self, idx, err):
        self.tree.update(idx, self._prio(err))

    def __len__(self):
        return self.tree.n_entries


class MultiDimReplay:
    def __init__(self, capacity_per_dim=12000):
        self.cap = capacity_per_dim
        self.buffers = {}

    def _buf(self, dim):
        if dim not in self.buffers:
            self.buffers[dim] = PERBuffer(self.cap)
        return self.buffers[dim]

    def add(self, dim, sample):
        self._buf(dim).add(1.0, sample)

    def ready_dims(self, min_size):
        return [d for d, b in self.buffers.items() if len(b) >= min_size]
