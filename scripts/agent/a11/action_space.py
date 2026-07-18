from __future__ import annotations

import math

import numpy as np

from .config import BETA_REF, ACTION_BETA_RATIO


def build_action_list(dim: int):
    """Return every legal (pos, beta).

    beta = 3, 7, 11, ... <= ceil(ACTION_BETA_RATIO * dim)
    pos  = 0, 1, 2, ... with pos + beta <= dim
    """
    beta_limit = math.ceil(ACTION_BETA_RATIO * dim)
    actions = []
    for beta in range(3, beta_limit + 1, 4):
        for pos in range(dim - beta + 1):
            actions.append((pos, beta))
    return actions


def build_action_spec(dim: int, device):
    import torch

    actions = build_action_list(dim)
    poss = np.array([pos for pos, _ in actions], dtype=np.int64)
    betas = np.array([beta for _, beta in actions], dtype=np.int64)
    r0, r1 = poss, poss + betas
    end_idx = poss + betas - 1
    area = betas.astype(np.float32) ** 2
    emb = np.stack(
        [
            betas / dim,
            poss / dim,
            (poss + betas) / dim,
            betas / BETA_REF,
            (poss + betas / 2.0) / dim,
        ],
        axis=1,
    ).astype(np.float32)

    def tensor(x, dtype):
        return torch.as_tensor(x, dtype=dtype, device=device)

    groups = {}
    for k, (pos, beta) in enumerate(actions):
        groups.setdefault(beta, ([], []))
        groups[beta][0].append(k)
        groups[beta][1].append(pos)
    beta_groups = {
        int(beta): (tensor(idx, torch.long), tensor(pos, torch.long))
        for beta, (idx, pos) in groups.items()
    }

    return {
        "action_list": actions,
        "num_actions": len(actions),
        "r0": tensor(r0, torch.long),
        "r1": tensor(r1, torch.long),
        "c0": tensor(r0, torch.long),
        "c1": tensor(r1, torch.long),
        "pos": tensor(poss, torch.long),
        "end_idx": tensor(end_idx, torch.long),
        "area": tensor(area, torch.float32),
        "emb": tensor(emb, torch.float32),
        "beta_groups": beta_groups,
    }
