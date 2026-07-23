from __future__ import annotations

import numpy as np

from .config import (
    ACTION_BETA_COUNT,
    ACTION_BETA_MAX,
    ACTION_BETA_MIN,
    BETA_REF,
    BGJ3_MIN_BETA,
    ENUMERATION_MAX_BETA,
    SIEVE_MIN_BETA,
)


def beta_values(dim: int) -> list[int]:
    """Return the compact beta grid for one lattice dimension.

    The grid starts at beta=21, reaches the full legal dimension (capped at 95),
    and always contains the enumeration/BGJ2 and BGJ2/BGJ3 routing boundaries
    whenever they are legal for the current dimension.
    """
    beta_max = min(int(dim), ACTION_BETA_MAX)
    if beta_max < ACTION_BETA_MIN:
        return []

    raw = np.geomspace(ACTION_BETA_MIN, beta_max, ACTION_BETA_COUNT)
    betas = {
        max(ACTION_BETA_MIN, min(beta_max, int(round(value))))
        for value in raw
    }
    betas.update({ACTION_BETA_MIN, beta_max})

    for boundary in (ENUMERATION_MAX_BETA, SIEVE_MIN_BETA, BGJ3_MIN_BETA):
        if ACTION_BETA_MIN <= boundary <= beta_max:
            betas.add(boundary)

    return sorted(betas)


def build_action_list(dim: int):
    """Return legal learned ``(pos, beta)`` actions.

    Large blocks are expensive, so positions are sampled with a half-block
    stride while always retaining both the leftmost and rightmost legal window.
    This keeps the action space compact without dropping any routing boundary.
    """
    actions: list[tuple[int, int]] = []
    for beta in beta_values(dim):
        final_pos = dim - beta
        pos_step = max(1, beta // 2)
        positions = list(range(0, final_pos + 1, pos_step))
        if final_pos not in positions:
            positions.append(final_pos)
        actions.extend((pos, beta) for pos in sorted(set(positions)))
    return actions


def build_action_spec(dim: int, device):
    import torch

    actions = build_action_list(dim)
    if not actions:
        raise ValueError(
            f"dimension {dim} is smaller than ACTION_BETA_MIN={ACTION_BETA_MIN}"
        )

    poss = np.array([pos for pos, _ in actions], dtype=np.int64)
    betas = np.array([beta for _, beta in actions], dtype=np.int64)
    r0, r1 = poss, poss + betas
    end_idx = np.clip(poss + betas - 1, 0, dim - 1)
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

    groups: dict[int, tuple[list[int], list[int]]] = {}
    for index, (pos, beta) in enumerate(actions):
        groups.setdefault(beta, ([], []))
        groups[beta][0].append(index)
        groups[beta][1].append(pos)

    beta_groups = {
        int(beta): (tensor(indices, torch.long), tensor(positions, torch.long))
        for beta, (indices, positions) in groups.items()
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
