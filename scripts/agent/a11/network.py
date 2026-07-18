from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ACT_EMB, CTX_DIM, DILATIONS, FEAT_C, GS_EMB, GS_LOC, NUM_GLOBALS

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=1.0):
        super().__init__()
        self.in_features, self.out_features, self.std_init = (
            in_features,
            out_features,
            std_init,
        )
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
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        return F.linear(x, self.weight_mu, self.bias_mu)


# ============================================================
# 网络组件
# ============================================================
class GSEncoder(nn.Module):
    def __init__(self, out_ch=GS_EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, out_ch, 5, padding=2),
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.LeakyReLU(0.01),
        )

    def forward(self, gs):
        return self.net(gs.unsqueeze(1))


class FiLMAxialBlock(nn.Module):
    def __init__(self, c, ctx_dim, dilation):
        super().__init__()
        self.col = nn.Conv2d(
            c, c, (3, 1), padding=(dilation, 0), dilation=(dilation, 1)
        )
        self.row = nn.Conv2d(
            c, c, (1, 3), padding=(0, dilation), dilation=(1, dilation)
        )
        self.pw = nn.Conv2d(2 * c, c, 1)
        self.film = nn.Linear(ctx_dim, 2 * c)

    def forward(self, x, ctx):
        h = self.pw(torch.cat([self.col(x), self.row(x)], dim=1))
        gamma, beta = self.film(ctx).chunk(2, dim=1)
        h = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(
            -1
        ).unsqueeze(-1)
        return x + F.leaky_relu(h, 0.01)


def _integral_region_mean(H, r0, r1, c0, c1, area):
    I = F.pad(H, (1, 0, 1, 0)).cumsum(2).cumsum(3)
    s = I[:, :, r1, c1] - I[:, :, r0, c1] - I[:, :, r1, c0] + I[:, :, r0, c0]
    return s / area.view(1, 1, -1)


def _region_max(H, spec):
    B, C, D, _ = H.shape
    out = H.new_empty(B, C, spec["num_actions"])
    for b, (idx, pos) in spec["beta_groups"].items():
        k = min(b, D)
        pooled = F.max_pool2d(H, kernel_size=k, stride=1)
        out[:, :, idx] = pooled[:, :, pos, pos]
    return out


class DimAgnosticQNet(nn.Module):
    def __init__(
        self, num_globals=NUM_GLOBALS, c=FEAT_C, gs_emb=GS_EMB, ctx_dim=CTX_DIM
    ):
        super().__init__()
        self.c = c
        self.gs_encoder = GSEncoder(gs_emb)
        in_ch = 1 + 3 * gs_emb + 2
        self.stem = nn.Conv2d(in_ch, c, 1)
        self.global_mlp = nn.Sequential(
            nn.Linear(num_globals, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, ctx_dim),
            nn.LeakyReLU(0.01),
        )
        self.blocks = nn.ModuleList([FiLMAxialBlock(c, ctx_dim, d) for d in DILATIONS])

        self.value_mlp = nn.Sequential(
            NoisyLinear(2 * c + ctx_dim, 128), nn.LeakyReLU(0.01), NoisyLinear(128, 1)
        )
        act_feat = 2 * c + ACT_EMB + GS_LOC + ctx_dim
        self.action_mlp = nn.Sequential(
            NoisyLinear(act_feat, 128), nn.LeakyReLU(0.01), NoisyLinear(128, 1)
        )

    def _build_input(self, cos, gs_emb):
        B, D, _ = cos.shape
        Cg = gs_emb.shape[1]
        gi = gs_emb.unsqueeze(3).expand(B, Cg, D, D)
        gj = gs_emb.unsqueeze(2).expand(B, Cg, D, D)
        idx = torch.linspace(0.0, 1.0, D, device=cos.device)
        pi = idx.view(1, 1, D, 1).expand(B, 1, D, D)
        pj = idx.view(1, 1, 1, D).expand(B, 1, D, D)
        return torch.cat([cos.unsqueeze(1), gi, gj, gi - gj, pi, pj], dim=1)

    def forward(self, cos, gs, glob, spec):
        gs_emb = self.gs_encoder(gs)
        x = self.stem(self._build_input(cos, gs_emb))
        ctx = self.global_mlp(glob)
        for blk in self.blocks:
            x = blk(x, ctx)

        value = self.value_mlp(
            torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3)), ctx], dim=1)
        )

        rmean = _integral_region_mean(
            x, spec["r0"], spec["r1"], spec["c0"], spec["c1"], spec["area"]
        ).permute(0, 2, 1)
        rmax = _region_max(x, spec).permute(0, 2, 1)
        B, A, _ = rmean.shape
        aemb = spec["emb"].unsqueeze(0).expand(B, A, -1)
        gs_start = gs.gather(1, spec["pos"].unsqueeze(0).expand(B, A))
        gs_end = gs.gather(1, spec["end_idx"].unsqueeze(0).expand(B, A))
        gs_loc = torch.stack([gs_start, gs_end, gs_start - gs_end], dim=2)
        ctx_b = ctx.unsqueeze(1).expand(B, A, -1)
        adv = self.action_mlp(
            torch.cat([rmean, rmax, aemb, gs_loc, ctx_b], dim=2)
        ).squeeze(-1)

        return value + adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
