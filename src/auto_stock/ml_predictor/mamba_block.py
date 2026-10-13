"""Mamba(선택적 상태공간모델, S6) 블록 — `MambaStock-main/mamba.py`(원본,
https://github.com/alxndrTL/mamba.py 기반)에서 그대로 가져왔다. 로직은 수정하지 않았다.

구조: `Mamba` = `ResidualBlock`(=`MambaBlock` + 정규화 + residual) 여러 개를 쌓은 것.
`MambaBlock`의 입력 x와 출력 y는 둘 다 (B, L, D)(B=배치, L=시퀀스 길이, D=모델 차원).
먼저 x를 (B, L, 2*ED)로 확장(E는 보통 2)해 x/z 두 갈래로 나눈다. x 갈래는 짧은 1D
컨볼루션 + SiLU 활성화를 거쳐 SSM에 들어가고, 그 출력에 silu(z)를 곱한다."""

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_stock.ml_predictor.pscan import pscan


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N
    expand_factor: int = 2  # E
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  # 학습 시 병렬 스캔(True) vs 순차 스캔(False)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)
        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        #     h : (B, ED, N)
        #     inputs : (B, ED, d_conv-1)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus의 역함수
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED), delta : (B, L, ED), A : (ED, N), B/C : (B, L, N), D : (ED)
        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # 순차(비병렬) 버전 — 짧은 시퀀스나 추론 시 참고용.
        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y

    def step(self, x, cache):
        # 자기회귀 추론 1스텝. x : (B, D), cache : (h, inputs). y : (B, D)
        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  # (B, ED), (B, ED)

        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)

        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  # (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED), h : (B, ED, N) -> y : (B, ED), h : (B, ED, N)
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = torch.split(
            deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h.squeeze(1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
