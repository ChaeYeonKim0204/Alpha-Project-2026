"""models_muvo_2D.py — MUVO 2D-token + action-conditional latent diffusion.

기존 RSSMTD(transition)를 제거하고, 인코더 latent 토큰 위에서 동작하는
**action-conditional latent diffusion**(`LatentFutureDiffusion`)으로 미래를 생성한다.
"어떤 행동을 하면 어떤 미래가 되는가"가 핵심 task — CFG(classifier-free guidance)로
action 효과를 분리하고 guidance scale w로 그 강도를 조절(counterfactual).
설계: `alpha26/archive/hj/DIFFUSION_FUTURE_DESIGN.md`.

유지: FPNDecoder / PositionEmbeddingSine / SensorFusionTransformer / SpeedEncoder /
      PolicyDecoder / _TokenHead (인코더·디코더 백본).
제거: ConvGRUCellGlo / RepresentationModelTD / RSSMTD.
추가: timestep_embedding / DiTBlock / LatentFutureDiffusion.
변경: TokenConvDecoder2D(단일 state 입력, in_channels=C), Model(forward/generate_futures/
      counterfactual_futures + latent 정규화, policy 토큰 없는 368-token state).
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config_muvo_2D import cfg


# ─────────────────────────────────────────────
# Image Encoder: FPN
# ─────────────────────────────────────────────

class FPNDecoder(nn.Module):
    """Bottom-up FPN: start from xs[0] (largest spatial = lowest stride),
    pool down to xs[-1] (smallest spatial = highest stride). Output at
    smallest spatial size.
    """

    def __init__(self, feature_info, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_info[0], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.skip_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_info[i], out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            for i in [1, 2]
        ])
        self.out_channels = out_channels

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        x = self.conv1(xs[0])
        for i, conv in enumerate(self.skip_convs):
            skip = xs[i + 1]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.adaptive_max_pool2d(x, output_size=skip.shape[-2:])
            x = conv(skip) + x
        return x


# ─────────────────────────────────────────────
# Position embeddings
# ─────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal positional embedding (upstream `muvo_2d/.../common.py:636-678`)."""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, tensor):
        B, C, h, w = tensor.shape
        not_mask = torch.ones((B, h, w), device=tensor.device)
        y_embed  = not_mask.cumsum(1, dtype=torch.float32)
        x_embed  = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps     = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ─────────────────────────────────────────────
# Transformer Fusion (encoder side)
# ─────────────────────────────────────────────

class SensorFusionTransformer(nn.Module):
    """Image-only token transformer. type embedding 2 slot (image/speed)."""

    def __init__(self, channels, num_layers=3, nhead=4, dropout=0.1):
        super().__init__()
        self.channels        = channels
        self.position_encode = PositionEmbeddingSine(num_pos_feats=channels // 2, normalize=True)
        # 2 slot: image=0, speed=1.
        self.type_embedding  = nn.Parameter(torch.zeros(2, channels))
        nn.init.uniform_(self.type_embedding)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _feature_to_tokens(self, x):
        return x.flatten(2).permute(2, 0, 1)

    def _add_pos_and_type(self, x, sensor_type_idx):
        tokens    = self._feature_to_tokens(x)
        pos       = self.position_encode(x)
        pos_tokens = self._feature_to_tokens(pos)
        NHW, B, C = tokens.shape
        type_embed = self.type_embedding[sensor_type_idx].view(1, 1, C).expand(NHW, B, C)
        return tokens + pos_tokens + type_embed

    def forward(self, cam_feat, speed_embed=None):
        """speed_embed: (B, C) or None. None이면 speed modulation 건너뜀."""
        B, C, Hc, Wc = cam_feat.shape

        cam_tokens = self._add_pos_and_type(cam_feat, sensor_type_idx=0)

        if speed_embed is not None:
            speed_token = speed_embed.unsqueeze(0) + self.type_embedding[1].view(1, 1, C)  # (1, B, C)
            cam_tokens  = cam_tokens + speed_token

        out = self.transformer_encoder(cam_tokens)
        cam_out = out.permute(1, 2, 0).reshape(B, C, Hc, Wc)
        return cam_out


# ─────────────────────────────────────────────
# Speed encoder
# ─────────────────────────────────────────────

class SpeedEncoder(nn.Module):
    """upstream `muvo_2d/muvo/models/muvo.py:124-130`: Linear(1,16) → ReLU → Linear(16,256) → ReLU."""

    def __init__(self, hidden_channels=16, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_channels), nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels), nn.ReLU(True),
        )

    def forward(self, speed):
        """speed: (B*S, 1) → (B*S, out_channels)."""
        return self.net(speed)


# ─────────────────────────────────────────────
# Policy decoder (upstream 그대로)
# ─────────────────────────────────────────────

class PolicyDecoder(nn.Module):
    """upstream `muvo_2d/muvo/models/decoder.py:11-26` 그대로. in_channels = C (mean-pooled token)."""

    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            nn.Linear(in_channels // 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(x)


# ─────────────────────────────────────────────
# Latent diffusion: timestep embedding + DiT block + module
# ─────────────────────────────────────────────

def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding. timesteps: (B,) int → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiTBlock(nn.Module):
    """DiT block: AdaLN-Zero self-attn + (standard) cross-attn to condition memory + AdaLN-Zero MLP."""

    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # AdaLN: t_emb → 6 modulation params (shift/scale/gate for self-attn and MLP)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, memory, t_emb):
        s1, c1, g1, s2, c2, g2 = self.adaLN(t_emb).chunk(6, dim=-1)
        h = self._modulate(self.norm1(x), s1, c1)
        x = x + g1.unsqueeze(1) * self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), memory, memory, need_weights=False)[0]
        h = self._modulate(self.norm3(x), s2, c2)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class LatentFutureDiffusion(nn.Module):
    """Action-conditional latent diffusion over future tokens (전체 horizon 동시 생성).

    조건: 항상-on(z_past + 관측 speed) + CFG 대상(action_future, p_uncond로 드롭).
    학습: v(또는 eps) 예측. 추론: DDIM + CFG. counterfactual: 같은 noise/z_past에서 action만 변경.
    """

    def __init__(self, channels, n_tokens, receptive_field, future_horizon,
                 image_token_hw,
                 depth=6, heads=8, mlp_ratio=4,
                 num_train_steps=1000, parameterization="v", p_uncond=0.1):
        super().__init__()
        self.channels = channels
        self.n_tokens = n_tokens
        self.receptive_field = receptive_field
        self.future_horizon = future_horizon
        self.num_train_steps = num_train_steps
        self.parameterization = parameterization
        self.p_uncond = p_uncond

        H_img, W_img = image_token_hw
        n_img = H_img * W_img
        assert n_img == n_tokens, f"{n_img} != {n_tokens}"

        # cosine ᾱ schedule (length T)
        self.register_buffer("alphas_cumprod", self._cosine_alphas_cumprod(num_train_steps))

        # ── token spatial pos embedding (image-only 2D sine), buffer (N, C) ──
        pe = PositionEmbeddingSine(num_pos_feats=channels // 2, normalize=True)
        with torch.no_grad():
            img_pos = pe(torch.zeros(1, channels, H_img, W_img)).flatten(2).permute(0, 2, 1)[0]
        self.register_buffer("token_pos", img_pos)  # (N, C)

        # temporal embeddings
        self.temporal_future = nn.Parameter(torch.zeros(future_horizon, channels))
        self.temporal_past = nn.Parameter(torch.zeros(receptive_field, channels))
        nn.init.normal_(self.temporal_future, std=0.02)
        nn.init.normal_(self.temporal_past, std=0.02)

        # condition projections
        self.action_mlp = nn.Sequential(
            nn.Linear(2, channels), nn.SiLU(), nn.Linear(channels, channels),
        )
        self.null_action = nn.Parameter(torch.zeros(channels))  # CFG null-action 임베딩
        nn.init.normal_(self.null_action, std=0.02)
        self.speed_proj = nn.Linear(channels, channels)

        # timestep MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(channels, heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(channels, channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    @staticmethod
    def _cosine_alphas_cumprod(T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = (1 - (ac[1:] / ac[:-1])).clamp(1e-4, 0.999)
        return torch.cumprod(1.0 - betas, dim=0)  # (T,)

    def _alpha_sigma(self, tau):
        ac = self.alphas_cumprod[tau]  # (B,)
        return ac.sqrt().view(-1, 1, 1, 1), (1.0 - ac).sqrt().view(-1, 1, 1, 1)

    def _embed_tokens(self, z, temporal):
        """z: (B, T, C, N), temporal: (T, C) → (B, T*N, C) with pos/temporal embeddings."""
        B, T, C, N = z.shape
        x = z.permute(0, 1, 3, 2)  # (B, T, N, C)
        x = x + self.token_pos.view(1, 1, N, C)
        x = x + temporal.view(1, T, 1, C)
        return x.reshape(B, T * N, C)

    def denoise(self, z_tau, tau, z_past, action_future, speed_emb, drop_action=None):
        """z_tau: (B, FH, C, N), tau: (B,), z_past: (B, RF, C, N),
        action_future: (B, FH, 2), speed_emb: (B, C). → pred (B, FH, C, N)."""
        B = z_tau.shape[0]
        C = self.channels
        x = self._embed_tokens(z_tau, self.temporal_future)            # (B, FH*N, C)
        past_tokens = self._embed_tokens(z_past, self.temporal_past)   # (B, RF*N, C)
        speed_token = self.speed_proj(speed_emb).unsqueeze(1)          # (B, 1, C)

        a_emb = self.action_mlp(action_future)                        # (B, FH, C)
        if drop_action is not None:
            null = self.null_action.view(1, 1, C)
            if isinstance(drop_action, bool):
                if drop_action:
                    a_emb = null.expand(B, self.future_horizon, C)
            else:
                a_emb = torch.where(drop_action.view(B, 1, 1), null, a_emb)

        memory = torch.cat([past_tokens, speed_token, a_emb], dim=1)  # (B, RF*N+1+FH, C)
        t_emb = self.t_mlp(timestep_embedding(tau, C))               # (B, C)
        for block in self.blocks:
            x = block(x, memory, t_emb)
        x = self.final_linear(self.final_norm(x))                    # (B, FH*N, C)
        x = x.reshape(B, self.future_horizon, self.n_tokens, C).permute(0, 1, 3, 2)
        return x

    def training_targets(self, z_past, z_future, action_future, speed_emb):
        """v(또는 eps) 예측 타깃. action은 p_uncond로 드롭(CFG 학습). 손실은 trainer가 계산."""
        B = z_future.shape[0]
        tau = torch.randint(0, self.num_train_steps, (B,), device=z_future.device)
        alpha, sigma = self._alpha_sigma(tau)
        eps = torch.randn_like(z_future)
        z_tau = alpha * z_future + sigma * eps
        target = (alpha * eps - sigma * z_future) if self.parameterization == "v" else eps
        drop = torch.rand(B, device=z_future.device) < self.p_uncond
        pred = self.denoise(z_tau, tau, z_past, action_future, speed_emb, drop_action=drop)
        z0_hat = self._pred_to_x0(pred, z_tau, alpha, sigma)
        return {"v_pred": pred, "v_target": target, "z0_hat": z0_hat}

    def _pred_to_x0(self, pred, z_tau, alpha, sigma):
        if self.parameterization == "v":
            return alpha * z_tau - sigma * pred
        return (z_tau - sigma * pred) / alpha.clamp_min(1e-4)

    def _pred_to_x0_eps(self, pred, z_tau, alpha, sigma):
        if self.parameterization == "v":
            return alpha * z_tau - sigma * pred, sigma * z_tau + alpha * pred
        x0 = (z_tau - sigma * pred) / alpha.clamp_min(1e-4)
        return x0, pred

    def _guided_pred(self, z, tau, z_past, action_future, speed_emb, w):
        if w == 1.0:
            return self.denoise(z, tau, z_past, action_future, speed_emb, drop_action=False)
        cond = self.denoise(z, tau, z_past, action_future, speed_emb, drop_action=False)
        uncond = self.denoise(z, tau, z_past, action_future, speed_emb, drop_action=True)
        return uncond + w * (cond - uncond)

    @torch.no_grad()
    def sample(self, z_past, action_future, speed_emb, guidance_scale=1.0,
               num_samples=1, ddim_steps=50, noise=None):
        """DDIM(eta=0) + CFG. → (B, K, FH, C, N)."""
        B = z_past.shape[0]
        K = num_samples
        zp = z_past.repeat_interleave(K, dim=0)
        af = action_future.repeat_interleave(K, dim=0)
        se = speed_emb.repeat_interleave(K, dim=0)
        shape = (B * K, self.future_horizon, self.channels, self.n_tokens)
        z = noise if noise is not None else torch.randn(shape, device=z_past.device)

        steps = torch.linspace(self.num_train_steps - 1, 0, ddim_steps,
                               device=z_past.device).round().long()
        for i in range(len(steps)):
            tau = steps[i].expand(B * K)
            pred = self._guided_pred(z, tau, zp, af, se, guidance_scale)
            alpha, sigma = self._alpha_sigma(tau)
            x0, eps = self._pred_to_x0_eps(pred, z, alpha, sigma)
            if i < len(steps) - 1:
                a_next, s_next = self._alpha_sigma(steps[i + 1].expand(B * K))
                z = a_next * x0 + s_next * eps
            else:
                z = x0
        return z.reshape(B, K, self.future_horizon, self.channels, self.n_tokens)

    @torch.no_grad()
    def counterfactual(self, z_past, action_list, speed_emb, guidance_scale=1.0,
                       ddim_steps=50, seed=None):
        """같은 z_past/speed/noise에서 action만 바꿔 미래 생성 → (B, A, FH, C, N).
        동일 noise seed 공유 → 결과 차이가 순수 action 효과."""
        B = z_past.shape[0]
        gen = None
        if seed is not None:
            gen = torch.Generator(device=z_past.device).manual_seed(int(seed))
        base_noise = torch.randn(
            (B, self.future_horizon, self.channels, self.n_tokens),
            device=z_past.device, generator=gen,
        )
        outs = []
        for a in action_list:
            z_fut = self.sample(z_past, a, speed_emb, guidance_scale=guidance_scale,
                                 num_samples=1, ddim_steps=ddim_steps, noise=base_noise)
            outs.append(z_fut[:, 0])  # (B, FH, C, N)
        return torch.stack(outs, dim=1)  # (B, A, FH, C, N)


# ─────────────────────────────────────────────
# Token-shape decoder
# ─────────────────────────────────────────────

class _TokenHead(nn.Module):
    """1x1 Conv head, returns dict {f'{sensor_type}_{downsample_factor}': (B, S, C, H, W)}."""

    def __init__(self, in_channels, out_channels, downsample_factor, sensor_type):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.sensor_type = sensor_type
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, B, S):
        out = self.head(x)
        _, C, H, W = out.shape
        return {f'{self.sensor_type}_{self.downsample_factor}': out.view(B, S, C, H, W)}


class TokenConvDecoder2D(nn.Module):
    """Token-shape `(B, S, C, H, W)` 입력 → ConvTranspose ladder + head_4/2/1.

    RSSM 제거로 입력이 state 1개(=C)가 됨 (이전 cat(h,s)=2C에서 변경).
    카메라: (10, 24) → 5× stride-2 → (320, 768). LiDAR: (2, 64) → 4× stride-2 → (32, 1024).
    """

    def __init__(self, in_channels, out_channels, base_hw, target_hw, sensor_type,
                 decoder_channels=256, n_basic_conv=None):
        super().__init__()
        self.sensor_type = sensor_type
        self.in_channels = in_channels
        self.target_hw = tuple(target_hw)
        self.base_hw = tuple(base_hw)

        scale_h = target_hw[0] // base_hw[0]
        scale_w = target_hw[1] // base_hw[1]
        if scale_h != scale_w or scale_h <= 0 or (scale_h & (scale_h - 1)) != 0:
            raise ValueError(
                f"target_hw / base_hw must be the same power of 2; got "
                f"target={target_hw}, base={base_hw} → {scale_h}x vs {scale_w}x"
            )
        total_doublings = int(math.log2(scale_h))
        if n_basic_conv is None:
            n_basic_conv = max(0, total_doublings - 3)
        self.n_basic_conv = int(n_basic_conv)
        if self.n_basic_conv + 3 != total_doublings:
            raise ValueError(
                f"n_basic_conv({self.n_basic_conv}) + 3 != total_doublings({total_doublings}) "
                f"for base={base_hw} → target={target_hw}"
            )

        n_channels = decoder_channels
        mid_ch  = decoder_channels // 2
        low_ch  = decoder_channels // 4
        tiny_ch = decoder_channels // 8

        layers = [
            nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1),
            nn.ELU(),
        ]
        for _ in range(self.n_basic_conv):
            layers += [
                nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1),
                nn.ELU(),
            ]
        self.pre_transpose_conv = nn.Sequential(*layers)

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d(n_channels, mid_ch, kernel_size=6, stride=2, padding=2),
            nn.ELU(),
        )
        self.head_4 = _TokenHead(mid_ch, out_channels, downsample_factor=4, sensor_type=sensor_type)

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, low_ch, kernel_size=6, stride=2, padding=2),
            nn.ELU(),
        )
        self.head_2 = _TokenHead(low_ch, out_channels, downsample_factor=2, sensor_type=sensor_type)

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(low_ch, tiny_ch, kernel_size=6, stride=2, padding=2),
            nn.ELU(),
        )
        self.head_1 = _TokenHead(tiny_ch, out_channels, downsample_factor=1, sensor_type=sensor_type)

    def forward(self, state_slice):
        """state_slice: (B, S, C, H, W) — single state (no cat(h,s))."""
        B, S, C, H, W = state_slice.shape
        x = state_slice.reshape(B * S, C, H, W)
        x = self.pre_transpose_conv(x)
        x = self.trans_conv1(x); out_4 = self.head_4(x, B, S)
        x = self.trans_conv2(x); out_2 = self.head_2(x, B, S)
        x = self.trans_conv3(x); out_1 = self.head_1(x, B, S)
        return {**out_4, **out_2, **out_1}


# ─────────────────────────────────────────────
# World Model
# ─────────────────────────────────────────────

class Model(nn.Module):
    def __init__(self, cfg, embedding_n_channels=None, transformer_encoder=None):
        super().__init__()
        self.cfg = cfg
        if embedding_n_channels is None:
            embedding_n_channels = getattr(cfg.MODEL, "EMBEDDING_DIM", 256)
        transformer_channels = getattr(cfg.MODEL.FUSION, "TRANSFORMER_CHANNELS", embedding_n_channels)
        self.embedding_n_channels = embedding_n_channels
        self.transformer_channels = transformer_channels
        self.receptive_field = cfg.RECEPTIVE_FIELD
        self.future_horizon = cfg.FUTURE_HORIZON

        if embedding_n_channels != transformer_channels:
            raise ValueError(
                f"EMBEDDING_DIM == FUSION.TRANSFORMER_CHANNELS 필요; "
                f"got {embedding_n_channels} != {transformer_channels}"
            )

        # Image encoder
        self.image_encoder = timm.create_model(
            "resnet18", pretrained=True, features_only=True, out_indices=[2, 3, 4]
        )
        self.image_fpn = FPNDecoder([128, 256, 512], out_channels=transformer_channels)

        self.transformer_encoder = (transformer_encoder if transformer_encoder is not None
                                    else SensorFusionTransformer(
                                        channels=transformer_channels,
                                        num_layers=getattr(cfg.MODEL.FUSION, "TRANSFORMER_LAYERS", 3),
                                        nhead=getattr(cfg.MODEL.FUSION, "TRANSFORMER_HEADS", 4),
                                        dropout=getattr(cfg.MODEL.FUSION, "TRANSFORMER_DROPOUT", 0.1),
                                    ))

        # Speed encoder
        speed_channels = getattr(cfg.MODEL, "SPEED_CHANNELS", 16)
        self.speed_normalisation = getattr(cfg.MODEL, "SPEED_NORMALISATION", 50.0)
        self.speed_encoder = SpeedEncoder(hidden_channels=speed_channels, out_channels=embedding_n_channels)

        # Token geometry (image-only)
        rssm_cfg = cfg.MODEL.RSSM_2D
        self.image_token_hw = tuple(rssm_cfg.IMAGE_TOKEN_HW)

        # ── Latent diffusion (RSSMTD 대체) ──
        diff_cfg = cfg.MODEL.DIFFUSION
        if not getattr(diff_cfg, "ENABLED", True):
            raise ValueError("cfg.MODEL.DIFFUSION.ENABLED must be True.")
        self.diffusion = LatentFutureDiffusion(
            channels=embedding_n_channels,
            n_tokens=self.n_total_tokens,
            receptive_field=self.receptive_field,
            future_horizon=self.future_horizon,
            image_token_hw=self.image_token_hw,
            depth=getattr(diff_cfg, "DEPTH", 6),
            heads=getattr(diff_cfg, "HEADS", 8),
            mlp_ratio=getattr(diff_cfg, "MLP_RATIO", 4),
            num_train_steps=getattr(diff_cfg, "NUM_TRAIN_STEPS", 1000),
            parameterization=getattr(diff_cfg, "PARAMETERIZATION", "v"),
            p_uncond=getattr(diff_cfg, "P_UNCOND", 0.1),
        )

        # TRAIN_DIFFUSION=False면 forward에서 diffusion 타깃 계산을 건너뛴다
        # (reconstruction-only overfit). diffusion 모듈은 생성해두되 미사용.
        self.train_diffusion = bool(getattr(diff_cfg, "TRAIN_DIFFUSION", True))

        # ── latent 정규화 (diffusion 입력은 정규화 공간, 디코더는 raw 공간) ──
        self.latent_norm = getattr(diff_cfg, "LATENT_NORM", "ema")
        self.latent_norm_momentum = getattr(diff_cfg, "LATENT_NORM_MOMENTUM", 0.99)
        self.latent_scale = getattr(diff_cfg, "LATENT_SCALE", 1.0)
        self.register_buffer("latent_mean", torch.zeros(embedding_n_channels))
        self.register_buffer("latent_std", torch.ones(embedding_n_channels))
        self.register_buffer("latent_initialized", torch.zeros(1))

        # Token-shape decoders: image-only.
        decoder_channels = getattr(rssm_cfg, "DECODER_CHANNELS", 2 * embedding_n_channels)
        self.image_decoder = TokenConvDecoder2D(
            in_channels=embedding_n_channels, out_channels=3,
            base_hw=self.image_token_hw, target_hw=tuple(cfg.DATA.RGB_RECON_SIZE),
            sensor_type='rgb', decoder_channels=decoder_channels,
        )

        # PolicyDecoder: mean-pooled token (C) → action
        self.policy_decoder = PolicyDecoder(in_channels=embedding_n_channels)

    # ─── slicing helpers ──────────────────────────────────────────────────────

    @property
    def n_image_tokens(self):
        return self.image_token_hw[0] * self.image_token_hw[1]

    @property
    def n_total_tokens(self):
        return self.n_image_tokens

    def _slice_token_state(self, state):
        """state: (B, S, C, N) → image (B, S, C, H_img, W_img)."""
        B, S, C, N = state.shape
        H_img, W_img = self.image_token_hw
        return state.reshape(B, S, C, H_img, W_img)

    def _decode_state(self, state):
        """state: (B, S, C, N) (raw latent). → rgb_{1,2,4}, action_pred."""
        img = self._slice_token_state(state)
        out = {}
        out.update(self.image_decoder(img))
        # action_pred: mean-pooled token (policy 토큰이 없으므로) → (B, S, C)
        B, S, C, N = state.shape
        pooled = state.mean(dim=-1).reshape(B * S, C)
        out["action_pred"] = self.policy_decoder(pooled).reshape(B, S, 2)
        return out

    # ─── latent normalization (diffusion 공간 ↔ raw 공간) ─────────────────────

    def latent_normalize(self, z):
        """raw latent z (B,S,C,N) → 정규화 공간. 학습 중 EMA로 채널별 mean/std 추정."""
        if self.latent_norm == "none":
            return z
        if self.latent_norm == "fixed":
            return z / self.latent_scale
        if self.training:
            with torch.no_grad():
                mean = z.mean(dim=(0, 1, 3))
                std = z.std(dim=(0, 1, 3)).clamp_min(1e-4)
                if self.latent_initialized.item() < 1:
                    self.latent_mean.copy_(mean)
                    self.latent_std.copy_(std)
                    self.latent_initialized.fill_(1.0)
                else:
                    m = self.latent_norm_momentum
                    self.latent_mean.mul_(m).add_(mean, alpha=1 - m)
                    self.latent_std.mul_(m).add_(std, alpha=1 - m)
        mean = self.latent_mean.view(1, 1, -1, 1)
        std = self.latent_std.view(1, 1, -1, 1)
        return (z - mean) / std

    def latent_denormalize(self, z):
        """정규화 공간 z (B,T,C,N) → raw latent (디코딩 직전)."""
        if self.latent_norm == "none":
            return z
        if self.latent_norm == "fixed":
            return z * self.latent_scale
        mean = self.latent_mean.view(1, 1, -1, 1)
        std = self.latent_std.view(1, 1, -1, 1)
        return z * std + mean

    # ─── encode ──────────────────────────────────────────────────────────────

    def encode_fuse_sequence(self, image, speed=None):
        """Returns (B, S, C, N) token-shape embedding (image-only, N = H_img * W_img)."""
        B, S, C_img, H_img, W_img = image.shape
        image_flat = image.reshape(B * S, C_img, H_img, W_img)
        cam_feat = self.image_fpn(self.image_encoder(image_flat))

        speed_emb = None
        if speed is not None:
            speed_t = speed.float().to(image.device)
            if speed_t.ndim == 2:
                speed_t = speed_t.unsqueeze(-1)
            speed_emb = self.speed_encoder(speed_t.reshape(B * S, 1) / self.speed_normalisation)

        cam_out = self.transformer_encoder(cam_feat, speed_embed=speed_emb)
        cam_tokens = cam_out.flatten(2)  # (B*S, C, N)
        return cam_tokens.reshape(B, S, self.transformer_channels, -1)

    def _encode_speed_obs(self, speed_obs):
        """speed_obs: (B,) or (B,1) → (B, C). 관측(현재) speed만 (미래 speed 금지)."""
        s = speed_obs.float().to(self.latent_mean.device)
        if s.ndim == 1:
            s = s.unsqueeze(-1)
        return self.speed_encoder(s / self.speed_normalisation)

    @staticmethod
    def _batch_action(batch):
        if "action" in batch:
            return batch["action"].float()
        if "throttle_brake" in batch and "steering" in batch:
            return torch.cat([batch["throttle_brake"], batch["steering"]], dim=-1).float()
        raise KeyError("batch must contain 'action' or both 'throttle_brake' and 'steering'.")

    def _future_action(self, action):
        """미래 프레임 [RF, RF+FH)를 만든 action: a[RF-1 : RF-1+FH] (off-by-one 정렬)."""
        rf, fh = self.receptive_field, self.future_horizon
        return action[:, rf - 1: rf - 1 + fh]

    # ─── forward (학습) ────────────────────────────────────────────────────────

    def forward(self, batch, **kwargs):
        image = batch["image"].float()
        speed = batch.get("speed", None)
        action = self._batch_action(batch)

        rf, fh = self.receptive_field, self.future_horizon

        z = self.encode_fuse_sequence(image, speed=speed)          # (B, S, C, N) raw
        recon = self._decode_state(z)                              # 전체 8프레임 디코딩 (raw)

        # reconstruction-only 모드: diffusion 타깃 계산 skip → out에 v_pred 없음 →
        # compute_loss가 diffusion 손실을 자동으로 건너뛴다.
        if not self.train_diffusion:
            return {**recon}, {"z": z}

        z_norm = self.latent_normalize(z)
        z_past = z_norm[:, :rf]
        z_future = z_norm[:, rf:rf + fh]
        action_future = self._future_action(action)               # (B, FH, 2)
        speed_obs = self._encode_speed_obs(speed[:, rf - 1]) if speed is not None \
            else torch.zeros(image.shape[0], self.embedding_n_channels, device=image.device)

        diff = self.diffusion.training_targets(z_past, z_future.detach(), action_future, speed_obs)

        out = {**recon,
               "v_pred": diff["v_pred"], "v_target": diff["v_target"], "z0_hat": diff["z0_hat"]}
        return out, {"z": z}

    # ─── 미래 생성 / counterfactual (추론·검증) ───────────────────────────────

    def _encode_past(self, batch):
        rf = self.receptive_field
        image = batch["image"][:, :rf].float()
        speed = batch.get("speed", None)
        speed_past = speed[:, :rf] if speed is not None else None
        z_past_raw = self.encode_fuse_sequence(image, speed=speed_past)
        z_past = self.latent_normalize(z_past_raw)
        if speed is not None:
            speed_emb = self._encode_speed_obs(speed[:, rf - 1])
        else:
            speed_emb = torch.zeros(image.shape[0], self.embedding_n_channels, device=image.device)
        return z_past, speed_emb

    @torch.no_grad()
    def generate_futures(self, batch, num_samples=1, guidance_scale=1.0, ddim_steps=50):
        """같은 행동(GT)에서 noise만 달리한 미래 K개. → list of K dict(rgb_*, action_pred)."""
        z_past, speed_emb = self._encode_past(batch)
        action_future = self._future_action(self._batch_action(batch))
        z_fut_k = self.diffusion.sample(z_past, action_future, speed_emb,
                                        guidance_scale=guidance_scale,
                                        num_samples=num_samples, ddim_steps=ddim_steps)
        return [self._decode_state(self.latent_denormalize(z_fut_k[:, k]))
                for k in range(num_samples)]

    @torch.no_grad()
    def decode_future_per_frame(self, batch, guidance_scale=1.0, ddim_steps=50):
        """미래 horizon은 joint diffusion으로 한 번에 생성하되, 각 미래 시점을
        **독립적으로 디코딩**(decode(z_fut[:, t]))해 프레임별로 따로 반환.

        → list of FH dict(rgb_*, action_pred). 각 dict의 frame 차원은 S=1."""
        z_past, speed_emb = self._encode_past(batch)
        action_future = self._future_action(self._batch_action(batch))
        z_fut = self.diffusion.sample(z_past, action_future, speed_emb,
                                      guidance_scale=guidance_scale,
                                      num_samples=1, ddim_steps=ddim_steps)
        z_fut = self.latent_denormalize(z_fut[:, 0])  # (B, FH, C, N) raw latent
        # 시점별 단일 프레임(S=1) slice를 따로 디코딩 → 프레임 독립 출력.
        return [self._decode_state(z_fut[:, t:t + 1]) for t in range(self.future_horizon)]

    @torch.no_grad()
    def counterfactual_futures(self, batch, action_list, guidance_scale=1.0,
                               ddim_steps=50, seed=0):
        """★ 같은 z_past/speed/noise에서 행동만 바꿔 미래 디코딩. → list of A dict."""
        z_past, speed_emb = self._encode_past(batch)
        z_fut_a = self.diffusion.counterfactual(z_past, action_list, speed_emb,
                                                guidance_scale=guidance_scale,
                                                ddim_steps=ddim_steps, seed=seed)
        return [self._decode_state(self.latent_denormalize(z_fut_a[:, a]))
                for a in range(len(action_list))]
