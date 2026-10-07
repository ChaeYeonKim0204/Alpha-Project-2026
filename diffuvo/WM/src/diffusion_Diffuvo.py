"""diffusion_Diffuvo.py — Action-conditioned latent diffusion future predictor.

`diffuvo_plan.md` §3.3 의 신규 모듈. RSSM transition을 대체해
`p_θ(z_future | z_past, action, speed)` 조건부 분포를 학습한다.

구성 (계획서 §3.3):
- `CosineNoiseSchedule` : cosine ᾱ 스케줄 + v-prediction 변환 헬퍼.
- `DiTBlock`           : self-attn(future 토큰) + cross-attn(past 토큰)
                         + adaLN-Zero(timestep+action+speed) + MLP.
- `LatentFutureDiffusion` : forward(v_pred) / sample(DDIM+CFG).

latent 모양: encoder 융합 토큰 `(B, S, C=256, N=368)`.
diffusion은 FH 프레임을 **joint**(동일 timestep 공유)로 denoise하고,
past RF 프레임은 cross-attention memory로만 들어간다.
"""
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Noise schedule (cosine ᾱ + v-prediction)
# ─────────────────────────────────────────────
class CosineNoiseSchedule(nn.Module):
    """Nichol & Dhariwal cosine ᾱ 스케줄.

    ᾱ(t) = f(t)/f(0),  f(t) = cos((t/T + s)/(1+s) · π/2)².
    v-prediction(target): v = √ᾱ·ε − √(1−ᾱ)·z₀.
    buffer로 등록해 모델과 함께 device 이동 / state_dict 저장된다.
    """

    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        super().__init__()
        self.timesteps = int(timesteps)
        steps = torch.arange(self.timesteps + 1, dtype=torch.float64) / self.timesteps
        f = torch.cos((steps + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = (f / f[0]).clamp(1e-8, 1.0)  # (T+1,), ᾱ at integer steps
        self.register_buffer("alpha_bar", alpha_bar.float(), persistent=False)

    def _gather(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ_t → (B,1,1,1) for broadcasting over (B,FH,C,N)."""
        ab = self.alpha_bar.to(t.device)[t.clamp(0, self.timesteps).long()]
        return ab.view(-1, 1, 1, 1)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """forward diffusion: z_t = √ᾱ·z₀ + √(1−ᾱ)·ε. returns (z_t, v_target)."""
        ab = self._gather(t)
        sqrt_ab = ab.sqrt()
        sqrt_om = (1.0 - ab).sqrt()
        z_t = sqrt_ab * z0 + sqrt_om * noise
        v_target = sqrt_ab * noise - sqrt_om * z0
        return z_t, v_target

    def predict_start_and_noise(self, z_t, t, v_pred):
        """v_pred → (z₀_hat, ε_hat). DDIM 스텝에 사용."""
        ab = self._gather(t)
        sqrt_ab = ab.sqrt()
        sqrt_om = (1.0 - ab).sqrt()
        z0 = sqrt_ab * z_t - sqrt_om * v_pred
        eps = sqrt_om * z_t + sqrt_ab * v_pred
        return z0, eps


# ─────────────────────────────────────────────
# Timestep / conditioning embeddings
# ─────────────────────────────────────────────
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """sinusoidal timestep embedding. t: (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ─────────────────────────────────────────────
# DiT block (self + cross attention, adaLN-Zero)
# ─────────────────────────────────────────────
class DiTBlock(nn.Module):
    """future 토큰 self-attn + past 토큰 cross-attn + MLP.

    조건 벡터 c(=timestep+action+speed)로 9개 modulation(scale/shift/gate ×3)을
    만든다(adaLN-Zero). modulation Linear은 0으로 init → gate=0에서 시작(identity).
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        # adaLN-Zero: c → (shift,scale,gate) × {self, cross, mlp}
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _mod(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, memory, c):
        (sh_sa, sc_sa, g_sa,
         sh_ca, sc_ca, g_ca,
         sh_mlp, sc_mlp, g_mlp) = self.modulation(c).chunk(9, dim=-1)

        h = self._mod(self.norm_sa(x), sh_sa, sc_sa)
        x = x + g_sa.unsqueeze(1) * self.self_attn(h, h, h, need_weights=False)[0]

        h = self._mod(self.norm_ca(x), sh_ca, sc_ca)
        x = x + g_ca.unsqueeze(1) * self.cross_attn(h, memory, memory, need_weights=False)[0]

        h = self._mod(self.norm_mlp(x), sh_mlp, sc_mlp)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


# ─────────────────────────────────────────────
# Latent future diffusion model
# ─────────────────────────────────────────────
class LatentFutureDiffusion(nn.Module):
    """DiT denoiser over future latent tokens, conditioned on past + action + speed.

    forward(z_future_noisy, t, z_past, action, speed) → v_pred  (학습)
    sample(z_past, action, speed, n_steps, cfg_scale, generator) → z_future_hat (추론)
    """

    def __init__(
        self,
        token_channels: int = 256,    # C
        n_tokens: int = 368,          # N (image 240 + lidar 128)
        action_dim: int = 2,
        receptive_field: int = 4,     # RF (past frames)
        future_horizon: int = 2,      # FH (future frames, jointly diffused)
        hidden_dim: int = 256,        # DiT model dim D
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        timesteps: int = 1000,
        schedule_s: float = 0.008,
    ):
        super().__init__()
        self.C = token_channels
        self.N = n_tokens
        self.RF = receptive_field
        self.FH = future_horizon
        self.D = hidden_dim
        self.action_dim = action_dim

        self.schedule = CosineNoiseSchedule(timesteps=timesteps, s=schedule_s)

        # token C ↔ model dim D projections (D=C면 거의 identity지만 일반화 위해 둠).
        self.in_proj = nn.Linear(self.C, self.D)
        self.past_proj = nn.Linear(self.C, self.D)
        self.out_proj = nn.Linear(self.D, self.C)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # learned positional embeddings: (frame, token) 분리.
        self.fut_frame_pos = nn.Parameter(torch.zeros(1, self.FH, 1, self.D))
        self.past_frame_pos = nn.Parameter(torch.zeros(1, self.RF, 1, self.D))
        self.token_pos = nn.Parameter(torch.zeros(1, 1, self.N, self.D))
        nn.init.trunc_normal_(self.fut_frame_pos, std=0.02)
        nn.init.trunc_normal_(self.past_frame_pos, std=0.02)
        nn.init.trunc_normal_(self.token_pos, std=0.02)

        # conditioning embeddings → D (모두 합산).
        self.t_embed = nn.Sequential(nn.Linear(self.D, self.D), nn.SiLU(), nn.Linear(self.D, self.D))
        self.action_embed = nn.Sequential(nn.Linear(action_dim, self.D), nn.SiLU(), nn.Linear(self.D, self.D))
        self.speed_embed = nn.Sequential(nn.Linear(1, self.D), nn.SiLU(), nn.Linear(self.D, self.D))
        # CFG용 null-action 임베딩 (action drop / uncond 분기에서 사용).
        self.null_action = nn.Parameter(torch.zeros(self.D))

        self.blocks = nn.ModuleList([
            DiTBlock(self.D, n_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(self.D, elementwise_affine=False, eps=1e-6)

    # ── conditioning ──────────────────────────
    def _cond(self, t, action, speed, drop_action=None):
        """timestep + action(or null) + speed → (B, D) 조건 벡터."""
        c = self.t_embed(timestep_embedding(t, self.D))
        if action is None:
            a = self.null_action.expand(t.shape[0], -1)
        else:
            a = self.action_embed(action.float())
            if drop_action is not None:  # (B,) bool → drop된 샘플만 null로
                a = torch.where(drop_action.view(-1, 1), self.null_action.expand_as(a), a)
        c = c + a
        if speed is not None:
            sp = speed.float()
            if sp.ndim == 1:
                sp = sp.unsqueeze(-1)
            c = c + self.speed_embed(sp)
        return c

    def _embed_past(self, z_past):
        """(B, RF, C, N) → memory (B, RF*N, D)."""
        B = z_past.shape[0]
        x = z_past.permute(0, 1, 3, 2)              # (B, RF, N, C)
        x = self.past_proj(x)                       # (B, RF, N, D)
        x = x + self.past_frame_pos + self.token_pos
        return x.reshape(B, self.RF * self.N, self.D)

    def _embed_future(self, z_future):
        """(B, FH, C, N) → tokens (B, FH*N, D)."""
        B = z_future.shape[0]
        x = z_future.permute(0, 1, 3, 2)            # (B, FH, N, C)
        x = self.in_proj(x)                         # (B, FH, N, D)
        x = x + self.fut_frame_pos + self.token_pos
        return x.reshape(B, self.FH * self.N, self.D)

    def _denoise(self, z_future_noisy, t, memory, action, speed, drop_action=None):
        """공통 denoiser pass → v_pred (B, FH, C, N)."""
        B = z_future_noisy.shape[0]
        c = self._cond(t, action, speed, drop_action=drop_action)
        x = self._embed_future(z_future_noisy)
        for block in self.blocks:
            x = block(x, memory, c)
        x = self.out_proj(self.norm_out(x))         # (B, FH*N, C)
        x = x.reshape(B, self.FH, self.N, self.C).permute(0, 1, 3, 2)
        return x                                    # (B, FH, C, N)

    # ── training ──────────────────────────────
    def forward(self, z_future_noisy, t, z_past, action, speed, drop_action=None):
        """v_pred 반환. (학습 loop에서 v_target과 MSE.)"""
        memory = self._embed_past(z_past)
        return self._denoise(z_future_noisy, t, memory, action, speed, drop_action=drop_action)

    def loss(self, z_future_clean, z_past, action, speed, p_drop=0.0, generator=None, weight=None):
        """편의용 v-MSE 손실. t~U(1,T), ε~N(0,I), p_drop으로 CFG action drop.
        weight(B,) 주면 per-sample 가중평균(정지 프레임 다운웨이팅용)."""
        B = z_future_clean.shape[0]
        device = z_future_clean.device
        t = torch.randint(1, self.schedule.timesteps + 1, (B,), device=device, generator=generator)
        noise = torch.randn(z_future_clean.shape, device=device, generator=generator)
        z_t, v_target = self.schedule.q_sample(z_future_clean, t, noise)
        drop = None
        if p_drop > 0:
            drop = torch.rand(B, device=device, generator=generator) < p_drop
        v_pred = self.forward(z_t, t, z_past, action, speed, drop_action=drop)
        if weight is None:
            return F.mse_loss(v_pred, v_target)
        per = F.mse_loss(v_pred, v_target, reduction="none").flatten(1).mean(dim=1)  # (B,)
        w = weight.to(per.dtype)
        return (per * w).sum() / w.sum().clamp_min(1e-8)

    # ── inference (DDIM + CFG) ────────────────
    @torch.no_grad()
    def sample(self, z_past, action, speed, n_steps=50, cfg_scale=1.0, generator=None):
        """DDIM 샘플링(eta=0). action CFG 적용 → z_future_hat (B, FH, C, N).

        cfg_scale<=1 이면 uncond pass 생략(순수 conditional).
        seed 제어는 generator로(같은 z_past·action에서 seed만 바꾸면 다른 미래).
        """
        device = z_past.device
        B = z_past.shape[0]
        memory = self._embed_past(z_past)
        z = torch.randn(B, self.FH, self.C, self.N, device=device, generator=generator)

        T = self.schedule.timesteps
        # T..1 을 n_steps개로 균등 subsample (DDIM).
        seq = torch.linspace(T, 1, steps=n_steps, device=device).round().long()
        seq = torch.unique_consecutive(seq)
        use_cfg = cfg_scale is not None and cfg_scale != 1.0

        for i in range(len(seq)):
            t = seq[i].expand(B)
            t_prev = seq[i + 1].expand(B) if i + 1 < len(seq) else torch.zeros(B, dtype=torch.long, device=device)

            v_cond = self._denoise(z, t, memory, action, speed)
            if use_cfg:
                v_uncond = self._denoise(z, t, memory, None, speed)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            z0, eps = self.schedule.predict_start_and_noise(z, t, v)
            ab_prev = self.schedule._gather(t_prev)
            z = ab_prev.sqrt() * z0 + (1.0 - ab_prev).sqrt() * eps
        return z

    # ── config 헬퍼 ───────────────────────────
    @classmethod
    def from_cfg(cls, cfg):
        """cfg.DIFFUSION(없으면 default)에서 하이퍼파라미터를 읽어 생성.

        token 차원은 cfg.MODEL / cfg.RECEPTIVE_FIELD / cfg.FUTURE_HORIZON에서,
        diffusion 하이퍼는 cfg.DIFFUSION에서 가져온다(미배선이면 클래스 default).
        """
        d = getattr(cfg, "DIFFUSION", SimpleNamespace())
        rssm2d = getattr(cfg.MODEL, "RSSM_2D", SimpleNamespace())
        img_hw = getattr(rssm2d, "IMAGE_TOKEN_HW", (10, 24))
        lid_hw = getattr(rssm2d, "LIDAR_TOKEN_HW", (2, 64))
        n_tokens = img_hw[0] * img_hw[1] + lid_hw[0] * lid_hw[1]
        C = getattr(cfg.MODEL, "EMBEDDING_DIM", 256)
        return cls(
            token_channels=C,
            n_tokens=int(getattr(d, "N_TOKENS", n_tokens)),
            action_dim=getattr(cfg.MODEL, "ACTION_DIM", 2),
            receptive_field=getattr(cfg, "RECEPTIVE_FIELD", 4),
            future_horizon=getattr(cfg, "FUTURE_HORIZON", 2),
            hidden_dim=int(getattr(d, "HIDDEN_DIM", C)),
            depth=int(getattr(d, "DEPTH", 6)),
            n_heads=int(getattr(d, "N_HEADS", 8)),
            mlp_ratio=float(getattr(d, "MLP_RATIO", 4.0)),
            dropout=float(getattr(d, "DROPOUT", 0.0)),
            timesteps=int(getattr(d, "TIMESTEPS", 1000)),
            schedule_s=float(getattr(d, "SCHEDULE_S", 0.008)),
        )
