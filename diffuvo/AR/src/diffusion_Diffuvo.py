"""diffusion_Diffuvo.py (Diffuvo_AR) — Frame-wise autoregressive latent diffusion.

Diffuvo_WM의 **joint** future denoiser(`LatentFutureDiffusion`)를
**frame-wise autoregressive(AR)** denoiser로 바꾼 변형이다.
encoder/decoder 기반 latent representation(`(B, S, C=256, N=368)`)은 그대로 두고,
미래 horizon 전체를 한 번에 denoise하던 구조만 교체한다.

핵심 설계 (사용자 확정):
- **Causal temporal transformer** : 프레임 시퀀스 `[past_0..RF-1, fut_0..FH-1]` 위에서
  미래 프레임 k는 과거 RF 프레임 + 자기보다 앞선 미래 프레임(0..k-1)만 본다(frame-level
  causal). 자기 자신은 noisy 토큰으로만 본다(spatial self-attn).
- **Teacher forcing** : 학습 시 컨텍스트로 들어가는 이전 미래 프레임은 항상 **clean GT**
  latent. 덕분에 frame-level causal mask 하나로 모든 미래 step을 **한 번의 forward로
  병렬 학습**한다(노이즈 query 토큰 ↔ clean memory 토큰 분리).
- **Per-step conditioning** : 각 미래 step k는 그 step의 action(`action[k]`)과 speed를
  따로 조건으로 받는다(WM은 첫 future action 하나만 썼다).

추론은 frame-by-frame 순차 rollout:
  step k에서 [past + 이미 생성된 future 0..k-1] 을 clean memory로 두고
  미래 프레임 1개를 DDIM으로 denoise → 다음 step의 컨텍스트로 누적.
비용은 joint 대비 약 FH배(=FH × n_steps denoiser pass).

DiTBlock은 WM과 동일하게 self-attn(spatial) + cross-attn(temporal memory) +
adaLN-Zero 구조를 재사용하되, (1) attention mask와 (2) per-token(프레임별) 조건 c를
받도록 일반화했다.
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

    AR에서는 미래 프레임마다 독립 timestep을 쓰므로 `_gather`가 (B,) 와 (B,K)
    timestep을 모두 받아 z(=`(B,K,C,N)`)에 broadcast되는 (B,K,1,1)을 돌려준다.
    """

    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        super().__init__()
        self.timesteps = int(timesteps)
        steps = torch.arange(self.timesteps + 1, dtype=torch.float64) / self.timesteps
        f = torch.cos((steps + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = (f / f[0]).clamp(1e-8, 1.0)  # (T+1,), ᾱ at integer steps
        self.register_buffer("alpha_bar", alpha_bar.float(), persistent=False)

    def _gather(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ_t → (B,K,1,1). t는 (B,) 또는 (B,K)."""
        ab = self.alpha_bar.to(t.device)[t.clamp(0, self.timesteps).long()]
        if ab.ndim == 1:        # (B,) → (B,1,1,1)
            return ab.view(-1, 1, 1, 1)
        return ab.view(ab.shape[0], ab.shape[1], 1, 1)  # (B,K) → (B,K,1,1)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """forward diffusion: z_t = √ᾱ·z₀ + √(1−ᾱ)·ε. returns (z_t, v_target).
        z0/noise: (B,K,C,N), t: (B,K) (또는 (B,))."""
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
    """sinusoidal timestep embedding. t: (...,) → (..., dim). 임의 leading shape 허용."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs  # (..., half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ─────────────────────────────────────────────
# DiT block (self + cross attention, adaLN-Zero, masked)
# ─────────────────────────────────────────────
class DiTBlock(nn.Module):
    """future 토큰 self-attn(spatial) + clean-memory cross-attn(temporal) + MLP.

    WM 버전과 구조는 같으나 두 가지가 일반화됨:
    - **self-attn은 프레임별(intra-frame) 로컬 어텐션**: query 토큰을 (B*K, N, D)로
      reshape해 각 미래 프레임 안에서만 attend한다. 프레임 격리 mask를 통째로 없애
      (1) 프레임 간 블록 계산 낭비 제거 (2) float-mask로 인한 fused-attn 비활성 회피.
    - cross-attn은 clean memory에 대해 frame-level causal mask(cross_mask)로 제한.
    - 조건 벡터 c가 **per-token** (B, L, D) 로 들어옴(프레임별 timestep/action 반영).
      adaLN-Zero modulation Linear은 0 init → gate=0(identity)에서 시작.
    """

    def __init__(self, dim: int, n_heads: int, n_tokens: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.N = n_tokens                              # 프레임당 토큰 수 (self-attn reshape용)
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
        # x:(B,L,D), shift/scale:(B,L,D)  — per-token modulation.
        return x * (1 + scale) + shift

    def forward(self, x, memory, c, cross_mask=None):
        # c: (B, L, D) per-token 조건. modulation → (B, L, 9D).
        (sh_sa, sc_sa, g_sa,
         sh_ca, sc_ca, g_ca,
         sh_mlp, sc_mlp, g_mlp) = self.modulation(c).chunk(9, dim=-1)

        # self-attn: 프레임별 로컬. (B, K*N, D) → (B*K, N, D) → attend → 복원. mask 불필요.
        B, L, D = x.shape
        K = L // self.N
        h = self._mod(self.norm_sa(x), sh_sa, sc_sa)
        hf = h.reshape(B * K, self.N, D)
        af = self.self_attn(hf, hf, hf, need_weights=False)[0].reshape(B, L, D)
        x = x + g_sa * af

        h = self._mod(self.norm_ca(x), sh_ca, sc_ca)
        x = x + g_ca * self.cross_attn(h, memory, memory, attn_mask=cross_mask, need_weights=False)[0]

        h = self._mod(self.norm_mlp(x), sh_mlp, sc_mlp)
        x = x + g_mlp * self.mlp(h)
        return x


# ─────────────────────────────────────────────
# Frame-wise autoregressive latent diffusion model
# ─────────────────────────────────────────────
class LatentARDiffusion(nn.Module):
    """Causal AR DiT denoiser over future latent frames.

    loss(z_future_clean, z_past, actions, speed) : teacher-forcing v-MSE (병렬, 1 forward)
    sample(z_past, action, speed, ...)           : frame-by-frame AR rollout (DDIM+CFG)

    latent token: encoder 융합 `(B, S, C, N)`. 미래 프레임 k는 과거 + future 0..k-1
    (둘 다 clean) 을 cross-attn memory로 보고, 자기 noisy 토큰끼리 spatial self-attn.
    """

    def __init__(
        self,
        token_channels: int = 256,    # C
        n_tokens: int = 368,          # N (image 240 + lidar 128)
        action_dim: int = 2,
        receptive_field: int = 4,     # RF (past frames)
        future_horizon: int = 4,      # FH (future frames, AR denoised)
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
        self.n_heads = n_heads

        self.schedule = CosineNoiseSchedule(timesteps=timesteps, s=schedule_s)

        # token C ↔ model dim D projections.
        self.in_proj = nn.Linear(self.C, self.D)     # noisy future query
        self.past_proj = nn.Linear(self.C, self.D)   # clean memory (past + future ctx)
        self.out_proj = nn.Linear(self.D, self.C)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # learned positional embeddings: (frame, token) 분리.
        # frame pos는 절대 프레임 인덱스(0..RF+FH-1)로 부여 → past/future 일관.
        self.frame_pos = nn.Parameter(torch.zeros(1, self.RF + self.FH, 1, self.D))
        self.token_pos = nn.Parameter(torch.zeros(1, 1, self.N, self.D))
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        nn.init.trunc_normal_(self.token_pos, std=0.02)

        # conditioning embeddings → D (모두 합산).
        self.t_embed = nn.Sequential(nn.Linear(self.D, self.D), nn.SiLU(), nn.Linear(self.D, self.D))
        self.action_embed = nn.Sequential(nn.Linear(action_dim, self.D), nn.SiLU(), nn.Linear(self.D, self.D))
        self.speed_embed = nn.Sequential(nn.Linear(1, self.D), nn.SiLU(), nn.Linear(self.D, self.D))
        # CFG용 null-action 임베딩.
        self.null_action = nn.Parameter(torch.zeros(self.D))

        self.blocks = nn.ModuleList([
            DiTBlock(self.D, n_heads, self.N, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(self.D, elementwise_affine=False, eps=1e-6)

        # ── 학습용 정적 cross-attn mask (RF,FH,N에만 의존) ──
        # cross_mask: 미래 query 프레임 k → clean memory 프레임 f<RF+k 만. (FH*N, (RF+FH)*N)
        # (self-attn은 DiTBlock이 프레임별 reshape로 처리 → 별도 mask 불필요.)
        self.register_buffer("cross_mask", self._build_cross_mask(), persistent=False)

    # ── mask 빌더 (float additive: 0 허용 / -inf 차단) ──
    def _build_cross_mask(self):
        RF, FH, N = self.RF, self.FH, self.N
        qf = torch.arange(FH).repeat_interleave(N)               # (FH*N,) 미래 query, global frame = RF+qf
        mf = torch.arange(RF + FH).repeat_interleave(N)          # ((RF+FH)*N,) memory global frame
        allow = mf[None, :] < (RF + qf)[:, None]                 # causal: memory frame < query global frame
        m = torch.zeros(FH * N, (RF + FH) * N)
        m.masked_fill_(~allow, float("-inf"))
        return m

    # ── conditioning ──────────────────────────
    def _cond(self, t, action, speed, drop_action=None):
        """프레임별 조건 c. 반환 (B, K, D).
        t: (B,K) | (B,), action: (B,K,A) | (B,A) | None, speed: (B,) | None.
        K(=프레임 수)는 t로부터 추론(학습 K=FH, 추론 K=1)."""
        if t.ndim == 1:
            t = t.unsqueeze(1)                        # (B,1)
        B, K = t.shape
        c = self.t_embed(timestep_embedding(t, self.D))   # (B,K,D)

        if action is None:
            a = self.null_action.view(1, 1, -1).expand(B, K, -1)
        else:
            if action.ndim == 2:                      # (B,A) → (B,K,A)
                action = action.unsqueeze(1).expand(B, K, -1)
            a = self.action_embed(action.float())     # (B,K,D)
            if drop_action is not None:               # (B,) bool → 해당 샘플 전 프레임 null
                a = torch.where(drop_action.view(B, 1, 1), self.null_action.view(1, 1, -1).expand_as(a), a)
        c = c + a
        if speed is not None:
            sp = speed.float()
            if sp.ndim == 1:
                sp = sp.unsqueeze(-1)                  # (B,1)
            c = c + self.speed_embed(sp).unsqueeze(1)  # (B,1,D) broadcast over K
        return c

    def _embed_memory(self, z_clean, frame_offset):
        """clean latent (B, K, C, N) → memory 토큰 (B, K*N, D). frame_offset: 절대 프레임 시작 인덱스."""
        B, K = z_clean.shape[0], z_clean.shape[1]
        x = z_clean.permute(0, 1, 3, 2)               # (B,K,N,C)
        x = self.past_proj(x)                         # (B,K,N,D)
        x = x + self.frame_pos[:, frame_offset:frame_offset + K] + self.token_pos
        return x.reshape(B, K * self.N, self.D)

    def _embed_query(self, z_noisy, frame_offset):
        """noisy future latent (B, K, C, N) → query 토큰 (B, K*N, D)."""
        B, K = z_noisy.shape[0], z_noisy.shape[1]
        x = z_noisy.permute(0, 1, 3, 2)               # (B,K,N,C)
        x = self.in_proj(x)                           # (B,K,N,D)
        x = x + self.frame_pos[:, frame_offset:frame_offset + K] + self.token_pos
        return x.reshape(B, K * self.N, self.D)

    def _run_blocks(self, query_x, memory, c_frame, cross_mask):
        """query_x:(B,K*N,D), memory:(B,M*N,D), c_frame:(B,K,D). → v_pred (B,K,C,N)."""
        B = query_x.shape[0]
        K = c_frame.shape[1]
        c = c_frame.repeat_interleave(self.N, dim=1)  # (B,K*N,D) per-token 조건
        x = query_x
        for block in self.blocks:
            x = block(x, memory, c, cross_mask=cross_mask)
        x = self.out_proj(self.norm_out(x))           # (B,K*N,C)
        x = x.reshape(B, K, self.N, self.C).permute(0, 1, 3, 2)
        return x                                       # (B,K,C,N)

    # ── training (parallel teacher forcing) ───
    def forward(self, z_future_noisy, t, z_past, z_future_clean, actions, speed, drop_action=None):
        """teacher-forcing v_pred 반환 (B,FH,C,N).

        memory = [past(clean) ; future_clean(GT)] → causal mask로 프레임 k는 f<RF+k 만 본다.
        z_future_noisy: (B,FH,C,N), t: (B,FH).
        """
        mem_past = self._embed_memory(z_past, frame_offset=0)                    # RF
        mem_fut = self._embed_memory(z_future_clean, frame_offset=self.RF)       # FH (teacher ctx)
        memory = torch.cat([mem_past, mem_fut], dim=1)                           # (B,(RF+FH)*N,D)
        query = self._embed_query(z_future_noisy, frame_offset=self.RF)          # (B,FH*N,D)
        c = self._cond(t, actions, speed, drop_action=drop_action)               # (B,FH,D)
        return self._run_blocks(query, memory, c, self.cross_mask)

    def loss(self, z_future_clean, z_past, actions, speed, p_drop=0.0, generator=None, weight=None):
        """teacher-forcing v-MSE. 프레임별 독립 t~U(1,T), ε~N(0,I).

        actions: (B,FH,A) 권장(미래 step별). (B,A)면 전 step 동일 action으로 broadcast.
        weight(B,): per-sample 가중평균(정지 프레임 다운웨이팅용).
        """
        B = z_future_clean.shape[0]
        device = z_future_clean.device
        t = torch.randint(1, self.schedule.timesteps + 1, (B, self.FH), device=device, generator=generator)
        noise = torch.randn(z_future_clean.shape, device=device, generator=generator)
        z_t, v_target = self.schedule.q_sample(z_future_clean, t, noise)
        drop = None
        if p_drop > 0:
            drop = torch.rand(B, device=device, generator=generator) < p_drop
        v_pred = self.forward(z_t, t, z_past, z_future_clean, actions, speed, drop_action=drop)
        if weight is None:
            return F.mse_loss(v_pred, v_target)
        per = F.mse_loss(v_pred, v_target, reduction="none").flatten(1).mean(dim=1)  # (B,)
        w = weight.to(per.dtype)
        return (per * w).sum() / w.sum().clamp_min(1e-8)

    # ── inference (AR rollout, DDIM + CFG) ────
    @torch.no_grad()
    def _denoise_step(self, z_noisy, t, memory, action, speed, drop_action=None):
        """단일 미래 프레임 denoise pass. z_noisy:(B,1,C,N), memory:(B,M*N,D) (clean, 모두 과거).

        memory가 전부 현재 프레임보다 과거이므로 cross-attn mask 불필요(전부 허용).
        self-attn은 단일 프레임이라 mask 불필요.
        """
        frame_idx = memory.shape[1] // self.N          # 현재 프레임의 절대 인덱스 = memory 프레임 수
        query = self._embed_query(z_noisy, frame_offset=frame_idx)   # (B,N,D)
        c = self._cond(t, action, speed, drop_action=drop_action)    # (B,1,D)
        v = self._run_blocks(query, memory, c, None)                 # (B,1,C,N)
        return v

    @torch.no_grad()
    def sample(self, z_past, action, speed, n_steps=50, cfg_scale=1.0, generator=None):
        """frame-by-frame AR DDIM 샘플링(eta=0) → z_future_hat (B, FH, C, N).

        action: (B,FH,A) 권장(미래 step별). (B,A)면 전 step 동일 action.
        각 미래 프레임을 n_steps DDIM으로 복원 후 다음 프레임의 clean memory로 누적.
        cfg_scale<=1 이면 uncond pass 생략. seed 제어는 generator.
        """
        device = z_past.device
        B = z_past.shape[0]
        if action is not None and action.ndim == 2:
            action = action.unsqueeze(1).expand(B, self.FH, -1)   # (B,FH,A)

        T = self.schedule.timesteps
        seq = torch.linspace(T, 1, steps=n_steps, device=device).round().long()
        seq = torch.unique_consecutive(seq)
        use_cfg = cfg_scale is not None and cfg_scale != 1.0

        memory = self._embed_memory(z_past, frame_offset=0)       # (B, RF*N, D) clean
        generated = []                                            # list of (B,1,C,N) clean

        for k in range(self.FH):
            a_k = action[:, k] if action is not None else None    # (B,A) | None
            z = torch.randn(B, 1, self.C, self.N, device=device, generator=generator)
            for i in range(len(seq)):
                t = seq[i].expand(B)
                t_prev = seq[i + 1].expand(B) if i + 1 < len(seq) else torch.zeros(B, dtype=torch.long, device=device)
                v_cond = self._denoise_step(z, t, memory, a_k, speed)
                if use_cfg:
                    v_uncond = self._denoise_step(z, t, memory, None, speed)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
                z0, eps = self.schedule.predict_start_and_noise(z, t, v)
                ab_prev = self.schedule._gather(t_prev)
                z = ab_prev.sqrt() * z0 + (1.0 - ab_prev).sqrt() * eps
            generated.append(z)                                   # clean z0 (t_prev=0)
            # 다음 프레임 컨텍스트로 누적 (절대 프레임 인덱스 = RF+k).
            mem_k = self._embed_memory(z, frame_offset=self.RF + k)
            memory = torch.cat([memory, mem_k], dim=1)

        return torch.cat(generated, dim=1)                        # (B, FH, C, N)

    # ── config 헬퍼 ───────────────────────────
    @classmethod
    def from_cfg(cls, cfg):
        """cfg.DIFFUSION(없으면 default)에서 하이퍼파라미터를 읽어 생성. WM과 동일 키."""
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
            future_horizon=getattr(cfg, "FUTURE_HORIZON", 4),
            hidden_dim=int(getattr(d, "HIDDEN_DIM", C)),
            depth=int(getattr(d, "DEPTH", 6)),
            n_heads=int(getattr(d, "N_HEADS", 8)),
            mlp_ratio=float(getattr(d, "MLP_RATIO", 4.0)),
            dropout=float(getattr(d, "DROPOUT", 0.0)),
            timesteps=int(getattr(d, "TIMESTEPS", 1000)),
            schedule_s=float(getattr(d, "SCHEDULE_S", 0.008)),
        )


# WM과의 import 호환을 위한 별칭 (trainer/predict는 이 이름을 그대로 import 가능).
LatentFutureDiffusion = LatentARDiffusion
