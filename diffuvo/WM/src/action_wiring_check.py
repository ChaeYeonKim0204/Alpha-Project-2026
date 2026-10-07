"""action_wiring_check.py — diffusion의 action 조건 경로가 실제로 작동하는지 5분 점검.

설계: past를 모든 샘플에서 동일하게 고정(정보 0) → 미래를 오직 action으로만 구분.
모델이 미래를 맞히려면 action을 쓸 수밖에 없는 합성 셋. tiny 학습 후
  - v-MSE→0 & sample이 action을 따라가면 → 배선 정상
  - 평균 수준에서 정체 & sample(a0)≈sample(a1) → action 경로 고장
"""
import torch
from config_Diffuvo import cfg
from diffusion_Diffuvo import LatentFutureDiffusion

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
diff = LatentFutureDiffusion.from_cfg(cfg).to(device)
C, N, RF, FH = diff.C, diff.N, diff.RF, diff.FH
B = 8
print(f"C={C} N={N} RF={RF} FH={FH} B={B} device={device}")

# 모든 샘플 동일한 past → past는 미래 구분에 무용.
z_past = torch.randn(1, RF, C, N, device=device).repeat(B, 1, 1, 1)
# 두 action, 두 개의 서로 다른 고정 미래. 절반씩.
a0 = torch.tensor([1.0, -1.0], device=device)
a1 = torch.tensor([-1.0, 1.0], device=device)
zf0 = torch.randn(1, FH, C, N, device=device)
zf1 = torch.randn(1, FH, C, N, device=device)
actions = torch.stack([a0] * (B // 2) + [a1] * (B // 2), 0)                       # (B,2)
z_future = torch.cat([zf0.repeat(B // 2, 1, 1, 1), zf1.repeat(B // 2, 1, 1, 1)], 0)
speed = torch.zeros(B, 1, device=device)

# 기준선: action 무시하고 두 미래의 평균만 예측할 때의 잔차 스케일(분리 성공 판정용).
mean_future = 0.5 * (zf0 + zf1)
ignore_floor = ((z_future - mean_future.repeat(B, 1, 1, 1)) ** 2).mean().item()
print(f"[baseline] action 무시(평균예측) 시 latent 잔차 ~ {ignore_floor:.4f}")

opt = torch.optim.AdamW(diff.parameters(), lr=1e-3)
diff.train()
for step in range(800):
    loss = diff.loss(z_future, z_past, actions, speed, p_drop=0.0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 799:
        print(f"step {step:4d}: v-MSE={loss.item():.4f}")

# action_embed로 gradient가 흐르는지(경로 연결 확인).
opt.zero_grad()
loss = diff.loss(z_future, z_past, actions, speed, p_drop=0.0)
loss.backward()
g_act = sum(p.grad.abs().sum().item() for p in diff.action_embed.parameters() if p.grad is not None)
print(f"[grad] action_embed grad L1 = {g_act:.4e}  (>0이어야 경로 연결)")

# 샘플이 action을 따라가는지(같은 seed로 a0/a1만 바꿔 비교).
diff.eval()
def relerr(pred, tgt):
    return ((pred - tgt) ** 2).mean().item() / (tgt ** 2).mean().item()
with torch.no_grad():
    g = torch.Generator(device=device).manual_seed(0)
    s0 = diff.sample(z_past[:1], a0[None], speed[:1], n_steps=50, cfg_scale=1.0, generator=g)
    g = torch.Generator(device=device).manual_seed(0)
    s1 = diff.sample(z_past[:1], a1[None], speed[:1], n_steps=50, cfg_scale=1.0, generator=g)
print(f"sample(a0) vs target0 relerr = {relerr(s0, zf0):.4f}  | vs target1 = {relerr(s0, zf1):.4f}")
print(f"sample(a1) vs target0 relerr = {relerr(s1, zf0):.4f}  | vs target1 = {relerr(s1, zf1):.4f}")
print(f"sample(a0) vs sample(a1) MSE = {((s0 - s1) ** 2).mean().item():.4f}  (클수록 action에 민감)")
