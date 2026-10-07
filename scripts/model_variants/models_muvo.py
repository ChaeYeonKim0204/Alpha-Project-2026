#고친이후
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config_muvo import cfg


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def sigmoid2(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
    return 2 * torch.sigmoid(tensor / 2) + min_value


# ─────────────────────────────────────────────
# Image Encoder: FPN
# ─────────────────────────────────────────────

class FPNDecoder(nn.Module):
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
# Transformer Fusion
# ─────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
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

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SensorFusionTransformer(nn.Module):
    def __init__(self, channels, num_layers=3, nhead=8, dropout=0.1):
        super().__init__()
        self.channels        = channels
        self.position_encode = PositionEmbeddingSine(num_pos_feats=channels // 2, normalize=True)
        self.type_embedding  = nn.Parameter(torch.zeros(2, channels))

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

    def forward(self, cam_feat, lidar_feat):
        B, C, Hc, Wc = cam_feat.shape
        _, _, Hl, Wl = lidar_feat.shape
        L_cam = Hc * Wc

        cam_tokens   = self._add_pos_and_type(cam_feat,   sensor_type_idx=0)
        lidar_tokens = self._add_pos_and_type(lidar_feat, sensor_type_idx=1)

        fused = torch.cat([cam_tokens, lidar_tokens], dim=0)
        out   = self.transformer_encoder(fused)

        cam_out   = out[:L_cam].permute(1, 2, 0).reshape(B, C, Hc, Wc)
        lidar_out = out[L_cam:].permute(1, 2, 0).reshape(B, C, Hl, Wl)
        return cam_out, lidar_out


class BasicBlock(nn.Module):
    """muvo/muvo/layers/layers.py:BasicBlock inline 이식 — ResNet basic block.

    Skip connection이 있어 gradient flow와 학습 안정성에 도움. upstream muvo의
    layers 패키지에 의존하지 않기 위해 알파 내부에 다시 구현.
    `stride>1`이나 `inplanes!=planes`일 때 downsample=True로 1×1 conv 우회로 추가.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.act2  = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        shortcut = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.act2(out + shortcut)


class SensorFeatureConv(nn.Module):
    """upstream muvo/muvo/models/mile.py:104-115 패턴.

    BasicBlock × 2 (residual) + AdaptiveAvgPool + Flatten. Transformer 출력
    spatial feature를 한 프레임당 vector 한 개로 압축한다.
    이전: plain Conv-BN-ReLU × 2 (skip connection 없음). 이제는 upstream과 동일.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            BasicBlock(in_channels,  out_channels, stride=2, downsample=True),
            BasicBlock(out_channels, out_channels),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# RSSM
# ─────────────────────────────────────────────

class RepresentationModel(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std    = 0.1
        self.module = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(True),
            nn.Linear(in_channels, 2 * self.latent_dim),
        )

    def forward(self, x):
        mu_log_sigma = self.module(x)
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)
        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class RSSM(nn.Module):
    """Recurrent State Space Model.

    `action_in_gru` 플래그로 두 가지 디자인 사이 전환 가능:

    * True (default, alpha 디자인) — `pre_gru_net([sample_t, latent_action_t])`.
      action이 GRU hidden state 진화에도 영향. 표준 Dreamer V1/V2/V3 패턴.
    * False (upstream muvo) — `pre_gru_net(sample_t)`. action은 GRU 이후 prior
      분포에만 영향. upstream `muvo/muvo/models/transition.py:163`과 일치.

    alpha의 default(True)는 표준에 더 부합하지만, `_muvo.py`가 upstream을 최대한
    살리는 변종이라 향후 ablation으로 False 비교 여지를 남긴다.
    """

    def __init__(self, embedding_dim, action_dim, hidden_state_dim, state_dim,
                 action_latent_dim, receptive_field, use_dropout=False, dropout_probability=0.0,
                 action_in_gru=True):
        super().__init__()
        self.embedding_dim      = embedding_dim
        self.state_dim          = state_dim
        self.action_dim         = action_dim
        self.hidden_state_dim   = hidden_state_dim
        self.action_latent_dim  = action_latent_dim
        self.receptive_field    = receptive_field
        self.use_dropout        = use_dropout
        self.dropout_probability = dropout_probability
        self.action_in_gru      = bool(action_in_gru)

        pre_gru_input_dim = state_dim + (action_latent_dim if self.action_in_gru else 0)
        self.pre_gru_net = nn.Sequential(
            nn.Linear(pre_gru_input_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )
        self.recurrent_model = nn.GRUCell(input_size=hidden_state_dim, hidden_size=hidden_state_dim)

        self.posterior_action_module = nn.Sequential(
            nn.Linear(action_dim, action_latent_dim), nn.LeakyReLU(True),
        )
        self.posterior = RepresentationModel(
            in_channels=hidden_state_dim + embedding_dim + action_latent_dim,
            latent_dim=state_dim,
        )
        self.prior_action_module = nn.Sequential(
            nn.Linear(action_dim, action_latent_dim), nn.LeakyReLU(True),
        )
        self.prior = RepresentationModel(
            in_channels=hidden_state_dim + action_latent_dim,
            latent_dim=state_dim,
        )

    def forward(self, input_embedding, action, h_init=None, s_init=None,
                use_sample=True, policy=None, continuation=False, init_action=None):
        output = {"prior": [], "posterior": []}
        batch_size, sequence_length, _ = input_embedding.shape

        h_t      = h_init if h_init is not None else input_embedding.new_zeros((batch_size, self.hidden_state_dim))
        sample_t = s_init if s_init is not None else input_embedding.new_zeros((batch_size, self.state_dim))

        for t in range(sequence_length):
            if t == 0:
                if init_action is not None:
                    action_t = init_action
                elif torch.is_tensor(continuation):
                    zero_a   = torch.zeros_like(action[:, 0])
                    action_t = torch.where(continuation.unsqueeze(-1), action[:, 0], zero_a)
                elif continuation:
                    action_t = action[:, 0]
                else:
                    action_t = torch.zeros_like(action[:, 0])
            else:
                if not torch.is_tensor(continuation) and continuation:
                    action_t = action[:, t]
                else:
                    action_t = action[:, t - 1]

            output_t = self.observe_step(h_t, sample_t, action_t, input_embedding[:, t],
                                         use_sample=use_sample, policy=policy)
            use_prior = (
                self.training and self.use_dropout
                and torch.rand(1).item() < self.dropout_probability and t > 0
            )
            sample_t = output_t["prior"]["sample"] if use_prior else output_t["posterior"]["sample"]
            h_t      = output_t["prior"]["hidden_state"]
            for key, value in output_t.items():
                output[key].append(value)

        return self.stack_list_of_dict_tensor(output, dim=1)

    def observe_step(self, h_t, sample_t, action_t, embedding_t, use_sample=True, policy=None):
        imagine_output   = self.imagine_step(h_t, sample_t, action_t, use_sample, policy=policy)
        latent_action_t  = self.posterior_action_module(action_t)
        post_mu, post_sigma = self.posterior(
            torch.cat([imagine_output["hidden_state"], embedding_t, latent_action_t], dim=-1)
        )
        sample_t = self.sample_from_distribution(post_mu, post_sigma, use_sample)
        return {
            "prior": imagine_output,
            "posterior": {
                "hidden_state": imagine_output["hidden_state"],
                "sample": sample_t,
                "mu":     post_mu,
                "sigma":  post_sigma,
            },
        }

    def imagine_step(self, h_t, sample_t, action_t, use_sample=True, policy=None):
        if policy is not None:
            action_t = policy(torch.cat([h_t, sample_t], dim=-1))

        latent_action_t = self.prior_action_module(action_t)
        if self.action_in_gru:
            gru_input = torch.cat([sample_t, latent_action_t], dim=-1)
        else:
            gru_input = sample_t
        h_t             = self.recurrent_model(self.pre_gru_net(gru_input), h_t)
        prior_mu, prior_sigma = self.prior(torch.cat([h_t, latent_action_t], dim=-1))
        sample_t        = self.sample_from_distribution(prior_mu, prior_sigma, use_sample)
        return {"hidden_state": h_t, "sample": sample_t, "mu": prior_mu, "sigma": prior_sigma}

    @staticmethod
    def sample_from_distribution(mu, sigma, use_sample):
        return mu + sigma * torch.randn_like(mu) if use_sample else mu

    @staticmethod
    def stack_list_of_dict_tensor(output, dim=1):
        new_output = {}
        for outer_key, outer_value in output.items():
            if outer_value:
                new_output[outer_key] = {
                    inner_key: torch.stack([x[inner_key] for x in outer_value], dim=dim)
                    for inner_key in outer_value[0].keys()
                }
        return new_output


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

def _decoder_base_size(target_size, n_pre_doublings=3):
    """target_size = constant_size × 2^(n_pre_doublings + 3 trans_conv stages).

    For 6-doubling (default): target / 64 must be integer in both dims.
    For 5-doubling (n_pre_doublings=2): target / 32 must be integer.
    """
    h, w = target_size
    scale = 2 ** (n_pre_doublings + 3)
    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"target_size={target_size} is not divisible by 2^{n_pre_doublings + 3}={scale}. "
            f"Either change target_size or n_pre_doublings."
        )
    return (h // scale, w // scale)


def _auto_n_pre_doublings(target_size, max_n=3, min_n=0):
    """Pick largest n in [min_n, max_n] such that target / 2^(n+3) is positive integer."""
    h, w = target_size
    for n in range(max_n, min_n - 1, -1):
        scale = 2 ** (n + 3)
        if h % scale == 0 and w % scale == 0 and (h // scale) >= 1 and (w // scale) >= 1:
            return n
    raise ValueError(f"No valid n_pre_doublings in [{min_n}, {max_n}] for target_size={target_size}.")


class SensorHead(nn.Module):
    """1x1 Conv only. Decoder owns exact output resolution; no bilinear fallback."""

    def __init__(self, in_channels, out_channels, downsample_factor, sensor_type):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.sensor_type       = sensor_type
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, B, S):
        out = self.head(x)
        _, C, H, W = out.shape
        return {f'{self.sensor_type}_{self.downsample_factor}': out.view(B, S, C, H, W)}


class SensorDecoder(nn.Module):
    """Upstream `muvo.models.common.ConvDecoder` 패턴 (common.py:549-632) 이식.

    구조: linear → pre_transpose (1 constant_size + N doublings) → 3 trans_conv + heads.

    총 stride-2 doublings = n_pre_doublings + 3 (trans_conv1/2/3, each feeds a head).
    target_size = constant_size × 2^(n_pre_doublings + 3) 이어야 한다.
    n_pre_doublings=None이면 target에 맞는 최대 n (≤3)을 자동 선택.

    채널 progression: decoder_channels → /2 → /4 → /8 (upstream과 동일).
    Heads는 SensorHead (1×1 conv only); decoder가 정확한 해상도를 만들어 옴.
    """

    def __init__(self, latent_n_channels, out_channels, target_size, sensor_type,
                 decoder_channels=512, n_pre_doublings=None):
        super().__init__()
        self.target_size      = tuple(target_size)
        if n_pre_doublings is None:
            n_pre_doublings = _auto_n_pre_doublings(self.target_size, max_n=3, min_n=0)
        self.n_pre_doublings  = int(n_pre_doublings)
        constant_size         = _decoder_base_size(self.target_size, n_pre_doublings=self.n_pre_doublings)

        mid_ch  = decoder_channels // 2     # → head_4 input
        low_ch  = decoder_channels // 4     # → head_2 input
        tiny_ch = decoder_channels // 8     # → head_1 input

        self.linear = nn.Sequential(
            nn.Linear(latent_n_channels, decoder_channels),
            nn.Unflatten(-1, (decoder_channels, 1, 1)),
        )

        # pre_transpose: 1 layer creates constant_size from (1,1), then n_pre_doublings ×2 upsamples.
        # Upstream pattern: early doublings use k=5/p=2/op=1, the last uses k=6/p=2.
        pre_layers = [
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=constant_size),
            nn.ELU(),
        ]
        for i in range(self.n_pre_doublings):
            is_last = (i == self.n_pre_doublings - 1)
            if is_last:
                pre_layers += [
                    nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=6, stride=2, padding=2),
                    nn.ELU(),
                ]
            else:
                pre_layers += [
                    nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.ELU(),
                ]
        self.pre_transpose_conv = nn.Sequential(*pre_layers)

        # trans_conv1/2/3: ×2 upsample + halve channels, each feeds a head.
        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, mid_ch, kernel_size=6, stride=2, padding=2),
            nn.ELU(),
        )
        self.head_4 = SensorHead(mid_ch, out_channels, downsample_factor=4, sensor_type=sensor_type)

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, low_ch, kernel_size=6, stride=2, padding=2),
            nn.ELU(),
        )
        self.head_2 = SensorHead(low_ch, out_channels, downsample_factor=2, sensor_type=sensor_type)

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(low_ch, tiny_ch, kernel_size=6, stride=2, padding=2),
            nn.ELU(),
        )
        self.head_1 = SensorHead(tiny_ch, out_channels, downsample_factor=1, sensor_type=sensor_type)

    def forward(self, hidden_state, sample):
        B, S, _ = hidden_state.shape
        x = torch.cat([hidden_state, sample], dim=-1).reshape(B * S, -1)
        x = self.linear(x)
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
        self.cfg                  = cfg
        if embedding_n_channels is None:
            embedding_n_channels = getattr(cfg.MODEL, "EMBEDDING_DIM", 256)
        transformer_channels = getattr(cfg.MODEL.FUSION, "TRANSFORMER_CHANNELS", embedding_n_channels)
        self.embedding_n_channels = embedding_n_channels
        self.transformer_channels = transformer_channels
        self.receptive_field      = cfg.RECEPTIVE_FIELD

        self.image_encoder = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=[2, 3, 4])
        self.image_fpn     = FPNDecoder([128, 256, 512], out_channels=transformer_channels)

        self.lidar_encoder = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=[2, 3, 4], in_chans=4)
        self.lidar_fpn     = FPNDecoder([128, 256, 512], out_channels=transformer_channels)

        self.transformer_encoder = (transformer_encoder if transformer_encoder is not None
                                    else SensorFusionTransformer(
                                        channels=transformer_channels,
                                        num_layers=getattr(cfg.MODEL.FUSION, "TRANSFORMER_LAYERS", 3),
                                        nhead=getattr(cfg.MODEL.FUSION, "TRANSFORMER_HEADS", 8),
                                        dropout=getattr(cfg.MODEL.FUSION, "TRANSFORMER_DROPOUT", 0.1),
                                    ))

        self.image_feature_compress = SensorFeatureConv(transformer_channels, embedding_n_channels)
        self.lidar_feature_compress = SensorFeatureConv(transformer_channels, embedding_n_channels)

        speed_channels        = getattr(cfg.MODEL, "SPEED_CHANNELS", 16)
        self.speed_normalisation = getattr(cfg.MODEL, "SPEED_NORMALISATION", 50.0)
        self.speed_encoder    = nn.Sequential(
            nn.Linear(1, speed_channels), nn.ReLU(True),
            nn.Linear(speed_channels, speed_channels), nn.ReLU(True),
        )
        self.features_combine = nn.Linear(2 * embedding_n_channels + speed_channels, embedding_n_channels)

        latent_dim = cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + cfg.MODEL.TRANSITION.STATE_DIM

        self.image_decoder = SensorDecoder(
            latent_n_channels=latent_dim, out_channels=3,
            target_size=tuple(cfg.DATA.RGB_RECON_SIZE), sensor_type='rgb',
        )
        self.lidar_decoder = SensorDecoder(
            latent_n_channels=latent_dim, out_channels=4,
            target_size=tuple(cfg.DATA.LIDAR_RANGE_VIEW_SIZE), sensor_type='lidar_reconstruction',
        )

        if not cfg.MODEL.TRANSITION.ENABLED:
            raise ValueError("cfg.MODEL.TRANSITION.ENABLED must be True.")

        self.rssm = RSSM(
            embedding_dim=embedding_n_channels,
            action_dim=cfg.MODEL.ACTION_DIM,
            hidden_state_dim=cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
            state_dim=cfg.MODEL.TRANSITION.STATE_DIM,
            action_latent_dim=cfg.MODEL.TRANSITION.ACTION_LATENT_DIM,
            receptive_field=self.receptive_field,
            use_dropout=cfg.MODEL.TRANSITION.USE_DROPOUT,
            dropout_probability=cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
            action_in_gru=getattr(cfg.MODEL.TRANSITION, "ACTION_IN_GRU", True),
        )

    def encode_fuse_sequence(self, image, lidar, speed=None):
        B, S, C_img,   H_img,   W_img   = image.shape
        _,  _, C_lidar, H_lidar, W_lidar = lidar.shape

        image_flat = image.reshape(B * S, C_img,   H_img,   W_img)
        lidar_flat = lidar.reshape(B * S, C_lidar, H_lidar, W_lidar)

        cam_feat   = self.image_fpn(self.image_encoder(image_flat))
        lidar_feat = self.lidar_fpn(self.lidar_encoder(lidar_flat))

        cam_out, lidar_out = self.transformer_encoder(cam_feat, lidar_feat)
        cam_emb   = self.image_feature_compress(cam_out).reshape(B, S, -1)
        lidar_emb = self.lidar_feature_compress(lidar_out).reshape(B, S, -1)

        if speed is None:
            speed = image.new_zeros(B, S)
        speed = speed.float().to(image.device)
        if speed.ndim == 2:
            speed = speed.unsqueeze(-1)
        speed_emb = self.speed_encoder(
            speed.reshape(B * S, 1) / self.speed_normalisation
        ).reshape(B, S, -1)

        return self.features_combine(torch.cat([cam_emb, lidar_emb, speed_emb], dim=-1))

    def forward(self, batch, h_init=None, s_init=None,
                deployment=False, continuation=False, init_action=None):
        image = batch["image"].float()
        lidar = batch["lidar"].float()
        speed = batch.get("speed", None)

        if "action" in batch:
            action = batch["action"].float()
        elif "throttle_brake" in batch and "steering" in batch:
            action = torch.cat([batch["throttle_brake"], batch["steering"]], dim=-1).float()
        else:
            raise KeyError("batch must contain 'action' or both 'throttle_brake' and 'steering'.")

        embedding_seq = self.encode_fuse_sequence(image, lidar, speed=speed)
        rssm_output   = self.rssm(
            input_embedding=embedding_seq,
            action=action,
            h_init=h_init, s_init=s_init,
            use_sample=True, policy=None,
            continuation=continuation,
            init_action=init_action,
        )

        output = {"prior": rssm_output["prior"], "posterior": rssm_output["posterior"]}
        ht = rssm_output["posterior"]["hidden_state"]
        st = rssm_output["posterior"]["sample"]
        if hasattr(self, "image_decoder"):
            output.update(self.image_decoder(ht, st))
        if hasattr(self, "lidar_decoder"):
            output.update(self.lidar_decoder(ht, st))
        return output, rssm_output

    def imagine(self, state_imagine, future_horizon=None):
        h_t      = state_imagine["hidden_state"]
        sample_t = state_imagine["sample"]
        device   = h_t.device

        if "action" in state_imagine:
            action = state_imagine["action"].float().to(device)
        elif "throttle_brake" in state_imagine and "steering" in state_imagine:
            action = torch.cat([
                state_imagine["throttle_brake"].float().to(device),
                state_imagine["steering"].float().to(device),
            ], dim=-1)
        else:
            raise KeyError("state_imagine must contain 'action' or both 'throttle_brake' and 'steering'.")

        if future_horizon is None:
            future_horizon = action.shape[1]
        future_horizon = min(future_horizon, action.shape[1])
        if future_horizon == 0:
            return {}, {}

        priors = {"hidden_state": [], "sample": [], "mu": [], "sigma": []}
        for t in range(future_horizon):
            prior_t  = self.rssm.imagine_step(h_t=h_t, sample_t=sample_t, action_t=action[:, t], use_sample=True, policy=None)
            h_t      = prior_t["hidden_state"]
            sample_t = prior_t["sample"]
            for key in priors:
                priors[key].append(prior_t[key])

        prior_seq = {k: torch.stack(v, dim=1) for k, v in priors.items()}
        output = {"prior": prior_seq}
        ht = prior_seq["hidden_state"]
        st = prior_seq["sample"]
        if hasattr(self, "image_decoder"):
            output.update(self.image_decoder(ht, st))
        if hasattr(self, "lidar_decoder"):
            output.update(self.lidar_decoder(ht, st))
        return output, {"prior": prior_seq}
