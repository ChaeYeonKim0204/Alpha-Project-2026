"""models_muvo_2D.py — MUVO 2D-token RSSM (RSSMTD) variant.

`models_muvo.py` 베이스 + 다음 신규 클래스 추가 + `Model` rewrite:
- `ConvGRUCellGlo` (`muvo_2d/.../transition_td.py:28-58`) — token-wise conv-GRU + global gate
- `RepresentationModelTD` (transition_td.py:61-124) — TransformerDecoder + 2-modality query
  (image / lidar; voxel/policy/3D 제거)
- `RSSMTD` (transition_td.py:127-301) — n_out_tokens = 240 + 128 = 368 (policy token 제거)
- `SpeedEncoder` (muvo.py:124-130) + type embedding modulation
- `TokenConvDecoder2D` — modality slice → ConvTranspose ladder + head_4/2/1

`Model` 변경:
- LiDAR encoder: `out_indices=cfg.MODEL.RSSM_2D.LIDAR_OUT_INDICES` = [1,2,3] (upstream MUVO 매칭)
- `SensorFeatureConv`/`features_combine` 제거 (token-shape state로 직접 전달)
- `encode_fuse_sequence` → `(B, S, C, n_tokens)` 반환
- `forward`/`imagine` → token-shape h/s (action 예측 제거, RGB/LiDAR만 디코드)
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config_Diffuvo import cfg


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def sigmoid2(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
    return 2 * torch.sigmoid(tensor / 2) + min_value


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
# Transformer Fusion (encoder side, alpha _muvo와 동일)
# ─────────────────────────────────────────────

class SensorFusionTransformer(nn.Module):
    """Image + LiDAR token fusion. type embedding 5 slot (image/lidar/policy/action/speed)
    이지만 fusion encoder는 image/lidar 2개만 input. policy/action/speed slot은
    RSSMTD 내부에서 사용.
    """

    def __init__(self, channels, num_layers=3, nhead=4, dropout=0.1):
        super().__init__()
        self.channels        = channels
        self.position_encode = PositionEmbeddingSine(num_pos_feats=channels // 2, normalize=True)
        # 5 slot: image=0, lidar=1, policy=2, action=3, speed=4. fusion encoder는 0/1만 사용.
        self.type_embedding  = nn.Parameter(torch.zeros(5, channels))
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

    def forward(self, cam_feat, lidar_feat, speed_embed=None):
        """speed_embed: (B, C) or None. None이면 speed modulation 건너뜀."""
        B, C, Hc, Wc = cam_feat.shape
        _, _, Hl, Wl = lidar_feat.shape
        L_cam   = Hc * Wc
        L_lidar = Hl * Wl

        cam_tokens   = self._add_pos_and_type(cam_feat,   sensor_type_idx=0)
        lidar_tokens = self._add_pos_and_type(lidar_feat, sensor_type_idx=1)

        tokens = [cam_tokens, lidar_tokens]
        # speed broadcast add — upstream `muvo.py:415-417` 패턴.
        # speed_embed (B, C)를 모든 token에 element-wise add (modulation).
        if speed_embed is not None:
            speed_token = speed_embed.unsqueeze(0) + self.type_embedding[4].view(1, 1, C)  # (1, B, C)
            tokens.append(speed_token)
        fused = torch.cat(tokens, dim=0)
        out   = self.transformer_encoder(fused)

        cam_out   = out[:L_cam].permute(1, 2, 0).reshape(B, C, Hc, Wc)
        lidar_out = out[L_cam:L_cam + L_lidar].permute(1, 2, 0).reshape(B, C, Hl, Wl)
        return cam_out, lidar_out


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
# 2D-token RSSM (RSSMTD)
# ─────────────────────────────────────────────

class ConvGRUCellGlo(nn.Module):
    """Token-wise Conv1d GRU + global gating (upstream `transition_td.py:28-58`)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv_z = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_r = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_q = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)

        self.w = nn.Conv1d(hidden_dim + input_dim, hidden_dim + input_dim, 1)

        self.conv_z_glo = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_r_glo = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)
        self.conv_q_glo = nn.Conv1d(hidden_dim + input_dim, hidden_dim, 1)

    def forward(self, h, x):
        h = h.permute(1, 2, 0)  # N, B, C -> B, C, N
        x = x.permute(1, 2, 0)

        hx = torch.cat([h, x], dim=1)
        glo = torch.sigmoid(self.w(hx)) * hx
        glo = glo.mean(dim=-1, keepdim=True)

        z = torch.sigmoid(self.conv_z(hx) + self.conv_z_glo(glo))
        r = torch.sigmoid(self.conv_r(hx) + self.conv_r_glo(glo))
        q = torch.tanh(self.conv_q(torch.cat([r * h, x], dim=1)) + self.conv_q_glo(glo))

        h = (1 - z) * h + z * q
        return h.permute(2, 0, 1)  # B, C, N -> N, B, C


class RepresentationModelTD(nn.Module):
    """upstream `transition_td.py:61-124` 축소 포팅: 2-modality query (image/lidar).

    voxel/policy slot 제거 → query_embed 2개 (image, lidar) + type_embeddings 2 slot.

    n_query_out = image_HW + lidar_HW = 368 by default (policy token 제거).
    """

    def __init__(self, in_channels, latent_dim,
                 image_token_hw=(10, 24), lidar_token_hw=(2, 64),
                 num_layers=3, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1
        self.image_token_hw = tuple(image_token_hw)
        self.lidar_token_hw = tuple(lidar_token_hw)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_channels, nhead=nhead
        )
        self.module = nn.TransformerDecoder(
            decoder_layer=self.transformer_decoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(in_channels, 2 * latent_dim)

        # 2 type slot: image=0, lidar=1 (policy query 제거로 action slot 불필요)
        self.type_embeddings = nn.Parameter(torch.zeros(1, 1, in_channels, 2))

        H_img, W_img = self.image_token_hw
        H_lid, W_lid = self.lidar_token_hw
        self.query_embed_image  = nn.Parameter(torch.zeros(1, in_channels, H_img, W_img))
        self.query_embed_lidar  = nn.Parameter(torch.zeros(1, in_channels, H_lid, W_lid))

        self.reset_parameters()

        self.pos_embedding = PositionEmbeddingSine(num_pos_feats=in_channels // 2, normalize=True)

    def reset_parameters(self):
        nn.init.uniform_(self.type_embeddings)
        nn.init.uniform_(self.query_embed_image)
        nn.init.uniform_(self.query_embed_lidar)

    def forward(self, x):
        bs = x.shape[1]

        # image queries: (1, C, H, W) -> add pos -> N, 1, C, then add type slot 0
        q_img = self.query_embed_image + self.pos_embedding(self.query_embed_image)
        q_img = q_img.flatten(2).permute(2, 0, 1) + self.type_embeddings[:, :, :, 0]

        q_lid = self.query_embed_lidar + self.pos_embedding(self.query_embed_lidar)
        q_lid = q_lid.flatten(2).permute(2, 0, 1) + self.type_embeddings[:, :, :, 1]

        query = torch.cat([q_img, q_lid], dim=0)  # (N_q, 1, C) — policy query 제거됨

        mu_log_sigma = self.fc(self.module(query.repeat(1, bs, 1), x))
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)
        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class RSSMTD(nn.Module):
    """Token-shape state RSSM (upstream `transition_td.py:127-301` 축소 포팅).

    n_out_tokens = H_img*W_img + H_lid*W_lid (default 240+128=368). policy token 제거됨.
    hidden_state_dim = state_dim = embedding_dim (= 256).

    Forward:
        input_embedding: (B, S, C, N_in) — encoder가 만든 token sequence
        action: (B, S, 2)
        returns: dict {"prior": {...}, "posterior": {...}} each with
                 hidden_state/sample/mu/sigma each shape (B, S, C, N_out)
    """

    def __init__(self, embedding_dim, action_dim, hidden_state_dim, state_dim,
                 image_token_hw=(10, 24), lidar_token_hw=(2, 64),
                 num_decoder_layers=3, nhead=4,
                 use_dropout=False, dropout_probability=0.0):
        super().__init__()
        assert embedding_dim == hidden_state_dim == state_dim, (
            f"RSSMTD requires embedding=hidden=state dim; got "
            f"{embedding_dim}/{hidden_state_dim}/{state_dim}"
        )
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim
        self.use_dropout = use_dropout
        self.dropout_probability = dropout_probability

        H_img, W_img = image_token_hw
        H_lid, W_lid = lidar_token_hw
        self.image_token_hw = (H_img, W_img)
        self.lidar_token_hw = (H_lid, W_lid)
        self.n_image_tokens = H_img * W_img
        self.n_lidar_tokens = H_lid * W_lid
        self.n_out_tokens = self.n_image_tokens + self.n_lidar_tokens

        # type embedding for input-stream tokens: 3 slot (embedding=0, hidden=1, action=2)
        self.type_embeddings = nn.Parameter(torch.zeros(1, 1, hidden_state_dim, 3))
        nn.init.uniform_(self.type_embeddings)

        self.pre_gru_net = nn.Sequential(
            nn.Linear(state_dim, hidden_state_dim),
            nn.LeakyReLU(True),
        )
        self.recurrent_model = ConvGRUCellGlo(input_dim=hidden_state_dim, hidden_dim=hidden_state_dim)

        self.posterior_action_module = nn.Sequential(
            nn.Linear(action_dim, hidden_state_dim), nn.LeakyReLU(True),
        )
        self.posterior = RepresentationModelTD(
            in_channels=hidden_state_dim, latent_dim=state_dim,
            image_token_hw=image_token_hw, lidar_token_hw=lidar_token_hw,
            num_layers=num_decoder_layers, nhead=nhead,
        )

        self.prior_action_module = nn.Sequential(
            nn.Linear(action_dim, hidden_state_dim), nn.LeakyReLU(True),
        )
        self.prior = RepresentationModelTD(
            in_channels=hidden_state_dim, latent_dim=state_dim,
            image_token_hw=image_token_hw, lidar_token_hw=lidar_token_hw,
            num_layers=num_decoder_layers, nhead=nhead,
        )

    def forward(self, input_embedding, action, use_sample=True, policy=None):
        """
        Inputs
        ------
            input_embedding: (B, S, C, N_in)
            action:          (B, S, action_dim)
        Returns
        -------
            output: {"prior": {...}, "posterior": {...}}
                each value dict has keys hidden_state/sample/mu/sigma
                each tensor shape (B, S, C, N_out)
        """
        output = {"prior": [], "posterior": []}
        batch_size, sequence_length, _, _ = input_embedding.shape

        h_t = input_embedding.new_zeros((self.n_out_tokens, batch_size, self.hidden_state_dim))
        sample_t = input_embedding.new_zeros((self.n_out_tokens, batch_size, self.state_dim))

        for t in range(sequence_length):
            if t == 0:
                action_t = torch.zeros_like(action[:, 0])
            else:
                action_t = action[:, t - 1]
            output_t = self.observe_step(
                h_t, sample_t, action_t, input_embedding[:, t],
                use_sample=use_sample, policy=policy,
            )
            use_prior = (
                self.training and self.use_dropout
                and torch.rand(1).item() < self.dropout_probability and t > 0
            )
            sample_t = output_t["prior"]["sample"] if use_prior else output_t["posterior"]["sample"]
            h_t = output_t["prior"]["hidden_state"]
            for key, value in output_t.items():
                output[key].append(value)

        return self.stack_list_of_dict_tensor(output, dim=1)

    def observe_step(self, h_t, sample_t, action_t, embedding_t, use_sample=True, policy=None):
        """One timestep: imagine_step (prior) → posterior via TransformerDecoder over
        [hidden, embedding, action] tokens.

        embedding_t: (B, C, N_in) — 1 frame from encoder
        """
        embedding_t = embedding_t.permute(2, 0, 1)  # (B, C, N_in) -> (N_in, B, C)
        imagine_output = self.imagine_step(h_t, sample_t, action_t, use_sample=use_sample, policy=policy)

        latent_action_t = self.posterior_action_module(action_t)[None]  # (1, B, C)
        embedding_t_tokens = embedding_t + self.type_embeddings[:, :, :, 0]
        latent_action_t_tokens = latent_action_t + self.type_embeddings[:, :, :, 2]

        posterior_mu, posterior_sigma = self.posterior(
            torch.cat([imagine_output["hidden_state"], embedding_t_tokens, latent_action_t_tokens], dim=0)
        )
        sample_t_new = self.sample_from_distribution(posterior_mu, posterior_sigma, use_sample=use_sample)

        posterior_output = {
            "hidden_state": imagine_output["hidden_state"],
            "sample": sample_t_new,
            "mu": posterior_mu,
            "sigma": posterior_sigma,
        }
        return {"prior": imagine_output, "posterior": posterior_output}

    def imagine_step(self, h_t, sample_t, action_t, use_sample=True, policy=None):
        latent_action_t = self.prior_action_module(action_t)[None]  # (1, B, C)

        input_t = self.pre_gru_net(sample_t)
        h_t_new = self.recurrent_model(h_t, input_t)
        h_t_tokens = h_t_new + self.type_embeddings[:, :, :, 1]
        latent_action_t_tokens = latent_action_t + self.type_embeddings[:, :, :, 2]

        prior_mu, prior_sigma = self.prior(torch.cat([h_t_tokens, latent_action_t_tokens], dim=0))
        sample_t_new = self.sample_from_distribution(prior_mu, prior_sigma, use_sample=use_sample)
        return {
            "hidden_state": h_t_new,
            "sample": sample_t_new,
            "mu": prior_mu,
            "sigma": prior_sigma,
        }

    @staticmethod
    def sample_from_distribution(mu, sigma, use_sample):
        return mu + sigma * torch.randn_like(mu) if use_sample else mu

    @staticmethod
    def stack_list_of_dict_tensor(output, dim=1):
        """Each per-timestep entry has shape (N, B, C); we permute → (B, C, N)
        then stack along sequence dim to get (B, S, C, N).
        """
        new_output = {}
        for outer_key, outer_value in output.items():
            if outer_value:
                new_output[outer_key] = {
                    inner_key: torch.stack(
                        [x[inner_key].permute(1, 2, 0) for x in outer_value], dim=dim
                    )
                    for inner_key in outer_value[0].keys()
                }
        return new_output


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
    """upstream `muvo_2d/muvo/models/decoder.py:ConvDecoder2D` 패턴 + alpha 해상도.

    Input은 token-shape `(B, S, C, H, W)`. Linear+Unflatten 단계 없음.
    카메라: (10, 24) → 5× stride-2 → (320, 768).
    LiDAR : (2, 64)  → 4× stride-2 → (32, 1024).
    pre_transpose_conv은 단순 1×1 reduce.
    각 stage(trans_conv1/2/3) 후 head_4/2/1 분기, channel halving.
    """

    def __init__(self, in_channels, out_channels, base_hw, target_hw, sensor_type,
                 decoder_channels=256, n_basic_conv=None):
        super().__init__()
        self.sensor_type = sensor_type
        self.in_channels = in_channels
        self.target_hw = tuple(target_hw)
        self.base_hw = tuple(base_hw)

        # 총 ×2 doubling 횟수 = log2(target_hw / base_hw). target_hw가 base_hw × 2^N이어야.
        scale_h = target_hw[0] // base_hw[0]
        scale_w = target_hw[1] // base_hw[1]
        if scale_h != scale_w or scale_h <= 0 or (scale_h & (scale_h - 1)) != 0:
            raise ValueError(
                f"target_hw / base_hw must be the same power of 2; got "
                f"target={target_hw}, base={base_hw} → {scale_h}x vs {scale_w}x"
            )
        total_doublings = int(math.log2(scale_h))
        # 3 doublings은 trans_conv1/2/3가 담당. 나머지는 pre_transpose에서.
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

        # 3×3 conv: sample(in=C) → decoder_channels. upstream MUVO와 동일하게 sample만 디코드.
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

    def forward(self, sample_slice):
        """Input shape (B, S, C, H, W) → (B*S, C, H, W). upstream MUVO와 동일하게 sample만 디코드."""
        B, S, C, H, W = sample_slice.shape
        x = sample_slice.reshape(B * S, C, H, W)
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

        # RSSMTD constraint: embedding=hidden=state. 강제로 align.
        if embedding_n_channels != transformer_channels:
            raise ValueError(
                f"RSSMTD requires EMBEDDING_DIM == FUSION.TRANSFORMER_CHANNELS; "
                f"got {embedding_n_channels} != {transformer_channels}"
            )

        # Image encoder: ResNet18 out_indices=[2,3,4] (stride-32 종료, alpha _muvo와 동일)
        self.image_encoder = timm.create_model(
            "resnet18", pretrained=True, features_only=True, out_indices=[2, 3, 4]
        )
        self.image_fpn = FPNDecoder([128, 256, 512], out_channels=transformer_channels)

        # LiDAR encoder: out_indices upstream MUVO 매칭 (v3, default [1,2,3])
        lidar_out_indices = list(getattr(cfg.MODEL.RSSM_2D, "LIDAR_OUT_INDICES", (1, 2, 3)))
        self.lidar_encoder = timm.create_model(
            "resnet18", pretrained=True, features_only=True,
            out_indices=lidar_out_indices, in_chans=4,
        )
        # ResNet18 feature channels: layer1=64, layer2=128, layer3=256, layer4=512
        # out_indices에 대응되는 채널을 lookup.
        _resnet18_channels = {0: 64, 1: 64, 2: 128, 3: 256, 4: 512}
        lidar_feature_channels = [_resnet18_channels[i] for i in lidar_out_indices]
        self.lidar_fpn = FPNDecoder(lidar_feature_channels, out_channels=transformer_channels)

        self.transformer_encoder = (transformer_encoder if transformer_encoder is not None
                                    else SensorFusionTransformer(
                                        channels=transformer_channels,
                                        num_layers=getattr(cfg.MODEL.FUSION, "TRANSFORMER_LAYERS", 3),
                                        nhead=getattr(cfg.MODEL.FUSION, "TRANSFORMER_HEADS", 4),
                                        dropout=getattr(cfg.MODEL.FUSION, "TRANSFORMER_DROPOUT", 0.1),
                                    ))

        # Speed encoder (v2): modulation, not separate token
        speed_channels = getattr(cfg.MODEL, "SPEED_CHANNELS", 16)
        self.speed_normalisation = getattr(cfg.MODEL, "SPEED_NORMALISATION", 50.0)
        self.speed_encoder = SpeedEncoder(hidden_channels=speed_channels, out_channels=embedding_n_channels)

        # RSSMTD
        if not cfg.MODEL.TRANSITION.ENABLED:
            raise ValueError("cfg.MODEL.TRANSITION.ENABLED must be True.")

        rssm_cfg = cfg.MODEL.RSSM_2D
        self.image_token_hw = tuple(rssm_cfg.IMAGE_TOKEN_HW)
        self.lidar_token_hw = tuple(rssm_cfg.LIDAR_TOKEN_HW)

        self.rssm = RSSMTD(
            embedding_dim=embedding_n_channels,
            action_dim=cfg.MODEL.ACTION_DIM,
            hidden_state_dim=cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
            state_dim=cfg.MODEL.TRANSITION.STATE_DIM,
            image_token_hw=self.image_token_hw,
            lidar_token_hw=self.lidar_token_hw,
            num_decoder_layers=getattr(rssm_cfg, "TRANSFORMER_DECODER_LAYERS", 3),
            nhead=getattr(rssm_cfg, "TRANSFORMER_DECODER_HEADS", 4),
            use_dropout=cfg.MODEL.TRANSITION.USE_DROPOUT,
            dropout_probability=cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
        )

        # Token-shape decoders: input is (B, S, C, H_token, W_token) = sample slice only
        # (upstream MUVO와 동일하게 posterior sample만 디코드, hidden_state는 디코드 안 함).
        # in_channels = embedding_n_channels.
        # decoder_channels: upstream `latent_n_channels=512` 매칭 (v4 RGB plateau fix).
        # cfg.MODEL.RSSM_2D.DECODER_CHANNELS로 override 가능 (default 512).
        decoder_channels = getattr(rssm_cfg, "DECODER_CHANNELS", 2 * embedding_n_channels)
        self.image_decoder = TokenConvDecoder2D(
            in_channels=embedding_n_channels, out_channels=3,
            base_hw=self.image_token_hw, target_hw=tuple(cfg.DATA.RGB_RECON_SIZE),
            sensor_type='rgb', decoder_channels=decoder_channels,
        )
        self.lidar_decoder = TokenConvDecoder2D(
            in_channels=embedding_n_channels, out_channels=4,
            base_hw=self.lidar_token_hw, target_hw=tuple(cfg.DATA.LIDAR_RANGE_VIEW_SIZE),
            sensor_type='lidar_reconstruction', decoder_channels=decoder_channels,
        )

    # ─── slicing helpers ──────────────────────────────────────────────────────

    @property
    def n_image_tokens(self):
        return self.image_token_hw[0] * self.image_token_hw[1]

    @property
    def n_lidar_tokens(self):
        return self.lidar_token_hw[0] * self.lidar_token_hw[1]

    @property
    def n_total_tokens(self):
        return self.n_image_tokens + self.n_lidar_tokens

    def _slice_token_state(self, state):
        """state: (B, S, C, N_total) → (image, lidar) modality blocks.

        image: (B, S, C, H_img, W_img)
        lidar: (B, S, C, H_lid, W_lid)
        """
        B, S, C, N = state.shape
        N_img = self.n_image_tokens
        N_lid = self.n_lidar_tokens
        H_img, W_img = self.image_token_hw
        H_lid, W_lid = self.lidar_token_hw

        img = state[:, :, :, :N_img].reshape(B, S, C, H_img, W_img)
        lid = state[:, :, :, N_img:N_img + N_lid].reshape(B, S, C, H_lid, W_lid)
        return img, lid

    def _decode_state(self, sample):
        """sample shape (B, S, C, N_total). upstream MUVO와 동일하게 posterior/prior의
        sample만 디코드(hidden_state는 사용 안 함). Returns dict including
        rgb_{1,2,4}, lidar_reconstruction_{1,2,4}.
        """
        img_s, lid_s = self._slice_token_state(sample)

        out = {}
        out.update(self.image_decoder(img_s))
        out.update(self.lidar_decoder(lid_s))
        return out

    def decode_rgb(self, state):
        """Diffuvo_WM 경로(계획서 §3.4): (B,S,C,N_total) → **image 토큰만** 디코드해
        RGB dict `{rgb_1, rgb_2, rgb_4}` 반환. LiDAR 토큰/디코더는 건드리지 않는다.

        AE recon·diffusion 샘플 디코드 양쪽에서 사용. (LiDAR는 fusion 입력으로만 유지)
        """
        img_s, _ = self._slice_token_state(state)
        return self.image_decoder(img_s)

    # ─── encode / forward / imagine ───────────────────────────────────────────

    def encode_fuse_sequence(self, image, lidar, speed=None):
        """Returns (B, S, C, N_in) token-shape embedding.

        N_in = N_image + N_lidar (= 240 + 128 = 368 by default).
        speed가 있으면 SensorFusionTransformer 내부에서 modulation으로 모든 token에 broadcast add.
        """
        B, S, C_img, H_img, W_img = image.shape
        _, _, C_lidar, H_lidar, W_lidar = lidar.shape

        image_flat = image.reshape(B * S, C_img, H_img, W_img)
        lidar_flat = lidar.reshape(B * S, C_lidar, H_lidar, W_lidar)

        cam_feat = self.image_fpn(self.image_encoder(image_flat))
        lidar_feat = self.lidar_fpn(self.lidar_encoder(lidar_flat))

        # Speed embedding (B*S, C) for modulation
        speed_emb = None
        if speed is not None:
            speed_t = speed.float().to(image.device)
            if speed_t.ndim == 2:
                speed_t = speed_t.unsqueeze(-1)
            speed_emb = self.speed_encoder(
                speed_t.reshape(B * S, 1) / self.speed_normalisation
            )

        cam_out, lidar_out = self.transformer_encoder(cam_feat, lidar_feat, speed_embed=speed_emb)

        # Flatten spatial → token dim. cam_out (B*S, C, H_img, W_img) → (B*S, C, N_img).
        cam_tokens = cam_out.flatten(2)
        lid_tokens = lidar_out.flatten(2)
        tokens = torch.cat([cam_tokens, lid_tokens], dim=-1)  # (B*S, C, N_in)
        return tokens.reshape(B, S, self.transformer_channels, -1)

    def forward(self, batch, h_init=None, s_init=None,
                deployment=False, continuation=False, init_action=None):
        """RSSMTD는 h_init/s_init/continuation/init_action을 사용하지 않는다 (단순 zero-init).
        호환성을 위해 kwarg는 받지만 무시.
        """
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
        rssm_output = self.rssm(input_embedding=embedding_seq, action=action, use_sample=True, policy=None)

        output = {"prior": rssm_output["prior"], "posterior": rssm_output["posterior"]}
        st = rssm_output["posterior"]["sample"]
        output.update(self._decode_state(st))
        return output, rssm_output

    def imagine(self, state_imagine, future_horizon=None):
        """state_imagine: dict with
            hidden_state (B, C, N_total)
            sample       (B, C, N_total)
            action       (B, fh, action_dim)
        """
        h_t = state_imagine["hidden_state"]
        sample_t = state_imagine["sample"]
        device = h_t.device

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

        # Convert (B, C, N) → (N, B, C) for RSSMTD imagine_step.
        h_t = h_t.permute(2, 0, 1).contiguous()
        sample_t = sample_t.permute(2, 0, 1).contiguous()

        priors = {"hidden_state": [], "sample": [], "mu": [], "sigma": []}
        for t in range(future_horizon):
            prior_t = self.rssm.imagine_step(
                h_t=h_t, sample_t=sample_t,
                action_t=action[:, t], use_sample=True, policy=None,
            )
            h_t = prior_t["hidden_state"]
            sample_t = prior_t["sample"]
            for key in priors:
                priors[key].append(prior_t[key])

        # Each prior_t value shape (N, B, C). Stack along 
        # seq dim, then permute → (B, S, C, N).
        prior_seq = {
            k: torch.stack([t.permute(1, 2, 0) for t in v], dim=1)
            for k, v in priors.items()
        }
        output = {"prior": prior_seq}
        st = prior_seq["sample"]
        output.update(self._decode_state(st))
        return output, {"prior": prior_seq}
