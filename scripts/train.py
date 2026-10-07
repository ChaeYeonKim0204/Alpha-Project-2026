import pyarrow as pa
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import timm
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless server: no display required
import matplotlib.pyplot as plt
from PIL import Image
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Optional
import math
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("ALPHA26_ROOT", "/home/carol/chaeyeon-kim/alpha26")).expanduser()
DATA_ROOT = Path(os.environ.get("CARLA_ARROW_ROOT", str(PROJECT_ROOT.parent / "processed"))).expanduser()

#######################
# cfg
#######################
cfg = SimpleNamespace(
    RECEPTIVE_FIELD=4,
    FUTURE_HORIZON=4,

    MODEL=SimpleNamespace(
        ACTION_DIM=2,
        TRANSITION=SimpleNamespace(
            ENABLED=True,
            HIDDEN_STATE_DIM=128,
            STATE_DIM=64,
            ACTION_LATENT_DIM=32,
            USE_DROPOUT=False,
            DROPOUT_PROBABILITY=0.0,
        ),
    ),

    LOSSES=SimpleNamespace(
        WEIGHT_PROBABILISTIC=1.0,
        WEIGHT_LIDAR_RE=1.0,
    ),

    OPTIMIZER=SimpleNamespace(
        LR=1e-4,
        WEIGHT_DECAY=0.01,
        ACCUMULATE_GRAD_BATCHES=1,
    ),

    SCHEDULER=SimpleNamespace(
        NAME='none',          # 'OneCycleLR' or 'none'
        PCT_START=0.3,
    ),

    STEPS=12000,  # 20 epochs ~= 600 steps/epoch * 20
    PRETRAINED=SimpleNamespace(
        PATH=None,
    ),
)

#######################
# point cloud to range view
#######################
to_tensor = transforms.ToTensor()

def point_cloud_to_range_view(points, H=64, W=512):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    yaw   = np.arctan2(y, x)
    pitch = np.arcsin(z / (r + 1e-8))

    yaw_img   = (yaw   / np.pi + 1) / 2
    pitch_img = (pitch / (np.pi/2) + 1) / 2
    col = np.clip((yaw_img   * (W - 1)).astype(int), 0, W - 1)
    row = np.clip(((1 - pitch_img) * (H - 1)).astype(int), 0, H - 1)

    # 거리 내림차순 정렬 → 가까운 점이 마지막에 덮어써서 closest-wins
    order = np.argsort(r)[::-1]
    rv = np.zeros((4, H, W), dtype=np.float32)
    rv[0, row[order], col[order]] = x[order]
    rv[1, row[order], col[order]] = y[order]
    rv[2, row[order], col[order]] = z[order]
    rv[3, row[order], col[order]] = r[order]
    return rv


class MUVODataset(Dataset):
    def __init__(self, arrow_file, seq_len=4):
        self.seq_len = seq_len
        self.arrow_file = arrow_file

        self._mmap = pa.memory_map(arrow_file, "r")
        reader = pa.ipc.open_stream(self._mmap)
        self.table = reader.read_all()
        self.n = self.table.num_rows

        # non-overlapping 윈도우: step=seq_len으로 TBPTT 연속성 보장
        self.index = []
        run_ids = self.table.column("run_id").to_pylist()
        for i in range(0, self.n - seq_len + 1, seq_len):
            if all(run_ids[i + j] == run_ids[i] for j in range(seq_len)):
                self.index.append(i)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i = self.index[idx]
        images, lidars, actions, speeds = [], [], [], []

        for j in range(self.seq_len):
            k = i + j
            row = self.table.slice(k, 1)
            img_bytes = row.column("image_front")[0].as_py()["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(to_tensor(img))

            lidar_np = np.array(row.column("lidar")[0].as_py())
            lidars.append(torch.tensor(point_cloud_to_range_view(lidar_np)))

            actions.append([row.column("throttle")[0].as_py(), row.column("steer")[0].as_py()])
            speeds.append(row.column("speed_kmh")[0].as_py())

        return {
            "image":  torch.stack(images),          # (S, 3, 600, 800)
            "lidar":  torch.stack(lidars),           # (S, 4, 64, 512)
            "action": torch.tensor(actions),         # (S, 2)
            "speed":  torch.tensor(speeds),          # (S,)
            "run_id": self.table.column("run_id")[i].as_py(),  # str — TBPTT run 경계 감지용
        }

#######################
# Image Encoder
#######################
class FPNDecoder(nn.Module):
    def __init__(self, feature_info, out_channels=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_info[2], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.skip_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_info[i], out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            for i in [1, 0]
        ])
        self.out_channels = out_channels

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        x = self.conv1(xs[2])

        for i, conv in enumerate(self.skip_convs):
            size = xs[1 - i].shape[-2:]
            x = conv(xs[1 - i]) + F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x

#######################
# Transformer Fusion
#######################
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        B, C, h, w = x.shape

        not_mask = torch.ones((B, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, C, h, w)
        return pos


class SensorFusionTransformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.position_encode = PositionEmbeddingSine(
            num_pos_feats=channels // 2,
            normalize=True
        )

        # sensor type embedding: 0=camera, 1=lidar
        self.type_embedding = nn.Parameter(torch.zeros(2, channels))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def feature_to_tokens(self, x):
        return x.flatten(2).permute(2, 0, 1)

    def add_pos_and_type(self, x, sensor_type_idx):
        tokens = self.feature_to_tokens(x)          # (H*W, B, C)
        pos = self.position_encode(x)               # (B, C, H, W)
        pos_tokens = self.feature_to_tokens(pos)    # (H*W, B, C)
        NHW, B, C = tokens.shape
        type_embed = self.type_embedding[sensor_type_idx].view(1, 1, C).expand(NHW, B, C)
        tokens = tokens + pos_tokens + type_embed
        return tokens

    def forward(self, cam_feat, lidar_feat):
        B, C, Hc, Wc = cam_feat.shape
        _, _, Hl, Wl = lidar_feat.shape
        L_cam = Hc * Wc

        cam_tokens   = self.add_pos_and_type(cam_feat,   sensor_type_idx=0)
        lidar_tokens = self.add_pos_and_type(lidar_feat, sensor_type_idx=1)

        fused_tokens = torch.cat([cam_tokens, lidar_tokens], dim=0)
        out = self.transformer_encoder(fused_tokens)

        cam_out   = out[:L_cam].permute(1, 2, 0).reshape(B, C, Hc, Wc)
        lidar_out = out[L_cam:].permute(1, 2, 0).reshape(B, C, Hl, Wl)
        return cam_out, lidar_out

#######################
# RSSM
#######################
class RepresentationModel(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.min_std = 0.1

        self.module = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(True),
            nn.Linear(in_channels, 2 * self.latent_dim),
        )

    def forward(self, x):
        def sigmoid2(tensor: torch.Tensor, min_value: float) -> torch.Tensor:
            return 2 * torch.sigmoid(tensor / 2) + min_value

        mu_log_sigma = self.module(x)
        mu, log_sigma = torch.split(mu_log_sigma, self.latent_dim, dim=-1)
        sigma = sigmoid2(log_sigma, self.min_std)
        return mu, sigma


class RSSM(nn.Module):
    def __init__(self, embedding_dim, action_dim, hidden_state_dim, state_dim,
                 action_latent_dim, receptive_field, use_dropout=False, dropout_probability=0.0):
        super().__init__()
        self.embedding_dim     = embedding_dim
        self.state_dim         = state_dim
        self.action_dim        = action_dim
        self.hidden_state_dim  = hidden_state_dim
        self.action_latent_dim = action_latent_dim
        self.receptive_field   = receptive_field
        self.use_dropout       = use_dropout
        self.dropout_probability = dropout_probability
        self.active_inference  = False

        self.pre_gru_net = nn.Sequential(
            nn.Linear(state_dim + action_latent_dim, hidden_state_dim),
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
                use_sample=True, policy=None, continuation=False):
        output = {"prior": [], "posterior": []}
        batch_size, sequence_length, _ = input_embedding.shape

        h_t = h_init if h_init is not None else input_embedding.new_zeros((batch_size, self.hidden_state_dim))
        sample_t = s_init if s_init is not None else input_embedding.new_zeros((batch_size, self.state_dim))

        for t in range(sequence_length):
            if continuation:
                action_t = action[:, t]
            else:
                action_t = torch.zeros_like(action[:, 0]) if t == 0 else action[:, t - 1]

            output_t = self.observe_step(h_t, sample_t, action_t, input_embedding[:, t],
                                         use_sample=use_sample, policy=policy)
            use_prior = (self.training and self.use_dropout
                         and torch.rand(1).item() < self.dropout_probability and t > 0)
            sample_t = output_t["prior"]["sample"] if use_prior else output_t["posterior"]["sample"]
            h_t = output_t["prior"]["hidden_state"]
            for key, value in output_t.items():
                output[key].append(value)

        return self.stack_list_of_dict_tensor(output, dim=1)

    def observe_step(self, h_t, sample_t, action_t, embedding_t, use_sample=True, policy=None):
        imagine_output = self.imagine_step(h_t, sample_t, action_t, use_sample, policy=policy)
        latent_action_t = self.posterior_action_module(action_t)
        posterior_mu_t, posterior_sigma_t = self.posterior(
            torch.cat([imagine_output["hidden_state"], embedding_t, latent_action_t], dim=-1)
        )
        sample_t = self.sample_from_distribution(posterior_mu_t, posterior_sigma_t, use_sample)
        return {
            "prior": imagine_output,
            "posterior": {
                "hidden_state": imagine_output["hidden_state"],
                "sample": sample_t,
                "mu": posterior_mu_t,
                "sigma": posterior_sigma_t,
            },
        }

    def imagine_step(self, h_t, sample_t, action_t, use_sample=True, policy=None):
        if self.active_inference:
            action_t = policy(torch.cat([h_t, sample_t], dim=-1))

        latent_action_t = self.prior_action_module(action_t)

        gru_input = torch.cat([sample_t, latent_action_t], dim=-1)
        input_t = self.pre_gru_net(gru_input)
        h_t = self.recurrent_model(input_t, h_t)

        prior_mu_t, prior_sigma_t = self.prior(torch.cat([h_t, latent_action_t], dim=-1))
        sample_t = self.sample_from_distribution(prior_mu_t, prior_sigma_t, use_sample)
        return {"hidden_state": h_t, "sample": sample_t, "mu": prior_mu_t, "sigma": prior_sigma_t}

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

#######################
# Decoder
#######################
class SensorHead(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor, sensor_type):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.sensor_type = sensor_type
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, B, S):
        out = self.head(x)
        _, C, H, W = out.shape
        return {
            f'{self.sensor_type}_{self.downsample_factor}': out.view(B, S, C, H, W)
        }


class SensorDecoder(nn.Module):
    """
    RGB   목표: (3, 600, 800)  constant_size=(19, 25)
    LiDAR 목표: (4, 64,  512)  constant_size=(2,  16)
    """
    def __init__(self, latent_n_channels, out_channels, constant_size, sensor_type):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_n_channels, 256),
            nn.Unflatten(-1, (256, 1, 1))
        )

        self.base_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=constant_size),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )
        self.head_4 = SensorHead(128, out_channels, downsample_factor=4, sensor_type=sensor_type)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )
        self.head_2 = SensorHead(64, out_channels, downsample_factor=2, sensor_type=sensor_type)

        self.head_1 = SensorHead(64, out_channels, downsample_factor=1, sensor_type=sensor_type)

    def forward(self, hidden_state, sample):
        B, S, _ = hidden_state.shape
        x = torch.cat([hidden_state, sample], dim=-1).view(B * S, -1)

        x = self.linear(x)
        x = self.base_conv(x)

        x = self.up1(x)
        out_4 = self.head_4(x, B, S)

        x = self.up2(x)
        out_2 = self.head_2(x, B, S)
        out_1 = self.head_1(x, B, S)

        outputs = {**out_4, **out_2}

        return outputs

#######################
# Model
#######################
class Model(nn.Module):
    def __init__(self, cfg, embedding_n_channels=128, transformer_encoder=None):
        super().__init__()
        self.cfg = cfg
        self.embedding_n_channels = embedding_n_channels
        self.receptive_field = self.cfg.RECEPTIVE_FIELD

        self.image_encoder = timm.create_model("resnet18", pretrained=True,
                                               features_only=True, out_indices=[2, 3, 4])
        self.image_fpn = FPNDecoder([128, 256, 512], out_channels=embedding_n_channels)

        self.lidar_encoder = timm.create_model("resnet18", pretrained=True,
                                               features_only=True, out_indices=[2, 3, 4], in_chans=4)
        self.lidar_fpn = FPNDecoder([128, 256, 512], out_channels=embedding_n_channels)

        self.transformer_encoder = (transformer_encoder if transformer_encoder is not None
                                    else SensorFusionTransformer(channels=embedding_n_channels))

        self.image_feature_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.lidar_feature_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
        self.features_combine = nn.Linear(2 * embedding_n_channels, embedding_n_channels)

        latent_dim = (self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM +
                      self.cfg.MODEL.TRANSITION.STATE_DIM)  # 128+64=192

        self.image_decoder = SensorDecoder(
            latent_n_channels=latent_dim,
            out_channels=3,
            constant_size=(19, 25),
            sensor_type='rgb',
        )
        self.lidar_decoder = SensorDecoder(
            latent_n_channels=latent_dim,
            out_channels=4,
            constant_size=(2, 16),
            sensor_type='lidar_reconstruction',
        )

        if not self.cfg.MODEL.TRANSITION.ENABLED:
            raise ValueError("cfg.MODEL.TRANSITION.ENABLED must be True.")

        self.rssm = RSSM(
            embedding_dim=embedding_n_channels,
            action_dim=self.cfg.MODEL.ACTION_DIM,
            hidden_state_dim=self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
            state_dim=self.cfg.MODEL.TRANSITION.STATE_DIM,
            action_latent_dim=self.cfg.MODEL.TRANSITION.ACTION_LATENT_DIM,
            receptive_field=self.receptive_field,
            use_dropout=self.cfg.MODEL.TRANSITION.USE_DROPOUT,
            dropout_probability=self.cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
        )

    def encode_fuse_sequence(self, image, lidar):
        B, S, C_img, H_img, W_img = image.shape
        _, _, C_lidar, H_lidar, W_lidar = lidar.shape

        image_flat = image.reshape(B * S, C_img, H_img, W_img)
        lidar_flat = lidar.reshape(B * S, C_lidar, H_lidar, W_lidar)

        cam_feat   = self.image_fpn(self.image_encoder(image_flat))
        lidar_feat = self.lidar_fpn(self.lidar_encoder(lidar_flat))

        _, C, _, _ = cam_feat.shape
        cam_feat   = F.adaptive_avg_pool2d(cam_feat,   (15, 20)).reshape(B, S, C, 15, 20)
        lidar_feat = F.adaptive_avg_pool2d(lidar_feat, (8,  16)).reshape(B, S, C, 8,  16)

        embeddings = []
        for t in range(S):
            cam_out, lidar_out = self.transformer_encoder(cam_feat[:, t], lidar_feat[:, t])
            cam_emb   = self.image_feature_compress(cam_out)    # (B, C)
            lidar_emb = self.lidar_feature_compress(lidar_out)  # (B, C)
            embedding = self.features_combine(torch.cat([cam_emb, lidar_emb], dim=-1))
            embeddings.append(embedding)

        return torch.stack(embeddings, dim=1)  # (B, S, C)

    def forward(self, batch, h_init=None, s_init=None, deployment=False, continuation=False):
        image  = batch["image"].float()
        lidar  = batch["lidar"].float()

        if "action" in batch:
            action = batch["action"].float()
        elif "throttle_brake" in batch and "steering" in batch:
            action = torch.cat([batch["throttle_brake"], batch["steering"]], dim=-1).float()
        else:
            raise KeyError("batch must contain 'action' or both 'throttle_brake' and 'steering'.")

        embedding_seq = self.encode_fuse_sequence(image, lidar)

        rssm_output = self.rssm(
            input_embedding=embedding_seq,
            action=action,
            h_init=h_init,
            s_init=s_init,
            use_sample=True,
            policy=None,
            continuation=continuation,
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

        priors = {"hidden_state": [], "sample": [], "mu": [], "sigma": []}

        for t in range(future_horizon):
            prior_t = self.rssm.imagine_step(
                h_t=h_t, sample_t=sample_t, action_t=action[:, t],
                use_sample=True, policy=None,
            )
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

#######################
# Training
#######################
class WorldModelTrainer(pl.LightningModule):

    def __init__(self, cfg=None, hparams=None, lr=None, embedding_n_channels=128):
        super().__init__()
        self.save_hyperparameters(ignore=["hparams"])

        self.cfg = cfg or hparams
        self.rf = self.cfg.RECEPTIVE_FIELD
        self.fh = self.cfg.FUTURE_HORIZON

        self.model = Model(self.cfg, embedding_n_channels=embedding_n_channels)
        self.load_pretrained_weights()

        self.lr = lr if lr is not None else getattr(self.cfg.OPTIMIZER, "LR", 1e-4)
        self.weight_probabilistic = getattr(self.cfg.LOSSES, "WEIGHT_PROBABILISTIC", 1.0)
        self.weight_lidar_re      = getattr(self.cfg.LOSSES, "WEIGHT_LIDAR_RE", 1.0)
        self.weight_rgb           = 0.1

        self._tbptt_h: Optional[torch.Tensor] = None
        self._tbptt_s: Optional[torch.Tensor] = None
        self._prev_run_id = None

    def load_pretrained_weights(self):
        if not hasattr(self.cfg, "PRETRAINED"):
            return
        pretrained_path = getattr(self.cfg.PRETRAINED, "PATH", None)
        if not pretrained_path or not os.path.isfile(pretrained_path):
            return
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        cleaned = {(k[len("model."):] if k.startswith("model.") else k): v
                   for k, v in checkpoint.items()}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print(f"Loaded: {pretrained_path}  missing={len(missing)}  unexpected={len(unexpected)}")

    def prepare_custom_batch(self, batch):
        if "action" in batch:
            if "throttle_brake" not in batch:
                batch["throttle_brake"] = batch["action"][..., 0:1]
            if "steering" not in batch:
                batch["steering"] = batch["action"][..., 1:2]
        if "range_view_pcd_xyzd" in batch and "lidar" not in batch:
            batch["lidar"] = batch["range_view_pcd_xyzd"].float()
        if "lidar" in batch and "range_view_label_1" not in batch:
            batch["range_view_label_1"] = batch["lidar"].float()
        if "image" in batch and "rgb_label_1" not in batch:
            batch["rgb_label_1"] = batch["image"].float()
        if "image" in batch:
            batch["image"] = batch["image"].float()
        if "lidar" in batch:
            batch["lidar"] = batch["lidar"].float()
        if "action" in batch:
            batch["action"] = batch["action"].float()
        return batch

    def forward(self, batch):
        batch = self.prepare_custom_batch(batch)
        output, state_dict = self.model.forward(batch)
        return output, state_dict

    @staticmethod
    def gaussian_kl_loss(prior, posterior, eps=1e-6):
        mu_p  = prior["mu"]
        sig_p = prior["sigma"].clamp_min(eps)
        mu_q  = posterior["mu"]
        sig_q = posterior["sigma"].clamp_min(eps)
        kl = (torch.log(sig_p / sig_q)
              + (sig_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sig_p.pow(2))
              - 0.5)
        return kl.mean()

    @staticmethod
    def resize_sequence_tensor(x, target_hw, mode="bilinear"):
        b, s, c, h, w = x.shape
        x = x.reshape(b * s, c, h, w)
        x = F.interpolate(x, size=target_hw, mode=mode,
                          **({} if mode == "nearest" else {"align_corners": False}))
        return x.reshape(b, s, c, *target_hw)

    def compute_loss(self, batch, output):
        losses = {}

        if "prior" in output and "posterior" in output:
            losses["probabilistic"] = (self.weight_probabilistic
                                       * self.gaussian_kl_loss(output["prior"], output["posterior"]))

        if "range_view_label_1" in batch:
            for factor in [2, 4]:
                key = f'lidar_reconstruction_{factor}'
                if key in output:
                    pred   = output[key]
                    target = batch["range_view_label_1"]
                    discount = 1 / factor
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="nearest")
                    if pred.shape[2] >= 3:
                        losses[f'lidar_xyz_{factor}'] = self.weight_lidar_re * discount * F.mse_loss(
                            pred[:, :, :3], target[:, :, :3])
                    losses[f'lidar_depth_{factor}'] = self.weight_lidar_re * discount * F.l1_loss(
                        pred[:, :, -1:], target[:, :, -1:])

        if "rgb_label_1" in batch:
            for factor in [2, 4]:
                key = f'rgb_{factor}'
                if key in output:
                    pred   = output[key]
                    target = batch["rgb_label_1"]
                    discount = 1 / factor
                    if pred.shape[-2:] != target.shape[-2:]:
                        target = self.resize_sequence_tensor(target, pred.shape[-2:], mode="bilinear")
                    losses[key] = self.weight_rgb * discount * F.l1_loss(pred, target)

        if not losses:
            raise RuntimeError("No valid loss computed.")
        return losses

    def loss_reducing(self, losses):
        return sum(losses.values())

    def shared_step(self, batch, mode="val"):
        batch = self.prepare_custom_batch(batch)
        output, state_dict = self.model.forward(batch)
        losses = self.compute_loss(batch, output)
        return losses, output, state_dict

    def training_step(self, batch, batch_idx):
        batch = self.prepare_custom_batch(batch)
        B      = batch["image"].shape[0]
        device = batch["image"].device
        hidden_dim = self.model.rssm.hidden_state_dim
        state_dim  = self.model.rssm.state_dim

        run_id = batch.get("run_id", None)
        current_run_id = run_id[0] if isinstance(run_id, (list, tuple)) else run_id
        is_new_run = (self._prev_run_id is None) or (current_run_id != self._prev_run_id)

        if self._tbptt_h is None or self._tbptt_s is None or is_new_run:
            h_prev = torch.zeros(B, hidden_dim, device=device)
            s_prev = torch.zeros(B, state_dim,  device=device)
        else:
            h_prev = self._tbptt_h[:B].to(device).detach()
            s_prev = self._tbptt_s[:B].to(device).detach()

        self._prev_run_id = current_run_id

        output, state_dict = self.model.forward(batch, h_init=h_prev, s_init=s_prev)

        self._tbptt_h = state_dict["posterior"]["hidden_state"][:, -1, :].detach()
        self._tbptt_s = state_dict["posterior"]["sample"][:, -1, :].detach()

        losses = self.compute_loss(batch, output)
        total_loss = self.loss_reducing(losses)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, value in losses.items():
            self.log(f"train_{name}", value, prog_bar=False, on_step=True, on_epoch=True)
        return total_loss

    def on_train_epoch_end(self):
        self._tbptt_h = None
        self._tbptt_s = None
        self._prev_run_id = None

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        losses, output, state_dict = self.shared_step(batch, mode="val")
        total_loss = self.loss_reducing(losses)
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=False, on_step=False, on_epoch=True)
        return {"val_loss": total_loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        losses, output, state_dict = self.shared_step(batch, mode="test")
        total_loss = self.loss_reducing(losses)
        self.log("test_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, value in losses.items():
            self.log(f"test_{name}", value, prog_bar=False, on_step=False, on_epoch=True)
        return {"test_loss": total_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr,
            weight_decay=getattr(self.cfg.OPTIMIZER, "WEIGHT_DECAY", 0.0),
        )
        scheduler_name = getattr(self.cfg.SCHEDULER, "NAME", "none")
        if scheduler_name == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr, total_steps=self.cfg.STEPS,
                pct_start=getattr(self.cfg.SCHEDULER, "PCT_START", 0.3),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer

#######################
# Visualization
#######################
def _latest_checkpoint(save_dir="./logs"):
    ckpts = glob.glob(os.path.join(save_dir, "**", "*.ckpt"), recursive=True)
    return max(ckpts, key=os.path.getmtime) if ckpts else None


@torch.no_grad()
def visualize_reconstruction(model, ds, sample_idx=0, time_idx=-1, out_path="reconstruction.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    sample = ds[sample_idx]
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }

    use_cuda_amp = device == "cuda"
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_cuda_amp):
        output, _ = model(batch)

    input_rgb  = batch["image"][0, time_idx].detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    pred_rgb   = output["rgb_2"][0, time_idx].detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    input_depth = batch["lidar"][0, time_idx, 3].detach().float().cpu()
    pred_depth  = output["lidar_reconstruction_2"][0, time_idx, 3].detach().float().cpu()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes[0, 0].imshow(input_rgb);        axes[0, 0].set_title("Input RGB")
    axes[0, 1].imshow(pred_rgb);         axes[0, 1].set_title("Reconstructed RGB")
    axes[1, 0].imshow(input_depth, cmap="magma"); axes[1, 0].set_title("Input LiDAR Depth")
    axes[1, 1].imshow(pred_depth,  cmap="magma"); axes[1, 1].set_title("Reconstructed LiDAR Depth")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

#######################
# main
#######################
def main():
    arrow_file = str(DATA_ROOT / "train_run_002.arrow")
    ds = MUVODataset(arrow_file, seq_len=cfg.RECEPTIVE_FIELD)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = WorldModelTrainer(cfg=cfg, lr=1e-4, embedding_n_channels=128)

    save_dir = "./logs"

    callbacks = [
        pl.callbacks.ModelSummary(),
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            every_n_train_steps=1000,
            save_top_k=-1,
        ),
    ]

    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator='auto',
        precision='16-mixed',
        max_steps=cfg.STEPS,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir=save_dir),
        log_every_n_steps=10,
        val_check_interval=600,
        check_val_every_n_epoch=None,
        limit_val_batches=0,
        accumulate_grad_batches=cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    return model, ds


if __name__ == '__main__':
    trained_model, train_ds = main()
    visualize_reconstruction(trained_model, train_ds, sample_idx=0, time_idx=-1)
