from typing import Any, Dict, List, TypeVar, Union

import torch
import torch.hub
from torch import nn
from torchvision import models

from ascent.datasets.tracking_dataset import OEDItem
from ascent.models.util import modify_channelvit_input_channels

# Define a recursive type alias for the structure
DeviceData = Union[torch.Tensor, Dict[str, Any], List[Any], Any]

# T captures the specific type of the input (e.g., a specific Dict subclass)
T = TypeVar("T", bound=DeviceData)


def to_device(data: T, device: torch.device) -> T:
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}  # type: ignore
    elif isinstance(data, torch.Tensor):
        return data.to(device)  # type: ignore
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]  # type: ignore
    return data


class LocalFeatureNet(nn.Module):
    def __init__(self, patch_size_xy: int, patch_size_z: int):
        super(LocalFeatureNet, self).__init__()
        self.patch_size_xy = patch_size_xy
        self.patch_size_z = patch_size_z
        self.z_offsets = [0]
        for i in range(1, patch_size_z // 2 + 1):
            if len(self.z_offsets) < patch_size_z:
                self.z_offsets.append(i)
            if len(self.z_offsets) < patch_size_z:
                self.z_offsets.append(-i)

        self.backbone = None
        self.backbone_kwargs = None
        self.out_features = None

    def get_patches_vectorized(self, x: OEDItem):
        patch_size = self.patch_size_xy

        image = x["image"][:, 0:1]  # only use the first channel
        D, H, W = image.shape[-3], image.shape[-2], image.shape[-1]
        object_positions = torch.round(x["coords"])
        N, S = object_positions.shape[:2]
        mask = x["object_ids"] == -1

        # Make base grid which will be used for all batches
        grid_l = -(patch_size // 2) * 2 / (W - 1)
        grid_r = (patch_size // 2 - 1) * 2 / (W - 1)
        grid_t = -(patch_size // 2) * 2 / (H - 1)
        grid_b = (patch_size // 2 - 1) * 2 / (H - 1)
        grid_range_x = torch.linspace(grid_l, grid_r, patch_size, device=image.device)
        grid_range_y = torch.linspace(grid_t, grid_b, patch_size, device=image.device)
        grid_y, grid_x = torch.meshgrid(grid_range_y, grid_range_x, indexing="ij")
        base_grid_xy = torch.stack(
            [grid_x, grid_y], dim=-1
        )  # (patch_size, patch_size, 2)
        base_grid_xy = base_grid_xy.unsqueeze(0).repeat(
            len(self.z_offsets), 1, 1, 1
        )  # (len(z_offsets), patch_size, patch_size, 2)
        grid_z = []
        for z_offset in self.z_offsets:
            grid_z.append(
                torch.full(
                    (patch_size, patch_size),
                    z_offset * 2 / (D - 1),
                    dtype=torch.float,
                    device=image.device,
                )
            )
        grid_z = torch.stack(grid_z, dim=0).unsqueeze(
            -1
        )  # (len(z_offsets), patch_size, patch_size, 1)
        base_grid_xyz = torch.cat(
            [base_grid_xy, grid_z], dim=-1
        )  # (len(z_offsets), patch_size, patch_size, 3)
        base_grid_xyz = base_grid_xyz.unsqueeze(0).repeat(
            S, 1, 1, 1, 1
        )  # (S, len(z_offsets), patch_size, patch_size, 3)

        patches = torch.empty(
            (N, S, len(self.z_offsets), patch_size, patch_size),
            dtype=image.dtype,
            device=image.device,
        )
        # Get center positions
        center_positions = torch.empty(
            (S, 1, 1, 1, 3), dtype=torch.float, device=image.device
        )
        for i in range(N):
            # zyx -> xyz
            center_positions[:, 0, 0, 0, 0] = (
                object_positions[i, :, 2] / (W - 1) * 2 - 1
            )
            center_positions[:, 0, 0, 0, 1] = (
                object_positions[i, :, 1] / (H - 1) * 2 - 1
            )
            center_positions[:, 0, 0, 0, 2] = (
                object_positions[i, :, 0] / (D - 1) * 2 - 1
            )

            # Get final sampling grid
            sampling_grid = (
                base_grid_xyz + center_positions
            )  # (S, len(z_offsets), patch_size, patch_size, 3)

            # Get input (image=(N, C, D, H, W))
            input = image[i : i + 1].expand(
                S, -1, -1, -1, -1
            )  # (1, C, D, H, W) -> (S, C, D, H, W)

            # Get patches
            patch = nn.functional.grid_sample(
                input,
                sampling_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,  # (S, C, len(z_offsets), patch_size, patch_size)
            )
            patches[i] = patch[:, 0]  # only use the first channel

        patches[mask] = 0
        return patches

    def get_patches(self, x: OEDItem):
        device = next(self.parameters()).device

        res = []
        mask = x["object_ids"] == -1
        for i in range(x["image"].shape[0]):
            image_zyx = x["image"][i, 0, ...]
            object_positions = x["coords"][i]
            min_x = int(torch.round(object_positions[~mask[i], 2].min()).int().item())
            max_x = int(torch.round(object_positions[~mask[i], 2].max()).int().item())
            min_y = int(torch.round(object_positions[~mask[i], 1].min()).int().item())
            max_y = int(torch.round(object_positions[~mask[i], 1].max()).int().item())
            pad_x_left = max(self.patch_size_xy // 2 - min_x, 0)
            pad_x_right = max(
                self.patch_size_xy // 2 - (image_zyx.shape[-1] - max_x), 0
            )
            pad_y_top = max(self.patch_size_xy // 2 - min_y, 0)
            pad_y_bottom = max(
                self.patch_size_xy // 2 - (image_zyx.shape[-2] - max_y), 0
            )
            # check if xy padding is needed
            if pad_x_left > 0 or pad_x_right > 0 or pad_y_top > 0 or pad_y_bottom > 0:
                image_zyx = nn.functional.pad(
                    image_zyx, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom)
                )

            # get patches for this frame
            list_patches = []
            # iterate over each position
            for j, pos in enumerate(object_positions):
                if mask[i, j]:
                    pseudo_patch = torch.zeros(
                        [self.patch_size_z, self.patch_size_xy, self.patch_size_xy],
                        dtype=image_zyx.dtype,
                        device=image_zyx.device,
                    )
                    list_patches.append(pseudo_patch)
                else:
                    pos_z, pos_y, pos_x = torch.round(pos).long()
                    cur_y = pad_y_top + pos_y
                    cur_x = pad_x_left + pos_x
                    # 2D local patch of a z-slice is a channel of multi-channel input for ChannelViT
                    # Channel order is z, z+1, z-1, z+2, z-2, ...
                    cur_patches = torch.zeros(
                        (len(self.z_offsets), self.patch_size_xy, self.patch_size_xy),
                        dtype=image_zyx.dtype,
                        device=image_zyx.device,
                    )
                    xy_offset = self.patch_size_xy // 2
                    for i, z_offset in enumerate(self.z_offsets):
                        cur_z = pos_z + z_offset
                        if cur_z < 0 or cur_z >= image_zyx.shape[-3]:
                            continue
                        cur_patches[i] = image_zyx[
                            cur_z,
                            cur_y - xy_offset : cur_y + xy_offset,
                            cur_x - xy_offset : cur_x + xy_offset,
                        ]
                    list_patches.append(cur_patches)

            list_patches = torch.stack(list_patches, dim=0).to(device)
            res.append(list_patches)

        res = torch.stack(res, dim=0)
        return res

    def forward(self, x: dict):
        assert len(x["image"].shape) == 5, "x should be batched OEDItem"
        device = next(self.parameters()).device
        if device != x["image"].device:
            x = to_device(x, device)

        # get local patches
        with torch.no_grad():
            y = self.get_patches_vectorized(x)  # (N, S, C, H, W)

        # get local features
        y = y.view(-1, y.shape[-3], y.shape[-2], y.shape[-1])  # (NS, C, H, W)
        y = self.backbone(y, **self.backbone_kwargs)  # (NS, D)
        y = y.view(x["object_ids"].shape[0], x["object_ids"].shape[1], -1)  # (N, S, D)

        return y


class LocalFeatureViT(LocalFeatureNet):
    def __init__(self, patch_size_xy: int, patch_size_z: int, pretrained: str):
        super(LocalFeatureViT, self).__init__(patch_size_xy, patch_size_z)

        self.extra_tokens = {
            "channels": torch.tensor([range(patch_size_z)], dtype=torch.long)
        }
        self.backbone = torch.hub.load(
            "insitro/ChannelViT", pretrained, pretrained=True
        )
        if patch_size_z != self.backbone.in_chans:
            self.backbone = modify_channelvit_input_channels(
                self.backbone, patch_size_z
            )
        self.backbone_kwargs = {"extra_tokens": self.extra_tokens}
        self.out_features = self.backbone.embed_dim


class LocalFeatureCNN(LocalFeatureNet):
    def __init__(
        self,
        patch_size_xy: int,
        patch_size_z: int,
        out_features: int,
        backbone: str,
        weights: str,
    ):
        super(LocalFeatureCNN, self).__init__(patch_size_xy, patch_size_z)

        if "resnet" in backbone:
            if backbone == "resnet18":
                self.backbone = models.resnet18(weights=weights)
            elif backbone == "resnet34":
                self.backbone = models.resnet34(weights=weights)
            elif backbone == "resnet50":
                self.backbone = models.resnet50(weights=weights)
            else:
                raise Exception("Invalid model name")
            self.backbone.conv1 = nn.Conv2d(
                patch_size_z,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            self.backbone.fc = nn.Linear(
                in_features=self.backbone.fc.in_features,
                out_features=out_features,
                bias=True,
            )
        elif "mobilenet" in backbone:
            if backbone == "mobilenet_v2":
                self.backbone = models.mobilenet_v2(weights=weights)
                self.backbone.features[0][0] = nn.Conv2d(
                    patch_size_z,
                    32,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
                self.backbone.classifier = nn.Linear(
                    in_features=self.backbone.classifier[1].in_features,
                    out_features=out_features,
                    bias=True,
                )
            elif "v3" in backbone:
                if backbone == "mobilenet_v3_small":
                    self.backbone = models.mobilenet_v3_small(weights=weights)
                elif backbone == "mobilenet_v3_large":
                    self.backbone = models.mobilenet_v3_large(weights=weights)
                else:
                    raise Exception("Invalid model name")
                self.backbone.features[0][0] = nn.Conv2d(
                    patch_size_z,
                    16,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                )
                self.backbone.classifier = nn.Linear(
                    in_features=self.backbone.classifier[0].in_features,
                    out_features=out_features,
                    bias=True,
                )
        else:
            raise Exception("Invalid model name")

        self.backbone_kwargs = {}
        self.out_features = out_features


def build_pe_mlp(num_layers, input_dim, hidden_dim, pe_norm):
    layers = []
    # First layer: from input_dim to hidden_dim
    layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=True))
    if pe_norm is not None and pe_norm != "None":
        if pe_norm == "batch":
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif pe_norm == "layer":
            layers.append(nn.GroupNorm(1, hidden_dim))
        else:
            raise ValueError(f"Invalid pe_norm: {pe_norm}")
    layers.append(nn.ReLU())

    # Intermediate layers: hidden_dim to hidden_dim
    for _ in range(num_layers - 2):
        layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True))
        if pe_norm is not None and pe_norm != "None":
            if pe_norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif pe_norm == "layer":
                layers.append(nn.GroupNorm(1, hidden_dim))
            else:
                raise ValueError(f"Invalid pe_norm: {pe_norm}")
        layers.append(nn.ReLU())

    # Final layer: no activation or BN after
    if num_layers > 1:
        layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True))
    return nn.Sequential(*layers)


class NETr(nn.Module):
    """
    NETr: Neuron Embedding Transformer
    """

    def __init__(
        self,
        # for ablation experiments
        use_local_features: bool,
        use_positional_encoding: bool,
        use_transformer: bool,
        # local feature
        lf_patch_size_xy: int,
        lf_patch_size_z: int,
        lf_pretrained: str,
        lf_weights: str | None,
        lf_finetune: bool,
        # positional encoding
        pe_num_mlp_layers: int,
        pe_norm: str | None,
        pe_scaling: str,
        pe_weights: str | None,
        # tracking transformer
        tr_d_model: int,
        tr_nhead: int,
        tr_num_encoder_layers: int,
        tr_dim_feedforward: int,
        tr_dropout: float,
        tr_activation: str,
        tr_weights: str | None,
    ):
        super(NETr, self).__init__()
        # Local feature extractor
        self.lf_patch_size_xy = lf_patch_size_xy
        self.lf_patch_size_z = lf_patch_size_z
        self.lf_pretrained = lf_pretrained
        self.lf_weights = lf_weights
        self.lf_finetune = lf_finetune
        self.use_local_features = use_local_features

        self.lf_module = None
        self.lf_proj = None

        if use_local_features:
            self.lf_module = LocalFeatureViT(
                lf_patch_size_xy, lf_patch_size_z, lf_pretrained
            )
            if lf_weights is not None and len(lf_weights) > 0 and lf_weights != "None":
                state_dict = torch.load(lf_weights)
                # replace "chvit." with "backbone."
                state_dict = {
                    k.replace("chvit.", "backbone."): v for k, v in state_dict.items()
                }
                self.lf_module.load_state_dict(state_dict)

            self.lf_proj = nn.Linear(self.lf_module.out_features, tr_d_model)
            # nn.Sequential(
            #    nn.Linear(self.lf_module.out_features, tr_d_model),
            # nn.ReLU(),
            # )
            if not self.lf_finetune:
                for param in self.lf_module.parameters():
                    param.requires_grad = False

        # MLP layer - position encoding
        self.pe_mlp = None
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pe_num_mlp_layers = pe_num_mlp_layers
            self.pe_norm = pe_norm
            self.pe_scaling = pe_scaling
            self.pe_mlp = build_pe_mlp(pe_num_mlp_layers, 3, tr_d_model, pe_norm)
            if pe_weights is not None and len(pe_weights) > 0 and pe_weights != "None":
                state_dict = torch.load(pe_weights)
                # replace "pe_mlp." with ""
                state_dict = {
                    k.replace("pe_mlp.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("pe_mlp.")
                }
                self.pe_mlp.load_state_dict(state_dict)
            # else:
            #     self.pe_mlp.apply(
            #         lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Conv1d) else None
            #     )

        # Transformer encoder
        self.use_transformer = use_transformer
        self.tr_d_model = tr_d_model
        self.encoder = None
        if use_transformer:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=tr_d_model,
                    nhead=tr_nhead,
                    dim_feedforward=tr_dim_feedforward,
                    dropout=tr_dropout,
                    activation=tr_activation,
                ),
                num_layers=tr_num_encoder_layers,
            )
            if tr_weights is not None and len(tr_weights) > 0 and tr_weights != "None":
                state_dict = torch.load(tr_weights)
                # replace "encoder." with ""
                state_dict = {
                    k.replace("encoder.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("encoder.")
                }
                self.encoder.load_state_dict(state_dict)

    def forward(self, x: dict):
        device = next(self.parameters()).device
        if device != x["image"].device:
            x = to_device(x, device)

        # N = Batch size (frames)
        # S = Number of objects per frame
        object_ids = x["object_ids"]
        N, S = object_ids.shape
        local_features = None
        # Local feature extraction
        if self.lf_module is not None:
            assert isinstance(self.lf_proj, torch.nn.Module), ""
            local_features = self.lf_module(x)  # (N, S, C_lf)
            local_features = local_features.view(
                -1, local_features.shape[-1]
            )  # (NS, C_lf)
            local_features = self.lf_proj(local_features)  # (NS, D)
            local_features = local_features.view(N, S, -1)  # (N, S, D)

        pos_encoding = None
        # Position encoding
        if self.pe_mlp is not None:
            with torch.no_grad():
                pos_encoding = []
                D, H, W = x["image"].shape[-3:]
                if self.pe_scaling == "physical":
                    center = torch.tensor(
                        [0, 0, 0], dtype=torch.float, device=device
                    )  # (3,)
                    scaling = x["spacing"]  # (N, 3)
                elif self.pe_scaling == "relative":
                    center = torch.tensor(
                        [D / 2.0, H / 2.0, W / 2.0], dtype=torch.float, device=device
                    )  # (3,)
                    scale_x = torch.tensor(
                        2 / x["image"].shape[-1], dtype=torch.float, device=device
                    )
                    scale_y = torch.tensor(
                        2 / x["image"].shape[-2], dtype=torch.float, device=device
                    )
                    space_xy = torch.mean(x["spacing"][..., 1:3], dim=-1)  # (N,)
                    scale_z = (
                        (scale_x + scale_y) / 2 * x["spacing"][..., 0] / space_xy
                    )  # (N,)
                    scaling = torch.stack(
                        [scale_z, scale_y.repeat(N), scale_x.repeat(N)],
                        dim=-1,
                    )  # (N, 3)
                else:
                    raise ValueError(f"Invalid pe_scaling: {self.pe_scaling}")

                for i in range(N):
                    pos_encoding.append(
                        (x["coords"][i] - center[None, :]) * scaling[i][None, :]
                    )
                pos_encoding = torch.stack(pos_encoding, dim=0)  # (N, S, 3)
                pos_encoding = pos_encoding.permute(0, 2, 1)  # (N, 3, S)
            pos_encoding = self.pe_mlp(pos_encoding)  # (N, D, S)
            pos_encoding = pos_encoding.permute(0, 2, 1)  # (N, S, D)

        # Transformer encoder
        if local_features is not None and pos_encoding is not None:
            neuron_features = local_features + pos_encoding  # (N, S, D)
        elif local_features is not None:
            neuron_features = local_features
        elif pos_encoding is not None:
            neuron_features = pos_encoding
        else:
            raise Exception(
                "At least one of local features or positional encoding should be used."
            )

        # Prepare the key padding mask if object_ids are provided.
        # object_ids should have shape (N,S)
        # Create a boolean mask: True for tokens to mask.
        key_padding_mask = object_ids == -1  # shape: (N,S)

        if self.encoder is not None:
            # Transformer encoder takes input in (S+1, N, D) format for historical reasons.
            neuron_features = neuron_features.permute(1, 0, 2)  # (S+1, N, D)
            matching_features = self.encoder(
                neuron_features, src_key_padding_mask=key_padding_mask
            )
            matching_features = matching_features.permute(1, 0, 2)  # (N, S+1, D)
        else:
            matching_features = neuron_features

        # Set nan to the masked tokens
        matching_features[key_padding_mask] = float("nan")

        return matching_features


if __name__ == "__main__":
    pass
