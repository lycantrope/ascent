# Configuration for training
from torch.optim import AdamW

from ascent.datasets.tracking_dataset import ObjectEmbeddingDataset3D
from ascent.datasets.transformations import (
    OEDColorJitter3D,
    OEDElasticDeformRandomGrid2Dfor3D,
    OEDObjectDropout,
    OEDObjectPositionJitter3D,
    OEDRandomResizedCropXY,
    OEDRandomRotation3D_XYplane,
)
from ascent.models.loss import NT_Xent
from ascent.models.netr import NETr

device = "auto"
port = 12345

model = {
    "class": NETr,  # Model class, will be initialized with params below
    "params": {
        "use_local_features": True,
        "use_positional_encoding": True,
        "use_transformer": True,
        "lf_patch_size_xy": 64,
        "lf_patch_size_z": 3,
        "lf_pretrained": "imagenet_channelvit_small_p16_DINO",
        "lf_weights": None,
        "lf_finetune": True,
        "pe_num_mlp_layers": 3,
        "pe_norm": "batch",
        "pe_scaling": "relative",
        "pe_weights": None,
        "tr_d_model": 256,
        "tr_nhead": 4,
        "tr_num_encoder_layers": 4,
        "tr_dim_feedforward": 512,
        "tr_dropout": 0.1,
        "tr_activation": "relu",
        "tr_weights": None,
    },
}

image_file = "path-to-your-image-data.h5"
coord_file = "path-to-your-detection-data.csv"
dataset = {
    "class": ObjectEmbeddingDataset3D,  # Dataset class, will be initialized with params below
    "params": {
        "image_file": image_file,
        "coord_file": coord_file,
        "image_channel": 0,
        "axis_order": "ZYX",
        "spacing": (1.5, 0.243, 0.243),
        "normalize": "percentile",
        "norm_p_low": 1,
        "norm_p_high": 99.99,
        "lazy_loading": True,
    },
}

transforms = [
    (
        OEDElasticDeformRandomGrid2Dfor3D,
        {"sigma": 15, "points": 3, "mode": "border", "order": 1, "allow_dropout": True},
    ),
    (OEDObjectPositionJitter3D, {"xy_jitter_range": 5.0, "z_jitter_range": 1.0}),
    (OEDRandomRotation3D_XYplane, {"degrees": 180, "allow_dropout": True}),
    (
        OEDRandomResizedCropXY,
        {
            "scale_x": (0.9, 1.1),
            "scale_y": (0.9, 1.1),
            "crop_size": (224, 448),
            "allow_dropout": True,
        },
    ),
    (OEDColorJitter3D, {"brightness": 0.2, "contrast": 0.2}),
    (OEDObjectDropout, {"p": 0.1}),
]


# Configuration for data loader. Adjust batch sizes and shuffling as necessary.
dataloader = {
    "num_workers": 6,
    "batch_size": 4,
}

# Loss function for training.
losses = [
    {
        "class": NT_Xent,
        "params": {"temperature": 5e-02},
        "weight": 1.0,
    },
]

optimizer = {
    "class": AdamW,  # Optimizer class
    "params": {"lr": 1e-04},
}


# scheduler = {
#     "class": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
#     "params": {
#         "T_0": 100,
#     },
# }

epochs = 50  # Total number of training epochs.
time_limit = 3600  # Optional time limit for training (in seconds). Comment out if needed.
save_every_n_epochs = 1  # Save model every n epochs
save_time_span = None  # Save model every n seconds

model_save_path = "path-to-your-netr-model.pth"  # Path to save the final model.
logfile = "path-to-your-log-file.log"  # Log file to record training and validation progress.
tensorboard_logdir = "path-to-your-tensorboard-directory/"  # Directory for TensorBoard logs.

# continue_training = True
