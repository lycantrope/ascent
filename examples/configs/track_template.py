# ---------------- DATASET ----------------
dataset_file_image = "path-to-your-image-data.h5"
dataset_file_coord = "path-to-your-detection-data.csv"
dataset_image_channel = 0
dataset_axis_order = "ZYX"
dataset_spacing = [1.26, 0.36, 0.36]  # physical spacing of the dataset in micrometers
dataset_normalize = "percentile"  # Options: 'none', 'percentile'
dataset_norm_p_low = 1.0  # Lower percentile for normalization
dataset_norm_p_high = 99.99  # Upper percentile for normalization

# ---------------- MODEL (NETr) -----------
model_ckpt = "path-to-your-netr-model.pth"
model_lf_patch_size_xy = 64
model_lf_patch_size_z = 3
model_lf_pretrained = "imagenet_channelvit_small_p16_DINO"
model_pe_num_mlp_layers = 3
model_pe_norm = "batch"
model_pe_scaling = "relative"
model_tr_d_model = 256
model_tr_nhead = 4
model_tr_num_encoder_layers = 4
model_tr_dim_feedforward = 512
model_tr_dropout = 0.1
model_tr_activation = "relu"
model_use_local_features = True
model_use_positional_encoding = True
model_use_transformer = True

# ---------------- TRACKING ---------------
tracking_momentum = 0.5
tracking_temperature = 0.05
tracking_max_gap_frames = 9999
tracking_w_within = 0

# ---------------- RUNTIME ----------------
runtime_output_dir = "test"
runtime_output_prefix = "sample"
runtime_batch_size_frame = 64
runtime_device = "cuda"
runtime_loglevel = "INFO"
