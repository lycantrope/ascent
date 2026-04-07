#!/usr/bin/env python
"""
ASCENT: feature extraction + Hungarian tracking
----------------------------------------------
Reads parameters from a YAML file and lets CLI flags override them.

Usage
-----
python tools/run_tracking.py --config configs/track_celegans.yaml \
       batch_size_frame 4 device cuda
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ascent.datasets.tracking_dataset import ObjectEmbeddingDataset3D
from ascent.models.netr import NETr
from ascent.utils.common import load_config, to_device
from ascent.utils.track.io import save_tracks_napari
from ascent.utils.track.tracker import HungarianTracker

# -----------------------------------------------------------------------------
# CLI: two-stage parsing
#   1) parse --config to know which file to load
#   2) load that file, inspect the dict, and add one argparse flag per key
# -----------------------------------------------------------------------------


def _infer_type(val: Any):
    """Return a callable suitable for argparse 'type=' that converts strings
    to the same type as *val*."""
    if isinstance(val, bool):
        # parse bools from typical strings: true/false/1/0/yes/no
        def _to_bool(x: str) -> bool:
            if x.lower() in {"true", "1", "yes", "y"}:
                return True
            if x.lower() in {"false", "0", "no", "n"}:
                return False
            raise argparse.ArgumentTypeError(f"Expect bool, got '{x}'")

        return _to_bool
    if isinstance(val, list):
        # list of scalars → let argparse collect with nargs="+"
        elem_type = _infer_type(val[0]) if val else str
        return lambda xs: [elem_type(t) for t in xs.split(",")]
    return type(val)


def parse_cli() -> dict:
    # -------- stage 1: we only know about --config --------
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", required=True, help="path/to/config.py")
    args, remaining = base.parse_known_args()

    cfg = load_config(args.config)

    # -------- stage 2: build a full parser from cfg keys --------
    parser = argparse.ArgumentParser(parents=[base], description="ASCENT tracking")
    for key, default in cfg.items():
        parser.add_argument(
            f"--{key}",
            type=_infer_type(default),
            default=default,
            help=f"(default: {default})",
        )

    full = parser.parse_args()  # parses sys.argv again
    # merge CLI values back into cfg dict
    for k in cfg.keys():
        cfg[k] = getattr(full, k)
    return cfg


def setup_logger(level: str, out_dir: Path, out_prefix: str) -> None:
    curtime = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | (%(levelname)s) %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                out_dir / f"{out_prefix}_run_{curtime}.log", mode="a", encoding="utf-8"
            ),
        ],
    )


def main() -> None:
    cfg = parse_cli()
    out_dir = Path(cfg["runtime_output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(cfg.get("loglevel", "INFO"), out_dir, cfg.get("runtime_output_prefix"))
    logging.info("ASCENT tracking started")
    logging.info(f"Full config:\n{cfg}")

    device = torch.device(cfg["runtime_device"])
    # ---------------- Dataset ----------------
    dataset = ObjectEmbeddingDataset3D(
        image_file=cfg["dataset_file_image"],
        coord_file=cfg["dataset_file_coord"],
        image_channel=cfg["dataset_image_channel"],
        axis_order=cfg["dataset_axis_order"],
        spacing=cfg["dataset_spacing"],
        normalize=cfg["dataset_normalize"],
        norm_p_low=cfg["dataset_norm_p_low"],
        norm_p_high=cfg["dataset_norm_p_high"],
        lazy_loading=True,
    )
    logging.info(f"Dataset loaded with {len(dataset)} frames")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["runtime_batch_size_frame"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    logging.info(
        f"Dataloader created with batch size {cfg['runtime_batch_size_frame']}"
    )

    # ---------------- Model ----------------
    model = NETr(
        lf_patch_size_xy=cfg["model_lf_patch_size_xy"],
        lf_patch_size_z=cfg["model_lf_patch_size_z"],
        lf_pretrained=cfg["model_lf_pretrained"],
        lf_weights=None,
        lf_finetune=False,
        pe_num_mlp_layers=cfg["model_pe_num_mlp_layers"],
        pe_norm=cfg["model_pe_norm"],
        pe_scaling=cfg["model_pe_scaling"],
        pe_weights=None,
        tr_d_model=cfg["model_tr_d_model"],
        tr_nhead=cfg["model_tr_nhead"],
        tr_num_encoder_layers=cfg["model_tr_num_encoder_layers"],
        tr_dim_feedforward=cfg["model_tr_dim_feedforward"],
        tr_dropout=cfg["model_tr_dropout"],
        tr_activation=cfg["model_tr_activation"],
        tr_weights=None,
        use_local_features=cfg["model_use_local_features"],
        use_positional_encoding=cfg["model_use_positional_encoding"],
        use_transformer=cfg["model_use_transformer"],
    ).to(device)
    logging.info(f"Model {model.__class__.__name__} created and moved to {device}")

    # load checkpoint
    ckpt = torch.load(cfg["model_ckpt"], map_location=device)
    model.load_state_dict(ckpt)
    logging.info(f"Model weights loaded from {cfg['model_ckpt']}")
    model.eval()

    # ---------------- Feature extraction ----------------
    pred_z, pred_oid = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logging.info(f"Processing batch {i + 1}/{len(dataloader)}")
            batch = to_device(batch, device)
            v = batch["object_ids"].view(-1)
            valid = v != -1

            z = model(batch).reshape(-1, cfg["model_tr_d_model"])
            z = z[valid]
            pred_z.append(z.cpu())
            pred_oid.extend(v[valid].tolist())

    pred_z = torch.cat(pred_z, dim=0)
    torch.save(pred_z, out_dir / f"{cfg['runtime_output_prefix']}_pred_z.pt")
    torch.save(pred_oid, out_dir / f"{cfg['runtime_output_prefix']}_pred_object_ids.pt")
    logging.info("Feature extraction completed")

    # ---------------- Tracking ----------------
    tracker = HungarianTracker(
        file_objects=cfg["dataset_file_coord"],
        file_z=(out_dir / f"{cfg['runtime_output_prefix']}_pred_z.pt"),
        file_object_ids=(
            out_dir / f"{cfg['runtime_output_prefix']}_pred_object_ids.pt"
        ),
        device=device,
        momentum=cfg["tracking_momentum"],
        temperature=cfg["tracking_temperature"],
    )
    logging.info("Tracker initialized")
    tracks = tracker.solve_dynamic_cutoff(
        cutoff_weight_within=cfg["tracking_w_within"],
        max_gap_frames=cfg["tracking_max_gap_frames"],
    )
    logging.info("Tracking completed")
    track_csv = out_dir / f"{cfg['runtime_output_prefix']}_tracks.csv"
    save_tracks_napari(track_csv, tracks)
    logging.info("Tracking saved to %s", track_csv)
    logging.info("ASCENT tracking finished")


if __name__ == "__main__":
    main()
