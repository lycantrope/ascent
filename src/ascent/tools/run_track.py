#!/usr/bin/env python
"""
ASCENT: feature extraction + Hungarian tracking
----------------------------------------------
Reads parameters from a YAML file and lets CLI flags override them.

Usage
-----
python tools/run_track.py --config configs/track_celegans.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from ascent.utils.common import load_config
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
