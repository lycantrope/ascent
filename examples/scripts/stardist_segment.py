#!/usr/bin/env python
"""Segment a 3‑D time‑lapse HDF5 stack with StarDist 3‑D and export
instance masks + centroid CSV compatible with ASCENT.

Input file layout
------------
Root group
└── t{frame}      (HDF5 group)
    └── c{channel} 3-D dataset (Z,Y,X)

Output files    
-----------
HDF5 mask file: one dataset per frame (`str(t)`) containing
                `labels[z, y, x]` (uint32)
CSV file with columns: object_id,t,z,y,x
    
Example
-------
conda activate stardist
python stardist_segment.py \
    --input raw_video.h5 \
    --input_channel 0 \
    --input_axis_order ZYX \
    --modelpath ~/models/celegans-free-NeRVE \
    --normalize 1 99.99 \
    --output_mask seg_masks.h5 \
    --output_centroids neuron_centroids.csv
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.measure import regionprops
from stardist.models import StarDist3D

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _parse_axis_order(order: str) -> tuple[int, int, int]:
    """Return tuple mapping (z, y, x) indices for given axis string."""
    order = order.upper()
    mapping = {ax: i for i, ax in enumerate(order)}
    return mapping["Z"], mapping["Y"], mapping["X"]


def _iter_frame_keys(h5: h5py.File) -> list[str]:
    frames = [k for k in h5.keys() if k.startswith("t")]
    # sort by integer time index
    frames.sort(key=lambda k: int(k[1:]))
    return frames


# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------


def segment_video(
    input_path: Path,
    model_path: Path,
    output_mask: Path,
    output_csv: Path,
    channel: int = 0,
    axis_order: str = "ZYX",
    pmin: float = 1.0,
    pmax: float = 99.99,
):
    z_idx, y_idx, x_idx = _parse_axis_order(axis_order)

    input_path = Path(input_path)
    model_path = Path(model_path)
    output_mask = Path(output_mask)
    output_csv = Path(output_csv)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load model (will look for config + weights in dir)
    print(f"[StarDist] loading model from {model_path}")
    print(f"Name: {model_path.name}  parent: {model_path.parent}")
    model = StarDist3D(None, name=model_path.name, basedir=str(model_path.parent))

    object_id_counter = 0
    offsets = []
    rows = []  # collect CSV rows

    t_start = _dt.datetime.now()
    with h5py.File(str(input_path), "r") as h5_in, h5py.File(str(output_mask), "w") as h5_out:
        frame_keys = _iter_frame_keys(h5_in)
        n_frame = len(frame_keys)

        for fidx, key in enumerate(frame_keys):
            t = int(key[1:])
            print(f"[StarDist] frame {fidx + 1}/{n_frame} (t={t})")

            img = h5_in[key][f"c{channel}"][:]
            # reorder axes to ZYX as StarDist expects
            img = np.moveaxis(img, (z_idx, y_idx, x_idx), (0, 1, 2))
            img = normalize(img, pmin, pmax, axis=(0, 1, 2))

            labels, _ = model.predict_instances(img)
            props = regionprops(labels)

            # record offset before this frame's detections
            offsets.append(object_id_counter)

            for prop in props:
                cz, cy, cx = prop.centroid
                min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
                rows.append(
                    {
                        "object_id": object_id_counter,
                        "t": t,
                        "z": cz,
                        "y": cy,
                        "x": cx,
                        "r_z": (max_z - min_z) / 2.0,
                        "r_y": (max_y - min_y) / 2.0,
                        "r_x": (max_x - min_x) / 2.0,
                    }
                )
                object_id_counter += 1

            # save labels (stored as uint16/32 depending on max label)
            ds = h5_out.create_dataset(str(t), data=labels.astype(np.uint32), compression="gzip")
            ds.attrs["axis_order"] = "ZYX"

        # store offsets so downstream tools can map frame → starting object id
        h5_out.create_dataset("oid_offset", data=np.array(offsets, dtype=np.uint32))

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    dt = _dt.datetime.now() - t_start
    print(f"Finished. Frames: {n_frame}  Objects: {object_id_counter}  Time: {dt}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run StarDist 3‑D segmentation on HDF5 timelapse.")
    p.add_argument("--input", required=True, help="path to input HDF5 video")
    p.add_argument(
        "--input_channel", type=int, default=0, help="channel index to segment (default: 0)"
    )
    p.add_argument(
        "--input_axis_order",
        default="ZYX",
        help="axis order in each volume (e.g., ZYX or YXZ)",
        choices=["ZYX", "ZXY", "YZX", "YXZ", "XYZ", "XZY"],
    )
    p.add_argument(
        "--modelpath", required=True, help="directory containing StarDist config & weights"
    )
    p.add_argument(
        "--normalize",
        nargs=2,
        type=float,
        default=[1.0, 99.99],
        metavar=("P_MIN", "P_MAX"),
        help="percentile range for intensity normalisation",
    )
    p.add_argument("--output_mask", required=True, help="output HDF5 mask file")
    p.add_argument("--output_centroids", required=True, help="output CSV with centroid coords")
    return p


def main():
    args = _build_parser().parse_args()
    segment_video(
        input_path=Path(args.input),
        model_path=Path(args.modelpath),
        output_mask=Path(args.output_mask),
        output_csv=Path(args.output_centroids),
        channel=args.input_channel,
        axis_order=args.input_axis_order,
        pmin=args.normalize[0],
        pmax=args.normalize[1],
    )


if __name__ == "__main__":
    main()
