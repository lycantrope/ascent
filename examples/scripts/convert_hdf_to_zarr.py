import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import zarr
import zarr.storage
from numcodecs import Blosc

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


def convert_h5_to_zarr(h5_path: Path, zarr_path: Path, compression: bool = True):
    with (
        h5py.File(h5_path, "r") as h5_file,
        zarr.storage.ZipStore(zarr_path, mode="w") as store,
    ):

        # t0, ... tn
        t_keys = (k for k in h5_file.keys() if str(k).startswith("t"))
        t_keys = sorted(t_keys, key=lambda x: int(x[1:]))

        root = zarr.group(store=store)
        for t in t_keys:
            src_grp = h5_file[t]
            assert isinstance(src_grp, h5py.Group), "This one should be group"
            dst_grp = root.create_group(t)

            for ch in src_grp.keys():
                vol = np.asarray(src_grp[ch])

                compressor = Blosc(
                    cname="zstd",
                    clevel=5,
                    shuffle=Blosc.SHUFFLE,
                )
                if not compression:
                    compressor = None
                dst_grp.create_dataset(
                    ch, chunks=vol.shape, data=vol, compressor=compressor
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("hdf_list", nargs="+", type=str)
    parser.add_argument("--no_compression", action="store_true", default=False)

    args = parser.parse_args()
    #
    hdf_list = args.hdf_list
    hdf_list = list(map(Path, hdf_list))
    hdf_list = [x for x in hdf_list if x.name.endswith(".h5")]
    if not hdf_list:
        parser.error(f"No valid hdf file found: {args._hdf_list}")

    for i, h5_path in enumerate(hdf_list, 1):
        print(f"[{i}/{len(hdf_list)}] Convert {h5_path} to zarr...")
        zarr_path = h5_path.with_suffix(".zarr.zip")
        if args.no_compression:
            convert_h5_to_zarr(h5_path, zarr_path, False)
        else:
            convert_h5_to_zarr(h5_path, zarr_path)
