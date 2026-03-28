import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ascent.datasets.tracking_dataset import ObjectEmbeddingDataset3D

home = Path(r"D:\kuan\zeng-nwb\TOY21\251203-1DA\activity\W4\hdf-sequence")


hdf_path = home / "crop-W4-activity.h5"
zarr_path = home / "crop-W4-activity.zarr.zip"

coord_files = home / "crop-W4-activity_peaks.csv"


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    return data


def verification():
    print("start verification")
    hdf_dset = ObjectEmbeddingDataset3D(
        image_file=str(hdf_path),
        coord_file=str(coord_files),
        lazy_loading=True,
    )
    zarr_dset = ObjectEmbeddingDataset3D(
        image_file=str(zarr_path),
        coord_file=str(coord_files),
    )

    hdf_loader = DataLoader(hdf_dset, batch_size=1, shuffle=False, drop_last=False)
    zarr_loader = DataLoader(zarr_dset, batch_size=1, shuffle=False, drop_last=False)

    for i, (d1, d2) in enumerate(zip(hdf_loader, zarr_loader)):
        for (k1, v1), (k2, v2) in zip(d1.items(), d2.items()):
            try:
                np.testing.assert_equal(np.array(v1), np.array(v2))

            except AssertionError as e:
                print(f"{k1} is unequal")
                raise e
        if i > 10:
            print("No problem")
            break


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if DEVICE != "cuda":
        print("Cannot find GPU")

    verification()
    # Benchmark
    for shuffle in [False, True]:
        for num_workers in [1, 2, 4]:
            for batch_size in [1, 2, 4]:
                print(
                    f"Zarr|shuffle:{shuffle}|n_workers:{num_workers:d}|batch_size:{batch_size:d}",
                    end="",
                )

                zarr_dset = ObjectEmbeddingDataset3D(
                    image_file=str(zarr_path),
                    coord_file=str(coord_files),
                )
                loader = DataLoader(
                    zarr_dset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    drop_last=False,
                )
                t0 = time.perf_counter()
                for i, d in enumerate(loader):
                    to_device(d, DEVICE)
                    if i > 20:
                        break

                print(f"|elapsed time:{time.perf_counter()-t0:.3f}s")
                del zarr_dset, loader

    for shuffle in [False, True]:
        for num_workers in [1, 2, 4]:
            for batch_size in [1, 2, 4]:

                print(
                    f"HDF5|shuffle:{shuffle}|n_workers:{num_workers:d}|batch_size:{batch_size:d}",
                    end="",
                )
                hdf_dset = ObjectEmbeddingDataset3D(
                    image_file=str(hdf_path),
                    coord_file=str(coord_files),
                    lazy_loading=True,
                )

                loader = DataLoader(
                    hdf_dset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    drop_last=False,
                    prefetch_factor=2,
                )
                t0 = time.perf_counter()
                for i, d in enumerate(loader):
                    to_device(d, DEVICE)
                    if i > 20:
                        break
                print(f"|elapsed time:{time.perf_counter()-t0:.3f}s")
                del hdf_dset, loader
