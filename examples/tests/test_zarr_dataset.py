from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ascent.datasets.tracking_dataset import ObjectEmbeddingDataset3D

home = Path(r"D:\kuan\zeng-nwb\TOY21\251203-1DA\activity\W4\hdf-sequence")


hdf_path = home / "crop-W4-activity.h5"
zarr_path = home / "crop-W4-activity.zarr.zip"

coord_files = home / "crop-W4-activity_peaks.csv"

hdf_dset = ObjectEmbeddingDataset3D(
    image_file=str(hdf_path),
    coord_file=str(coord_files),
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
            np.testing.assert_equal(np.array(v1), v2)

        except AssertionError as e:
            print(f"{k1} is unequal")
            raise e
    if i > 10:
        print("No problem")
        break
