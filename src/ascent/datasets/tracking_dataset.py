import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import zarr
import zarr.storage
from torch.utils.data import Dataset


class OEDItem(dict):
    def __init__(
        self,
        t: int,
        image: torch.Tensor,
        object_ids: torch.Tensor,
        coords: torch.Tensor,
        spacing: torch.Tensor = torch.tensor([1.0, 1.0, 1.0]),
        n_max_objects: int | None = None,
    ):
        super().__init__()
        assert torch.is_tensor(image), "image must be a tensor"

        self["t"] = t
        self["image"] = image

        # padding to max_objects
        if n_max_objects is not None and len(object_ids) < n_max_objects:
            object_ids = torch.cat(
                [
                    object_ids,
                    torch.ones(n_max_objects - len(object_ids), dtype=torch.int64) * -1,
                ],
                dim=0,
            )
            coords = torch.cat(
                [coords, torch.ones(n_max_objects - len(coords), 3) * -1], dim=0
            )

        self["object_ids"] = object_ids
        self["coords"] = coords
        self["spacing"] = spacing

    def new_item(self, **kwargs):
        new = OEDItem(
            t=kwargs.get("t", self["t"]),
            image=kwargs.get("image", self["image"]),
            object_ids=kwargs.get("object_ids", self["object_ids"]),
            coords=kwargs.get("coords", self["coords"]),
            spacing=kwargs.get("spacing", self["spacing"]),
            n_max_objects=kwargs.get("n_max_objects", None),
        )
        return new

    @staticmethod
    def get_item(batched_x, idx):
        assert (
            len(batched_x["image"].shape) == 5
        ), "get_item is only supported for batched OEDItems"
        new = OEDItem(
            t=batched_x["t"][idx],
            image=batched_x["image"][idx],
            object_ids=batched_x["object_ids"][idx],
            coords=batched_x["coords"][idx],
            spacing=batched_x["spacing"][idx],
            n_max_objects=None,
        )
        return new


# class ZarrSequence:
#     def __init__(
#         self,
#         zarr_path: str | os.PathLike,
#         image_channel=0,
#         axis_order: str = "ZYX",
#     ):
#         self._data = None
#         # This is for one dataset contains (T,Z,Y,X)
#         self.zarr_path = Path(zarr_path)
#         assert self.zarr_path.name.endswith((".zarr", ".zarr.zip"))

#         self.is_zip = self.zarr_path.name.endswith(".zarr.zip")

#         self._data = (
#             zarr.storage.ZipStore(self.zarr_path, mode="r")
#             if self.is_zip
#             else self.zarr_path
#         )

#         handler = zarr.open_group(self._data, mode="r")

#         keys = (k for k in handler.keys() if str(k).startswith("t"))
#         # [t0, t1, t2, t..., tn]
#         keys = sorted(keys, key=lambda x: int(x[1:]))

#         try:
#             # Verify dataset here
#             assert len(keys) > 0, "No image seqeunce found in Zarr"
#             k = next(iter(keys))
#             dset = handler[k]
#             assert isinstance(dset, zarr.Group), "Must be group"
#             assert (
#                 f"c{image_channel}" in dset.keys()
#             ), f"No channel found in dataset: {list(dset.keys())}"

#             shape = tuple(np.asarray(dset[f"c{image_channel}"]).shape)
#             assert len(shape) == 3, "Must be 3D image"
#         finally:
#             self.close()

#         self.keys = {int(k[1:]): f"{k}/c{image_channel}" for k in keys}
#         self.shape = tuple(shape["ZYX".index(a)] for a in axis_order)
#         self.max_frame = max(self.keys.keys())

#     def get_filepath(self) -> Path:
#         return self.zarr_path

#     def init(self) -> None:
#         if self._data is None:
#             self._data = (
#                 zarr.storage.ZipStore(self.zarr_path, mode="r")
#                 if self.is_zip
#                 else self.zarr_path
#             )
#             handler = zarr.open_group(self._data, mode="r")
#             tmp = {k: handler[v] for k, v in self.keys.items()}

#             self._data_sequence = {
#                 k: arr for k, arr in tmp.items() if isinstance(arr, zarr.Array)
#             }

#     def close(self) -> None:
#         if self._data is not None:
#             if isinstance(self._data, zarr.storage.ZipStore):
#                 try:
#                     self._data.close()
#                 except TypeError as _:
#                     # h5py\_hl\files.py", line 631, in close
#                     # Error in TypeError: bad operand type for unary ~: 'NoneType'
#                     pass
#             self._data = None
#             self._data_sequence = {}

#     def __len__(self) -> int:
#         return len(self.keys)

#     def __getstate__(self):
#         # This garantee everything can be pickled.
#         if self.data is not None:
#             self.close()
#         return super().__getstate__()

#     def __del__(self):
#         self.close()


#     @property
#     def data(
#         self,
#     ) -> dict[int, zarr.Array]:
#         self.init()
#         return self._data_sequence
class ZarrStack:
    def __init__(
        self,
        zarr_path: str | os.PathLike,
        image_channel=0,
        axis_order: str = "ZYX",
    ):
        self.data = None
        # This is for one dataset contains (T,Z,Y,X)
        self.zarr_path = Path(zarr_path)
        assert self.zarr_path.name.endswith((".zarr", ".zarr.zip"))

        self.is_zip = self.zarr_path.name.endswith(".zarr.zip")

        self.store = (
            zarr.storage.ZipStore(self.zarr_path, mode="r")
            if self.is_zip
            else self.zarr_path
        )
        self.data = zarr.open_array(self.store, mode="r")

        self.image_channel = image_channel
        assert image_channel < self.data.shape[1], "image_channel is greater than data"
        self.max_frame = self.data.shape[0]

        shape = self.data.shape[2:]  # only z, y, x
        self.shape = tuple(shape["ZYX".index(a)] for a in axis_order)
        self.close()

    def get_filepath(self) -> Path:
        return self.zarr_path

    def init(self) -> None:
        if self.data is None:
            self.store = (
                zarr.storage.ZipStore(self.zarr_path, mode="r")
                if self.is_zip
                else self.zarr_path
            )
            self.data = zarr.open_array(self.store, mode="r")

    def close(self) -> None:
        if self.data is not None:
            if self.is_zip and isinstance(self.store, zarr.storage.ZipStore):
                self.store.close()
            self.data = None

    def __len__(self) -> int:
        return self.max_frame

    def __getstate__(self):
        # This garantee everything can be pickled.
        if self.data is not None:
            self.close()
        return super().__getstate__()

    def __del__(self):
        self.close()

    def get_image_at(self, t: int) -> np.ndarray:
        if self.data is None:
            self.init()
        assert self.data is not None, "Must be accessed after init"
        return np.ascontiguousarray(self.data[t, self.image_channel])


class ObjectEmbeddingDataset3D(Dataset):
    def __init__(
        self,
        image_file: str,
        coord_file: str,
        image_channel: int = 0,
        axis_order: str = "ZYX",
        spacing: torch.Tensor | Sequence[float] = torch.tensor([1.0, 1.0, 1.0]),
        normalize: str = "none",
        norm_p_low: float = 1.0,
        norm_p_high: float = 99.99,
        lazy_loading: bool = False,
        frame_sample_method: str | None = None,
        frame_sample_size: int | None = None,
        empty_image: bool = False,
    ):
        """
        Initialize the dataset with a microscopy volume stack and corresponding object coordinates.

        Args:
            image_file (str): Path to the volume stack image file with TCZYX dimensions.
            coord_file (str): Path to the file containing object coordinates.
            image_channel (int, optional): Specific channel to be included from the image stack.
            axis_order (str, optional): Axis order of the image stack (default is 'ZYX').
            spacing (torch.Tensor or Sequence[float]): Spacing between pixels in the image stack.
            normalize (str, optional): Normalization method for the image data. Options are 'none' or 'percentile'.
            norm_p_low (float, optional): Lower percentile for normalization (default is 1.0).
            norm_p_high (float, optional): Upper percentile for normalization (default is 99.99).
            lazy_loading (bool): Flag to enable lazy loading of the image stack.
            frame_sample_method (str, optional): Method to sample frames from the dataset.
            frame_sample_size (int, optional): Number of frames to sample from the dataset.
            empty_image (bool): If True, returns an empty image tensor with the correct dimensions.
                Useful for getting the image dimensions without loading actual data.

        Normalization
        -------------
        Optional per-frame percentile normalization:
            - Set `normalize="percentile"` to enable.
            - `norm_p_low` and `norm_p_high` define the lower/upper percentiles (default 1 and 99.99).
            - We *do not clip* after normalization; values outside [0, 1] are retained.
            - Implemented with torch.quantile for speed and device locality.
        """
        super().__init__()
        logging.info(f"Loading dataset from {image_file} and {coord_file}")
        # Load the entire stack into memory (TCZYX)
        self.image_file = image_file
        self.image_channel = image_channel
        axis_order = axis_order.upper()
        assert set(axis_order) == {"Z", "Y", "X"}
        self._axis_permute = tuple("ZYX".index(a) for a in axis_order)
        if isinstance(spacing, Sequence):
            spacing = torch.tensor(spacing, dtype=torch.float32)
        self.spacing = spacing
        assert image_file.endswith(
            (".h5", ".zarr", ".zarr.zip")
        ), f"Unsupported image file format: {image_file}"
        self.normalize = normalize
        self.norm_p_low = float(norm_p_low)
        self.norm_p_high = float(norm_p_high)
        if self.normalize not in ("none", "percentile"):
            raise ValueError(
                f"normalize must be 'none' or 'percentile', got {self.normalize}"
            )
        if not (
            self.norm_p_low < self.norm_p_high
            and 0.0 <= self.norm_p_low
            and self.norm_p_high <= 100.0
        ):
            raise ValueError("Require 0 <= norm_p_low < norm_p_high <= 100")
        self.empty_image = empty_image
        if empty_image:
            self.image_dim = None

        self.zarr_data = None
        if image_file.endswith(".h5"):
            # lazy loading
            with h5py.File(image_file, "r") as f:
                list_keys = [int(k[1:]) for k in f.keys() if str(k).startswith("t")]
            assert len(list_keys) > 0, "No frame found"
            self.max_frame = max(list_keys) + 1
            self.image_stack = [None] * self.max_frame

            self.load_objects(coord_file, frame_sample_method, frame_sample_size)
            # Load all images into memory if not using lazy loading
            if not lazy_loading:
                with h5py.File(image_file, "r") as f:
                    for t in self.frame_list:
                        self.get_image_at(t)
        else:
            self.zarr_data = ZarrStack(image_file, image_channel)
            self.max_frame = self.zarr_data.max_frame
            # load_objects
            self.load_objects(coord_file, frame_sample_method, frame_sample_size)

    @staticmethod
    def _percentile_normalize(
        x: torch.Tensor, p_low: float, p_high: float
    ) -> torch.Tensor:
        """
        Percentile normalization without post-clipping.

        Scales x s.t. p_low -> 0 and p_high -> 1 using:
            x' = (x - q_low) / (q_high - q_low)
        where q_low = quantile(x, p_low/100), q_high = quantile(x, p_high/100).

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor for a single frame/volume. Any shape, will be flattened for quantiles.
        p_low, p_high : float
            Percentiles in [0, 100], with p_low < p_high.

        Returns
        -------
        torch.Tensor
            Normalized tensor (not clipped).
        """
        if not x.is_floating_point():
            x = x.float()
        # Compute quantiles on the same device/dtype as x
        q = torch.tensor(
            [p_low / 100.0, p_high / 100.0], device=x.device, dtype=x.dtype
        )
        q_low, q_high = torch.quantile(x.reshape(-1), q)
        return (x - q_low) / (q_high - q_low)

    def _maybe_normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.normalize == "percentile":
            return self._percentile_normalize(image, self.norm_p_low, self.norm_p_high)
        return image

    def load_objects(self, coord_file, frame_sample_method, frame_sample_size):
        # Assuming coordinates are stored in a CSV format: object_id, frame_index, z_coord, x_coord, y_coord
        self.objects_by_frame = defaultdict(list)
        frame_set = set()
        self.max_objects = 0
        with open(coord_file, "r") as file:
            file.readline()  # Skip header
            for line in file:
                if line == "":
                    continue
                tokens = line.strip().split(",")
                object_id, frame, z, y, x = tokens[:5]

                try:
                    frame = int(frame)
                except ValueError:
                    frame = int(float(frame))
                object_id = int(object_id)
                z, y, x = float(z), float(y), float(x)
                assert frame <= self.max_frame, f"Frame index {frame} out of bounds"
                self.objects_by_frame[frame].append((object_id, frame, z, y, x))
                n_obj = len(self.objects_by_frame[frame])
                self.max_objects = max(n_obj, self.max_objects)
                frame_set.add(frame)

        self.frame_list = sorted(frame_set)

        # sample frames
        if frame_sample_method not in (None, "regular", "first"):
            raise ValueError(f"Unsupported frame sample method: {frame_sample_method}")

        if frame_sample_method is None:
            return

        assert (
            isinstance(frame_sample_size, int) and frame_sample_size > 0
        ), "frame_sample_size must be natural number"

        logging.info(
            f"Sample {frame_sample_size} / {len(self.frame_list)} frames using {frame_sample_method}"
        )

        if frame_sample_method == "regular":
            sample_interval = len(self.frame_list) // frame_sample_size
            self.frame_list = self.frame_list[::sample_interval][:frame_sample_size]
            assert (
                len(self.frame_list) == frame_sample_size
            ), "Sampled frames do not match the expected size"
        elif frame_sample_method == "first":
            self.frame_list = self.frame_list[:frame_sample_size]

        self.objects_by_frame = {
            frame: self.objects_by_frame[frame] for frame in self.frame_list
        }
        self.max_objects = max(
            [len(self.objects_by_frame[frame]) for frame in self.frame_list]
        )
        logging.info(f"Sampled frames: {self.frame_list}")

    def get_image_at(self, t: int) -> torch.Tensor:
        if self.zarr_data is not None:
            if self.empty_image:
                shape = (
                    self.image_dim
                    if self.image_dim is not None
                    else (0, *self.zarr_data.shape)
                )
                return torch.empty(shape, dtype=torch.float32)
            vol = self.zarr_data.get_image_at(t)
            vol = np.moveaxis(vol, self._axis_permute, (0, 1, 2))
            # add channel dim
            vol = np.expand_dims(vol, axis=0).astype("f4")
            vol = torch.tensor(vol, dtype=torch.float32)
            return self._maybe_normalize(vol)
        else:
            # Method to fetch the image at time t
            if self.image_stack[t] is None:
                # lazy loading
                assert self.image_file.split(".")[-1] == "h5"
                if self.empty_image:
                    if self.image_dim is None:
                        with h5py.File(self.image_file, "r") as f:
                            vol = f[f"t{t}/c{self.image_channel}"][:]
                            # re-order axes to Z Y X
                            vol = np.moveaxis(vol, self._axis_permute, (0, 1, 2))
                            self.image_dim = vol.shape

                    # create empty image - it will be used for getting the dimension of the image
                    self.image_stack[t] = torch.empty(
                        0, self.image_dim[-3], self.image_dim[-2], self.image_dim[-1]
                    )
                else:
                    with h5py.File(self.image_file, "r") as f:
                        vol = f[f"t{t}/c{self.image_channel}"][:]
                        # re-order axes to Z Y X
                        vol = np.moveaxis(vol, self._axis_permute, (0, 1, 2))
                        # add channel dim
                        vol = np.expand_dims(vol, axis=0)
                        vol = torch.tensor(vol, dtype=torch.float32)
                        # normalize per loaded frame (no clipping afterwards)
                        vol = self._maybe_normalize(vol)
                        self.image_stack[t] = vol

            return self.image_stack[t]

    def get_frame(self, idx):
        return self.frame_list[idx]

    def __getitem__(self, idx):
        # frame_idx
        t = self.frame_list[idx]
        image = self.get_image_at(t)
        # collate all the objects in idx frame
        objects = self.objects_by_frame[t]
        object_ids = torch.tensor([obj[0] for obj in objects], dtype=torch.int64)
        coords = torch.tensor(
            [(z, y, x) for obj_id, t, z, y, x in objects],
            dtype=torch.float32,
        )
        return OEDItem(t, image, object_ids, coords, self.spacing, self.max_objects)

    def __len__(self):
        return len(self.objects_by_frame.keys())
