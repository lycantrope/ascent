import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
import torch
import torch.nn.functional as F

from ascent.utils.track.common import Spot, Track


class HT_Object(Spot):
    """
    Represents a single detected object (e.g., a neuron candidate) at a specific time/frame.

    Attributes
    ----------
    id : str
        Unique identifier of the object.
    t : int
        The frame/time index at which this object was detected.
    coords : tuple(float, float, float)
        Spatial coordinates (z, y, x) of the object.
    z : torch.Tensor or None
        Feature vector representing the object in the latent (volume) embedding space.

    Methods
    -------
    to(device: str)
        Move the object's feature vectors to the specified device.
    """

    __slots__ = ("z",)

    def __init__(
        self,
        spot_id: str,
        t: int,
        coord: tuple[float, float, float],
        z: torch.Tensor,
    ):
        super().__init__(spot_id, t, coord)
        self.z = z  # volume embedding

    def to(self, device):
        """Move the object's feature vectors to the specified device."""
        if self.z is not None:
            self.z = self.z.to(device)
        return self

    def __str__(self):
        return f"{self.id}: (t={self.t}, z={self.coord[0]:.0f}, y={self.coord[1]:.0f}, x={self.coord[2]:.0f})"

    def __repr__(self):
        return str(self)


class HT_Track(Track):
    """
    Represents a track that links a sequence of objects across multiple frames.

    The track representation is updated using a momentum-based moving average of the
    latest object's feature vector. This reduces memory usage by moving older objects off the GPU.

    Parameters
    ----------
    id : str
        Unique identifier for the track.
    object : HT_Object
        The initial object of the track.
    momentum : float, optional
        A value in [0, 1]. Higher means more inertia for old states.

    Attributes
    ----------
    id : str
        Track identifier.
    objects : list[HT_Object]
        List of all objects assigned to this track.
    v : torch.Tensor or None
        The representative volume embedding vector(s) for this track.
    momentum : float
        Momentum factor.

    Methods
    -------
    append(obj: HT_Object)
        Append a new object and update the representative vectors.
    """

    def __init__(
        self,
        track_id: str,
        object: HT_Object,
        momentum: float = 0.5,
        update_vectors: bool = True,
    ):
        super().__init__(track_id)
        self.objects: list[HT_Object] = []
        self.v = None  # representative volume embedding vector(s)
        self.momentum = momentum
        self.update_vectors = update_vectors

        self.append(object)

    def append(self, obj: HT_Object):
        """Append a new object to the track and update the representative vectors."""
        self.objects.append(obj)
        self.add(obj)

        if not self.update_vectors or obj.z is None:
            return

        # Update volume embedding representation if available.
        if self.v is None:
            self.v = obj.z.unsqueeze(0)
        else:
            self.v = self.momentum * self.v + (1 - self.momentum) * obj.z.unsqueeze(0)

        # self.objects[-1].to("cpu")

    def __str__(self):
        if len(self.objects) == 1:
            return f"Track {self.id}, Objects: [{self.objects[0]}]"
        elif len(self.objects) < 5:
            return f"Track {self.id}, Objects: [{', '.join([str(o) for o in self.objects])}]"
        else:
            return f"Track {self.id}, Objects: [{self.objects[0]}, ..., {self.objects[-1]}]"

    def __repr__(self):
        return str(self)


class MatrixNormalizer:
    def __init__(
        self,
        dist_norm_method: str = "standardize",
    ):
        """Set the distance normalization method for the combined cost matrix."""
        assert dist_norm_method in (
            "standardize",
            "minmax",
            "distribution",
            "none",
        ), f"Unknown dist_norm_method: {dist_norm_method}"

        self.dist_norm_method = dist_norm_method

    def __call__(self, cost_matrix: torch.Tensor, eps=1e-16) -> torch.Tensor:
        if self.dist_norm_method == "standardize":
            return (cost_matrix - torch.mean(cost_matrix)) / (
                torch.std(cost_matrix) + eps
            )
        elif self.dist_norm_method == "minmax":
            return (cost_matrix - torch.min(cost_matrix)) / (
                torch.max(cost_matrix) - torch.min(cost_matrix) + eps
            )
        elif self.dist_norm_method == "distribution":
            n_within = cost_matrix.shape[1]
            cost_matrix_flat = cost_matrix.flatten()
            cost_matrix_flat = cost_matrix_flat.sort()[0]
            mean_within = torch.mean(cost_matrix_flat[:n_within])
            mean_inter = torch.mean(cost_matrix_flat[n_within:])
            # make mean_within = 0, mean_inter = 1
            return (cost_matrix - mean_within) / (mean_inter - mean_within + eps)
        else:
            return cost_matrix


class HungarianTracker:
    """
    A tracker that uses the Hungarian algorithm for assigning objects to tracks frame-by-frame.
    This version always uses momentum mode for track representation.
    It can incorporate volume embeddings (using cosine similarity) and/or geometric descriptors
    (using Euclidean distance) for cost computation. The distance matrices from each modality are
    normalized and combined so that both contribute significantly.
    The "scale" parameter (a 3-tuple) is used to scale the z, y, x coordinates differently when
    computing geometric descriptors.

    Parameters
    ----------
    file_objects : str
        Path to a CSV file containing object information (object_id, t, x, y, z).
    file_z : str or None
        Path to a file containing the volume embedding feature vectors (.pt/.pth, .npy).
    file_object_ids : str or None
        Path to a file containing object IDs corresponding to the volume embedding vectors. (Optional)
    device : str | torch.device, optional
        Device to use for computations, e.g. 'cpu' or 'cuda'. Default is 'cpu'.
    scale : tuple of float, optional
        A 3-tuple (scale_z, scale_y, scale_x) to scale the (z, y, x) coordinates for geometric descriptors.
        Default is (1.0, 1.0, 1.0).
    **kwargs:
        Additional parameters (e.g., momentum, k for nnd, support_radius for local_pca/usc).

    Attributes
    ----------
    mode : str
        Always set to "momentum".
    device : str
        Computation device.
    scale : tuple of float
        Scaling factors for (z, y, x) coordinates.
    objects : list[HT_Object]
        List of all objects parsed.
    frame_index : list of (int, int)
        Start and end indices of objects per frame.
    tracks : list[HT_Track]
        Currently active tracks.
    track_id_num : int
        Counter for generating new track IDs.
    """

    def __init__(
        self,
        file_objects: str | os.PathLike,
        file_z: str | os.PathLike,
        file_object_ids: str | os.PathLike | None = None,
        device: str | torch.device = "cpu",
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        momentum: float = 0.5,
        temperature: float = 0.05,
        normalizer=MatrixNormalizer("standardize"),
        **kwargs,
    ):
        self.device = device
        self.scale = scale
        self.momentum = momentum
        self.temperature = temperature

        self.objects, self.frame_index = self._parse_objects(
            file_objects,
            file_z,
            file_object_ids,
            device,
            **kwargs,
        )
        self.tracks: list[HT_Track] = []
        self.v = None
        self._normalizer = normalizer

    @property
    def track_id_num(self) -> int:
        return len(self.tracks)

    def _parse_objects(
        self,
        file_objects: str | os.PathLike,
        file_z: str | os.PathLike,
        file_object_ids: str | os.PathLike | None,
        device,
        **kwargs,
    ) -> tuple[list[HT_Object], list[tuple[int, int]]]:
        """
        Parse object and volume embedding files to create HT_Object instances.

        Returns
        -------
        objects : list of HT_Object
            Loaded objects with their feature vectors on the specified device.
        frame_index : list of (int, int)
            Start and end indices of objects per frame.
        """
        logging.debug(f"Loading objects from {file_objects}")
        df_objects = pd.read_csv(
            file_objects,
            dtype={
                "object_id": "str",
                "t": "int",
                "x": "float",
                "y": "float",
                "z": "float",
            },
        )
        df_objects.sort_values("t", inplace=True, ignore_index=True)

        logging.debug(f"Loaded {df_objects.index.size} objects.")

        # Load volume embeddings if used.
        logging.debug(f"Loading volume embeddings from {file_z}")

        file_z = Path(file_z)
        if file_z.suffix.endswith((".pt", ".pth")):
            Z = torch.load(file_z, map_location=device, weights_only=True)
        elif file_z.suffix.endswith((".np", ".npy")):
            arr = np.load(file_z)
            Z = torch.tensor(arr, dtype=torch.float32, device=device)
        else:
            raise ValueError(
                "Unknown volume embedding file format. Supported: .pt/.pth, .npy"
            )

        logging.debug(f"Loaded {Z.shape[0]} volume embeddings.")
        if Z.shape[0] != df_objects.index.size:
            raise ValueError(
                "Number of volume embedding vectors does not match number of objects."
            )

        if file_object_ids is None:
            # Assuming the index is same as z_idx, if file_object_ids is not provided.
            df_objects["z_idx"] = df_objects.index.to_series(name="z_idx")
        else:
            logging.debug(
                f"Loading object IDs for volume embeddings from {file_object_ids}"
            )
            assert Path(file_object_ids).name.endswith((".pth", ".pt"))

            pt_object_ids = torch.load(
                file_object_ids, map_location="cpu", weights_only=True
            )
            # cast object IDs to string
            pt_object_ids = [str(obj_id) for obj_id in pt_object_ids]
            logging.debug(f"Loaded {len(pt_object_ids)} object IDs.")
            # map object id to index in Z
            map_obj_id_to_z_idx = {obj_id: i for i, obj_id in enumerate(pt_object_ids)}
            df_objects["z_idx"] = df_objects["object_id"].map(map_obj_id_to_z_idx)

        # Create HT_Object instances.
        # Ensure the column order, so that we can unpack tuple in order.
        df_objects = df_objects[["object_id", "t", "x", "y", "z", "z_idx"]]
        objects = [
            HT_Object(object_id, t, (z, y, x), Z[z_idx])
            for object_id, t, x, y, z, z_idx in df_objects.itertuples(
                index=False,
                name=None,
            )
        ]

        frame_index = (
            df_objects.index.to_series().groupby(df_objects["t"]).agg(["first", "last"])
        )
        frame_index["last"] += 1
        frame_index = list(frame_index.itertuples(index=False, name=None))

        return objects, frame_index

    def estimate_max_distance(self, arr_distance: torch.Tensor, weight_within=0.5):
        """
        Estimate the max_distance threshold by splitting distances into within-track and inter-track subsets.
        """

        # Best fit from active_track to new_objects
        arr_dist_within, _ = torch.min(arr_distance, dim=1)
        mean_within = torch.mean(arr_dist_within)
        mean_inter = (arr_distance.sum() - arr_dist_within.sum()) / (
            arr_distance.numel() - arr_dist_within.numel()
        )
        max_distance = (
            weight_within * mean_within.item() + (1 - weight_within) * mean_inter.item()
        )

        return max_distance

    @torch.no_grad()
    def update_one_frame(
        self,
        new_objects: list[HT_Object],
        weight_within: float,
        max_gap_frames: int,
    ):
        """
        Update the tracker for one frame using the Hungarian algorithm.
        This method computes cost matrices from volume embeddings (using cosine similarity)
        """
        if len(self.tracks) == 0:
            self.tracks = [
                HT_Track(str(i), obj, momentum=self.momentum)
                for i, obj in enumerate(new_objects)
            ]
            return

        active_tracks = []
        inactive_tracks = []

        for track in self.tracks:
            if (
                track.objects[-1].t + max_gap_frames + 1 >= new_objects[0].t
                and track.v is not None
            ):
                active_tracks.append(track)
            else:
                inactive_tracks.append(track)

        if len(active_tracks) == 0:
            raise ValueError("No active tracks with volume embeddings found. ")

        v_tracks_cat = torch.concatenate([track.v for track in active_tracks]).to(
            self.device
        )  # (len(v_tracks), feat)
        v_objects = torch.stack([obj.z for obj in new_objects if obj.z is not None]).to(
            self.device
        )  # (len(new_objects), feat)

        # calculate Cosine Similarity and eventual normalized similarity
        # (old_objs, 1, feat)
        # (1, new_objs, feat)
        pred_CS = (
            F.cosine_similarity(
                v_tracks_cat.unsqueeze(1),
                v_objects.unsqueeze(0),
                dim=2,
            )
            / self.temperature
        )
        # apply row-wise softmax and column-wise softmax and average the two
        pred_CS_sm_r = F.softmax(pred_CS, dim=0)
        pred_CS_sm_c = F.softmax(pred_CS, dim=1)
        pred_sim = (pred_CS_sm_r + pred_CS_sm_c) / 2

        dist_matrix = -pred_sim
        # dist_matrix = -1 * F.cosine_similarity(v_tracks_cat, v_objects, dim=2)
        # normalize
        # (n_active, n_new_object)
        cost_matrix = self._normalizer(dist_matrix)

        max_distance = self.estimate_max_distance(
            cost_matrix,
            weight_within,
        )
        cost_matrix_np = cost_matrix.cpu().numpy()
        print(cost_matrix_np.shape)
        # Hungarian/Kuhn-Munkres algorithm here. Actually, it used Jonker-Volgenant
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix_np)

        assigned_tracks = set()
        assigned_objects = set()
        dismissed_objects = set(range(len(new_objects)))
        for i, j in zip(row_ind, col_ind):
            dist = cost_matrix_np[i, j]
            if dist >= max_distance:
                continue
            active_tracks[i].append(new_objects[j])
            assigned_tracks.add(i)
            assigned_objects.add(j)
            dismissed_objects.remove(j)

        for j in sorted(dismissed_objects):
            # we increment the id by count
            track_id = str(len(inactive_tracks) + len(active_tracks) + 1)
            active_tracks.append(
                HT_Track(
                    track_id,
                    new_objects[j],
                    momentum=self.momentum,
                )
            )
        self.tracks = active_tracks + inactive_tracks

    def solve_dynamic_cutoff(
        self,
        cutoff_weight_within: float = 0.5,
        max_gap_frames: int = 1,
    ) -> list[HT_Track]:
        """
        Solve the entire tracking problem sequentially using dynamic cutoff estimation.
        This method dynamically updates max_distance for each frame based on the distances
        between objects in the current frame and the previous frame.
        """
        tik_all = datetime.datetime.now()
        print(
            f"Using weight_within: {cutoff_weight_within:.2f}, and max_gap_frames: {max_gap_frames:d} to solve the tracking"
        )
        for start, end in self.frame_index:
            new_objects = self.objects[start:end]
            self.update_one_frame(new_objects, cutoff_weight_within, max_gap_frames)

        tok_all = datetime.datetime.now()
        total_seconds = (tok_all - tik_all).total_seconds()
        print(
            f"Total tracking time: {total_seconds}s for {len(self.frame_index)} frames. ({total_seconds / len(self.frame_index):.4f}s per frame)."
        )
        return self.tracks
