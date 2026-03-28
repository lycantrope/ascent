import datetime
import logging
import os

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

    def __init__(
        self,
        id: str,
        t: int,
        coord: tuple[float, float, float],
        z: torch.Tensor,
    ):
        Spot.__init__(self, id, t, coord)
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

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other: Spot):
        return self.id == other.id


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

    def __init__(self, id: str, object: HT_Object, momentum: float = 0.5, update_vectors=True):
        Track.__init__(self, id)
        self.objects = []
        self.v = None  # representative volume embedding vector(s)
        self.momentum = momentum
        self.update_vectors = update_vectors

        self.append(object)

    def append(self, obj: HT_Object):
        """Append a new object to the track and update the representative vectors."""
        self.objects.append(obj)
        self.add(obj)

        # Update volume embedding representation if available.
        if self.update_vectors:
            if obj.z is not None:
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
    device : str, optional
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
        file_objects: str,
        file_z: str | None,
        file_object_ids_z: str | None = None,
        device="cpu",
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        momentum: float = 0.5,
        temperature: float = 0.05,
        **kwargs,
    ):
        self.device = device
        self.scale = scale
        self.momentum = momentum
        self.temperature = temperature

        self.objects, self.frame_index = self._parse_objects(
            file_objects,
            file_z,
            file_object_ids_z,
            device,
            **kwargs,
        )
        self.tracks: list[HT_Track] = []
        self.track_id_num = 0
        self.v = None
        self.dist_norm_method = "standardize"

    def set_dist_norm_method(self, dist_norm_method: str):
        """Set the distance normalization method for the combined cost matrix."""
        self.dist_norm_method = dist_norm_method

    def _normalize_cost_matrix(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        if self.dist_norm_method == "standardize":
            return (cost_matrix - torch.mean(cost_matrix)) / torch.std(cost_matrix)
        elif self.dist_norm_method == "minmax":
            return (cost_matrix - torch.min(cost_matrix)) / (
                torch.max(cost_matrix) - torch.min(cost_matrix)
            )
        elif self.dist_norm_method == "distribution":
            n_within = cost_matrix.shape[1]
            cost_matrix_flat = cost_matrix.flatten()
            cost_matrix_flat = cost_matrix_flat.sort()[0]
            mean_within = torch.mean(cost_matrix_flat[:n_within])
            mean_inter = torch.mean(cost_matrix_flat[n_within:])
            # make mean_within = 0, mean_inter = 1
            return (cost_matrix - mean_within) / (mean_inter - mean_within)
        elif self.dist_norm_method == "none":
            return cost_matrix
        else:
            raise ValueError(f"Unknown dist_norm_method: {self.dist_norm_method}")

    def _parse_objects(
        self,
        file_objects: str,
        file_z: str | None,
        file_object_ids_z: str | None,
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
            dtype={"object_id": "str", "t": "int", "x": "float", "y": "float", "z": "float"},
        )
        df_objects.sort_values("t", inplace=True, ignore_index=True)
        object_ids = df_objects["object_id"].values
        logging.debug(f"Loaded {len(df_objects)} objects.")

        # Load volume embeddings if used.
        logging.debug(f"Loading volume embeddings from {file_z}")
        ext = os.path.splitext(file_z)[1]
        if ext in [".pt", ".pth"]:
            Z = torch.load(file_z, map_location=device, weights_only=True)
        elif ext in [".np", ".npy"]:
            arr = np.load(file_z)
            Z = torch.tensor(arr, dtype=torch.float32, device=device)
        else:
            raise ValueError("Unknown volume embedding file format. Supported: .pt/.pth, .npy")
        logging.debug(f"Loaded {Z.shape[0]} volume embeddings.")
        if Z.shape[0] != len(df_objects):
            raise ValueError("Number of volume embedding vectors does not match number of objects.")

        if file_object_ids_z is not None:
            logging.debug(f"Loading object IDs for volume embeddings from {file_object_ids_z}")
            pt_object_ids = torch.load(file_object_ids_z, map_location="cpu", weights_only=True)
            # cast object IDs to string
            pt_object_ids = [str(obj_id) for obj_id in pt_object_ids]
            logging.debug(f"Loaded {len(pt_object_ids)} object IDs.")
            # map object id to index in Z
            map_obj_id_to_z_idx = {obj_id: i for i, obj_id in enumerate(pt_object_ids)}
        else:
            map_obj_id_to_z_idx = None

        # Create HT_Object instances.
        objects = []
        for i, row in df_objects.iterrows():
            obj_id = row["object_id"]
            t = row["t"]
            coords = (row["z"], row["y"], row["x"])  # (z, y, x)
            z_idx = map_obj_id_to_z_idx[obj_id] if map_obj_id_to_z_idx is not None else i
            z_vec = Z[z_idx]
            obj = HT_Object(obj_id, t, coords, z_vec)
            objects.append(obj)

        frame_index = []
        for t_val in df_objects["t"].unique():
            subset = df_objects[df_objects["t"] == t_val]
            start_idx = subset.index[0]
            end_idx = subset.index[-1] + 1
            frame_index.append((start_idx, end_idx))

        return objects, frame_index

    def estimate_max_distance(self, arr_distance, n_within, weight_within=0.5):
        """
        Estimate the max_distance threshold by splitting distances into within-track and inter-track subsets.
        (Unchanged from your original code.)
        """
        arr_distance = arr_distance.sort()[0]
        arr_dist_within = arr_distance[:n_within]
        arr_dist_inter = arr_distance[n_within:]

        mean_within = torch.mean(arr_dist_within).item()
        mean_inter = torch.mean(arr_dist_inter).item()

        max_distance = weight_within * mean_within + (1 - weight_within) * mean_inter

        return max_distance

    def update_one_frame(
        self,
        new_objects: list[HT_Object],
    ):
        """
        Update the tracker for one frame using the Hungarian algorithm.
        This method computes cost matrices from volume embeddings (using cosine similarity)
        """
        if len(self.tracks) == 0:
            for new_object in new_objects:
                self.tracks.append(
                    HT_Track(str(self.track_id_num), new_object, momentum=self.momentum)
                )
                self.track_id_num += 1
        else:
            active_tracks = [
                track
                for track in self.tracks
                if track.objects[-1].t >= new_objects[0].t - self.max_gap_frames - 1
            ]
            inactive_tracks = [
                track
                for track in self.tracks
                if track.objects[-1].t < new_objects[0].t - self.max_gap_frames - 1
            ]

            cost_matrix = None
            v_tracks = [track.v for track in active_tracks if track.v is not None]
            if len(v_tracks) > 0:
                v_tracks_cat = torch.cat(v_tracks).unsqueeze(1)  # (len(v_tracks), 1, feat)
                v_objects = torch.stack(
                    [obj.z for obj in new_objects if obj.z is not None]
                ).unsqueeze(0)

                # calculate Cosine Similarity and eventual normalized similarity
                pred_CS = F.cosine_similarity(v_tracks_cat, v_objects, dim=2) / self.temperature
                # apply row-wise softmax and column-wise softmax and average the two
                pred_CS_sm_r = F.softmax(pred_CS, dim=0)
                pred_CS_sm_c = F.softmax(pred_CS, dim=1)
                pred_sim = (pred_CS_sm_r + pred_CS_sm_c) / 2

                dist_matrix = -pred_sim
                # dist_matrix = -1 * F.cosine_similarity(v_tracks_cat, v_objects, dim=2)
                # normalize
                cost_matrix = self._normalize_cost_matrix(dist_matrix)
            else:
                raise ValueError("No active tracks with volume embeddings found. ")

            max_distance = self.estimate_max_distance(
                cost_matrix.flatten(),
                len(active_tracks),
                self.cutoff_weight_within,
            )
            cost_matrix_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix_np)

            assigned_tracks = set()
            assigned_objects = set()

            for i, j in zip(row_ind, col_ind):
                dist = cost_matrix_np[i, j]
                if dist < max_distance:
                    active_tracks[i].append(new_objects[j])
                    assigned_tracks.add(i)
                    assigned_objects.add(j)

            for j, obj in enumerate(new_objects):
                if j not in assigned_objects:
                    active_tracks.append(
                        HT_Track(str(self.track_id_num), obj, momentum=self.momentum)
                    )
                    self.track_id_num += 1

            self.tracks = active_tracks + inactive_tracks

    def solve_dynamic_cutoff(
        self,
        cutoff_weight_within: float = 0.5,
        max_gap_frames: int = 1,
    ):
        """
        Solve the entire tracking problem sequentially using dynamic cutoff estimation.
        This method dynamically updates max_distance for each frame based on the distances
        between objects in the current frame and the previous frame.
        """
        self.cutoff_weight_within = cutoff_weight_within
        self.max_gap_frames = max_gap_frames

        tik_all = datetime.datetime.now()
        for t, (start, end) in enumerate(self.frame_index):
            new_objects = self.objects[start:end]
            self.update_one_frame(new_objects)

        tok_all = datetime.datetime.now()
        total_seconds = (tok_all - tik_all).total_seconds()
        print(
            f"Total tracking time: {total_seconds}s for {len(self.frame_index)} frames. ({total_seconds / len(self.frame_index):.4f}s per frame)."
        )

    def save_tracks_napari(self, file_output_tracks):
        """
        Save the resulting tracks to a Napari-compatible track CSV file.
        """
        with open(file_output_tracks, "w") as f:
            f.write("TrackID,ObjectID,t,z,y,x\n")
            for track in self.tracks:
                for obj in track.objects:
                    f.write(
                        f"{track.id},{obj.id},{obj.t},{obj.coord[0]},{obj.coord[1]},{obj.coord[2]}\n"
                    )

if __name__ == "__main__":
    pass
