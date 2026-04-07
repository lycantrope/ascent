import logging
import os
from collections import defaultdict
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import center_of_mass
from scipy.spatial import distance


class Spot:
    def __init__(self, spot_id: str, t: int, coord: Sequence[float] = ()):
        self.id = spot_id
        self.t = t
        self.coord = coord
        self.next: Spot | None = None
        self.prev: Spot | None = None

    def __repr__(self) -> str:
        return f"Spot({self.id}, t={self.t})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


# Head for iteration
class TrackIterator:
    def __init__(self, head: Spot | None):
        self.current = head

    def __iter__(self):
        return self

    def __next__(self) -> Spot:
        if self.current is None:
            raise StopIteration
        else:
            item = self.current
            self.current = self.current.next
            return item


class Track:
    def __init__(self, track_id: str):
        self.id: str = track_id
        self.head: Spot | None = None
        self.tail: Spot | None = None

    def __iter__(self) -> Iterator[Spot]:
        return TrackIterator(self.head)

    def add(self, spot: Spot):
        if self.head is None and self.tail is None:
            self.head = spot
            self.tail = spot
        elif self.tail is not None:
            # assert self.tail.t < spot.t
            if self.tail.t == spot.t:
                logging.warning(
                    f"Adding spot {spot} to track {self} with the same frame as the tail {self.tail}."
                )
            spot.prev = self.tail
            self.tail.next = spot
            self.tail = spot

    def pop(self):
        """Remove and return the last spot from the track."""
        assert (
            self.tail is not None
            and self.tail.prev is not None
            and self.tail.next is None
        )
        if self.tail == self.head:
            spot = self.tail
            self.head = None
            self.tail = None
        else:
            spot = self.tail
            self.tail = self.tail.prev
            self.tail.next = None

        spot.prev = None
        return spot

    def t_set(self):
        return set((spot.t for spot in self))

    def __repr__(self):
        return f"Track({self.id})"

    def __len__(self):
        return sum((1 for _ in self))


def track_to_dict(tracks: dict[str, Track]) -> dict[int, dict[str, str]]:
    """Converts a dictionary of Track objects to a dictionary of tracks by frame and spot_id."""
    track_dict = {}
    for track_id, track in tracks.items():
        for spot in track:
            if spot.t not in track_dict:
                track_dict[spot.t] = {}
            track_dict[spot.t][spot.id] = track_id
    return track_dict


def _count_tracking_errors_from_dicts(
    gt_dict: dict[int, dict[str, str]], pred_dict: dict[int, dict[str, str]]
) -> dict:
    frame_errors = {}  # Store errors by frame for detailed analysis

    # Gather all frames from both ground truth and predictions
    all_frames = set(gt_dict.keys()).union(pred_dict.keys())
    sorted_frames = sorted(list(all_frames))

    # Track matching dictionary
    track_matches = {}  # Maps predicted track IDs to ground truth track IDs
    matched_gt_tracks = set()  # Set of ground truth tracks that have been matched

    error_by_track_gt = defaultdict(lambda: defaultdict(int))
    error_by_track_pred = defaultdict(lambda: defaultdict(int))

    for frame in sorted_frames:
        gt_frame_tracks = gt_dict.get(frame, {})
        pred_frame_tracks = pred_dict.get(frame, {})

        # Initialize track_matches in the first frame
        if frame == sorted_frames[0]:
            for p_spot_id, p_track_id in pred_frame_tracks.items():
                gt_track_id = gt_frame_tracks.get(p_spot_id, None)
                if gt_track_id is None:
                    # TODO: This could be a false positive. Currently we don't count false positives.
                    logging.warning(
                        f"False positive: Spot {p_spot_id} in pred track {p_track_id} not in GT."
                    )
                else:
                    track_matches[p_track_id] = gt_track_id
                    matched_gt_tracks.add(gt_track_id)

            missing = []
            for spot_id, gt_track_id in gt_frame_tracks.items():
                if spot_id not in pred_frame_tracks:
                    error_by_track_gt[gt_track_id]["missing"] += 1
                    missing.append(spot_id)

            frame_errors[frame] = {"mismatch": [], "missing": missing}
            continue

        # For subsequent frames, check track continuity
        mismatch = []
        current_matches = {}
        for p_spot_id, p_track_id in pred_frame_tracks.items():
            gt_track_id = gt_frame_tracks.get(p_spot_id, None)
            if gt_track_id is None:
                # TODO: This could be a false positive. Currently we don't count false positives.
                logging.warning(
                    f"False positive: Spot {p_spot_id} in pred track {p_track_id} not in GT."
                )
            else:
                tracked_gt_track_id = track_matches.get(p_track_id)

                # Check if the gt_track is matched in a previous frame
                if gt_track_id in matched_gt_tracks:
                    # GT track has already been matched in a previous frame
                    # Check if the previous and current ground truth track IDs match
                    if tracked_gt_track_id != gt_track_id:
                        mismatch.append(
                            {
                                "Spot": p_spot_id,
                                "PredTrack": p_track_id,
                                "TrueGTTrack": gt_track_id,
                                "TrackedGTTrack": tracked_gt_track_id,
                            }
                        )
                        error_by_track_gt[gt_track_id]["mismatch"] += 1
                        error_by_track_pred[p_track_id]["mismatch"] += 1
                else:
                    # This is a new GT track that has not been matched yet
                    if tracked_gt_track_id is not None:
                        # This is a mismatch
                        # Predicted track is supposed to start at this frame, meaning tracked_gt_track_id should be None
                        mismatch.append(
                            {
                                "Spot": p_spot_id,
                                "PredTrack": p_track_id,
                                "TrueGTTrack": gt_track_id,
                                "TrackedGTTrack": tracked_gt_track_id,
                            }
                        )
                        error_by_track_gt[gt_track_id]["mismatch"] += 1
                        error_by_track_pred[p_track_id]["mismatch"] += 1

                current_matches[p_track_id] = gt_track_id
                matched_gt_tracks.add(gt_track_id)

        # Count false negatives
        missing = []
        for spot_id, gt_track_id in gt_frame_tracks.items():
            if spot_id not in pred_frame_tracks:
                error_by_track_gt[gt_track_id]["missing"] += 1
                missing.append(spot_id)

        # Update tracking history
        track_matches.update(current_matches)

        # Record frame-specific errors
        frame_errors[frame] = {
            "mismatch": mismatch,
            "missing": missing,
            "objects": len(gt_frame_tracks),
            "mismatch_ratio": len(mismatch) / len(gt_frame_tracks),
            "missing_ratio": len(missing) / len(gt_frame_tracks),
            "MOTA": 1 - (len(mismatch) + len(missing)) / len(gt_frame_tracks),
        }

    total_mismatch = sum(len(frame_errors[fr]["mismatch"]) for fr in frame_errors)
    total_missing = sum(len(frame_errors[fr]["missing"]) for fr in frame_errors)
    total_objects = sum(len(gt_dict[frame]) for frame in gt_dict)
    for errors in error_by_track_gt.values():
        errors["total"] = errors["mismatch"] + errors["missing"]
    for errors in error_by_track_pred.values():
        errors["total"] = errors["mismatch"]

    return {
        "total_mismatch": total_mismatch,
        "total_missing": total_missing,
        "total_objects": total_objects,
        "mismatch_ratio": total_mismatch / total_objects,
        "missing_ratio": total_missing / total_objects,
        "MOTA": 1 - (total_mismatch + total_missing) / total_objects,
        "details_by_frame": frame_errors,
        "details_by_track_gt": error_by_track_gt,
        "details_by_track_pred": error_by_track_pred,
    }


def count_tracking_errors(
    gt_tracks: dict[str, Track], pred_tracks: dict[str, Track]
) -> dict:
    """
    Counts the tracking errors between ground truth tracks and predicted tracks.

    Args:
        gt_tracks (dict of Track): Ground truth tracks.
        pred_tracks (dict of Track): Predicted tracks.

    Returns:
        dict: Dictionary containing counts of different types of errors.
    """
    gt_dict = track_to_dict(gt_tracks)
    pred_dict = track_to_dict(pred_tracks)

    return _count_tracking_errors_from_dicts(gt_dict, pred_dict)


def map_gt_to_pred_spots(
    gt_spots, pred_spots, dist_threshold, scale=(1.5, 0.3226, 0.3226)
):
    for t in gt_spots["t"].unique():
        df_c_g_t = gt_spots[gt_spots["t"] == t]
        df_c_p_t = pred_spots[pred_spots["t"] == t]
        coord_g = df_c_g_t[["z", "y", "x"]].values * scale
        coord_p = df_c_p_t[["z", "y", "x"]].values * scale
        dist = distance.cdist(coord_g, coord_p)
        min_dist_g_to_p = np.min(dist, axis=1)
        min_dist_g_to_p_idx = np.argmin(dist, axis=1)
        min_dist_g_to_p_oid_p = df_c_p_t.index[min_dist_g_to_p_idx]
        gt_spots.loc[df_c_g_t.index, "min_dist"] = min_dist_g_to_p
        gt_spots.loc[df_c_g_t.index, "min_dist_oid_p"] = min_dist_g_to_p_oid_p

        min_dist_p_to_g = np.min(dist, axis=0)
        mis_dist_p_to_g_idx = np.argmin(dist, axis=0)
        min_dist_p_to_g_oid_g = df_c_g_t.index[mis_dist_p_to_g_idx]
        pred_spots.loc[df_c_p_t.index, "min_dist"] = min_dist_p_to_g
        pred_spots.loc[df_c_p_t.index, "min_dist_oid_g"] = min_dist_p_to_g_oid_g

        # get mutual nearest neighbors
        mutual_nn = np.zeros_like(dist, dtype=bool)
        for i, j in enumerate(min_dist_g_to_p_idx):
            if mis_dist_p_to_g_idx[j] == i:
                mutual_nn[i, j] = True

        gt_spots.loc[df_c_g_t.index, "mutual_nn"] = mutual_nn.any(axis=1)
        pred_spots.loc[df_c_p_t.index, "mutual_nn"] = mutual_nn.any(axis=0)

    df_c_g_valid = gt_spots[
        (gt_spots["min_dist"] <= dist_threshold) & gt_spots["mutual_nn"]
    ]
    oid_gt = df_c_g_valid.index
    oid_pred = df_c_g_valid["min_dist_oid_p"]
    map_gt_to_pred = dict(zip(oid_gt, oid_pred))

    return map_gt_to_pred


def map_gt_to_pred_spots_from_label(
    gt_spots,
    pred_spots,
    file_label,
    dist_threshold,
    scale,
    file_out_gt=None,
    file_out_pred=None,
):
    if file_out_gt is not None and os.path.exists(file_out_gt):
        gt_spots = pd.read_csv(file_out_gt, index_col=0)
        gt_spots["mapping"] = gt_spots["mapping"].astype(int)
    else:
        with h5py.File(file_label, "r") as f:
            oid_offset = f["oid_offset"][()]
            scale = np.array(scale)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # optimal mapping to each other
            # -1 means there is no mapping
            gt_spots.loc[:, "mapping"] = -1
            pred_spots.loc[:, "mapping"] = -1
            for t in gt_spots["t"].unique():
                # read the instance segmentation result
                label_t = np.array(f[str(t)])
                # get the center of mass of each instance
                cm_t = center_of_mass(
                    label_t, label_t, range(1, int(label_t.max()) + 1)
                )

                # get the spots for time t
                df_c_g_t = gt_spots[gt_spots["t"] == t]
                df_c_p_t = pred_spots[pred_spots["t"] == t]

                # Mapping from gt to pred; First, find the label of each gt spot.
                # The label + oid_offset is the object id in the pred spots.
                coord_gt = np.round(df_c_g_t[["z", "y", "x"]].values).astype(int)
                label_gt_spots = label_t[
                    coord_gt[:, 0], coord_gt[:, 1], coord_gt[:, 2]
                ].astype(int)
                label_gt_spots[label_gt_spots == 0] = -1
                label_gt_spots[label_gt_spots > 0] += oid_offset[t] - 1
                gt_spots.loc[df_c_g_t.index, "mapping"] = label_gt_spots.astype(int)

                # Next, for the ones that have the same label, select the one with the smallest distance to the centroid of that label.
                lbl_cnt = gt_spots.loc[df_c_g_t.index, "mapping"].value_counts()
                lbl_cnt.index = lbl_cnt.index.astype(int)
                lbl_cnt = lbl_cnt[lbl_cnt > 1]
                for lbl in lbl_cnt.index:
                    if lbl < 0:
                        continue
                    df_c_g_lbl = df_c_g_t[
                        gt_spots.loc[df_c_g_t.index, "mapping"] == lbl
                    ]
                    # bug fix: lbl - oid_offset[t] - 1 => lbl - oid_offset[t]
                    cm_lbl = cm_t[lbl - oid_offset[t]]
                    dist = distance.cdist(
                        df_c_g_lbl[["z", "y", "x"]].values * scale, [cm_lbl * scale]
                    )
                    min_dist_idx = df_c_g_lbl.index[np.argmin(dist)]
                    df_c_g_free = df_c_g_lbl.drop(min_dist_idx)
                    gt_spots.loc[df_c_g_free.index, "mapping"] = -1

                # Fill the mapping column in the pred spots
                for idx in df_c_g_t.index:
                    gt_mapping = gt_spots.loc[idx, "mapping"]
                    if gt_mapping >= 0:
                        pred_spots.loc[gt_mapping, "mapping"] = idx

                # Next, for the ones that did not match with any pred spots, find the closest pred spot which is not matched with any gt spots.
                df_c_g_t_left = gt_spots[
                    (gt_spots["t"] == t) & (gt_spots["mapping"] < 0)
                ]
                df_c_p_t_left = pred_spots[
                    (pred_spots["t"] == t) & (pred_spots["mapping"] < 0)
                ]
                coord_g = torch.from_numpy(
                    df_c_g_t_left[["z", "y", "x"]].values * scale
                ).to(device)
                coord_p = torch.from_numpy(df_c_p_t[["z", "y", "x"]].values * scale).to(
                    device
                )
                # calculate the distance between each gt spot and each pred spot
                dist = torch.cdist(coord_g, coord_p, p=2)
                dist_order_g_to_p = torch.argsort(dist, dim=1)
                temp_mapping = {}
                for i in range(dist_order_g_to_p.shape[0]):
                    j = dist_order_g_to_p[i, 0].item()
                    # if the pred spot is not matched with any gt spots
                    if df_c_p_t.index[j] in df_c_p_t_left.index:
                        # if the pred spot is the closest to the gt spot (mutual nearest neighbor)
                        if (
                            j not in temp_mapping or dist[i, j] < temp_mapping[j][1]
                        ) and dist[i, j] < dist_threshold:
                            temp_mapping[j] = (i, dist[i, j])
                # assign the mapping
                for j, (i, d) in temp_mapping.items():
                    gt_spots.loc[df_c_g_t_left.index[i], "mapping"] = df_c_p_t.index[j]
                    pred_spots.loc[df_c_p_t.index[j], "mapping"] = df_c_g_t_left.index[
                        i
                    ]

    # get the mapping
    df_c_g_valid = gt_spots[gt_spots["mapping"] >= 0]
    oid_gt = df_c_g_valid.index.astype(str)
    oid_pred = df_c_g_valid["mapping"].values.astype(str)
    map_gt_to_pred = dict(zip(oid_gt, oid_pred))

    if file_out_gt is not None and not os.path.exists(file_out_gt):
        gt_spots.to_csv(file_out_gt)
    if file_out_pred is not None and not os.path.exists(file_out_pred):
        pred_spots.to_csv(file_out_pred)

    return map_gt_to_pred


def count_tracking_errors_different_detections(
    gt_tracks: dict[str, Track],
    pred_tracks: dict[str, Track],
    gt_to_pred_correspondence: dict[str, str],
) -> dict:
    """
    Counts the tracking errors between ground truth tracks and predicted tracks,
    given that the prediction may not share the same detection IDs as the GT.

    We ignore predicted detections that do not appear in 'gt_to_pred_correspondence',
    effectively treating them as 'unlabeled positives' that do not incur an error.

    Args:
        gt_tracks (dict[str, Track]): Ground truth tracks, with spot IDs matching GT-labeled spots.
        pred_tracks (dict[str, Track]): Predicted tracks, with spot IDs from a (potentially) different detection set.
        gt_to_pred_correspondence (dict[str, str]): Mapping from GT spot_id -> pred spot_id.

    Returns:
        dict: Dictionary containing counts of different types of errors, e.g. mismatch and missing.
              {
                  "total_mismatch": ...,
                  "total_missing": ...,
                  "details_by_frame": ...
              }
    """

    # Step 1: rename predicted detection IDs to GT detection IDs where possible
    # We do that by inverting the 'gt_to_pred_correspondence' so we can quickly
    # find which GT ID each pred ID might correspond to.
    # Actually the dict is GT->pred, so we want pred->gt
    pred_to_gt = {}
    for gt_id, pred_id in gt_to_pred_correspondence.items():
        pred_to_gt[pred_id] = gt_id

    # Step 2: Convert GT and Pred tracks to frame-wise dictionaries
    gt_dict = track_to_dict(gt_tracks)  # frame -> {gt_spot_id: gt_track_id}
    # We'll create a "remapped" dict for pred, ignoring unknown spots
    # which do not appear in pred_to_gt
    pred_dict_remapped = {}

    # Turn predicted tracks into (frame -> {pred_spot_id : pred_track_id})
    raw_pred_dict = track_to_dict(pred_tracks)

    # For each pred_spot_id in raw_pred_dict, see if it belongs to pred_to_gt
    # If not, ignore. If it does, rename it to the GT ID, so we can treat it as if it were the same ID as GT
    for frame, frame_dict in raw_pred_dict.items():
        remapped_frame_dict = {}
        for pred_spot_id, pred_track_id in frame_dict.items():
            # see if pred_spot_id is in pred_to_gt
            if pred_spot_id in pred_to_gt:
                # rename the spot to the GT spot id
                gt_spot_id = pred_to_gt[pred_spot_id]
                # for the track ID, we do the same approach as "count_tracking_errors":
                # We want "pred_track_id" but keep it or maybe we keep it.
                # We'll keep it as is, but the "spot ID" is replaced by gt_spot_id
                remapped_frame_dict[gt_spot_id] = pred_track_id
            else:
                # This predicted detection is not in the GT set => ignore
                pass
        if remapped_frame_dict:
            pred_dict_remapped[frame] = remapped_frame_dict

    # Step 3: Reuse the logic from 'count_tracking_errors' by feeding these dictionaries
    #   - 'gt_dict' remains the same
    #   - 'pred_dict' is replaced by 'pred_dict_remapped'
    # We'll replicate the logic from count_tracking_errors
    return _count_tracking_errors_from_dicts(gt_dict, pred_dict_remapped)


def find_correspondence(gt_tracks: dict[str, Track], pred_tracks: dict[str, Track]):
    """
    Find correspondence between GT tracks and predicted tracks based on shared spots.

    Args:
        gt_tracks (dict): Dictionary of ground truth Track objects with track_id as keys.
        pred_tracks (dict): Dictionary of predicted Track objects with target_id as keys.

    Returns:
        correspondence (dict): Dictionary mapping GT track_id to predicted target_id.
    """
    correspondence = {}
    for gt_id, gt_track in gt_tracks.items():
        max_shared_spots = 0
        best_match = None
        for pred_id, pred_track in pred_tracks.items():
            gt_spotids = [spot.id for spot in gt_track]
            pred_spotids = [spot.id for spot in pred_track]
            shared_spots = len(set(gt_spotids) & set(pred_spotids))
            if shared_spots > max_shared_spots:
                max_shared_spots = shared_spots
                best_match = pred_id
        if best_match is not None:
            correspondence[gt_id] = best_match
    return correspondence
