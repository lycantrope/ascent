import logging
import os

import pandas as pd
from ascent.utils.track.common import Spot, Track


def read_spots_trackmate(
    path_spots: str, path_tracks: str = None, filter_spots_in_track=True, **kwargs
) -> pd.DataFrame:
    """
    Read the spots CSV file and return a pandas DataFrame with the spots.

    Args:
        path_spots: Path to the spots CSV file
        path_tracks: Path to the tracks CSV file. Only necessary if filter_spots_in_track is True
        filter_spots_in_track: Boolean to determine whether to filter the spots in the tracks
        **kwargs: Additional arguments to pass to the read_csv function

    Returns:
        Pandas DataFrame with the spots
    """
    df_s = pd.read_csv(path_spots, index_col=0, **kwargs)
    df_s["TRACK_LABEL"] = None

    if path_tracks is not None and filter_spots_in_track:
        df_t = pd.read_csv(path_tracks, index_col=2, **kwargs)
        df_s = df_s[df_s.TRACK_ID.isin(df_t.index)]
        df_s["TRACK_LABEL"] = df_s.TRACK_ID.map(df_t.LABEL)

    # Sort the spots by frame
    df_s = df_s.sort_values(by=["FRAME"])

    return df_s


def read_tracks_trackmate(
    path_spots: str, path_tracks: str | None = None, skiprows: list[int] | None = None
) -> dict[str, Track]:
    """
    Read the spots and tracks CSV files and return a dictionary of tracks with
    the spots in each track.

    Args:
        path_spots: Path to the spots CSV file
        path_tracks: Path to the tracks CSV file (optional)
        skiprows: List of row numbers to skip in the CSV file (optional)

    Returns:
        Dictionary of tracks with the spots in each track
    """
    logging.debug("Reading spots and tracks CSV files: %s" % path_spots)
    assert os.path.exists(path_spots), f"File not found: {path_spots}"

    tracks = {}
    if path_tracks is None:
        df_s = pd.read_csv(path_spots, index_col=0, skiprows=skiprows, dtype={"TRACK_ID": int})
        df_s.sort_values("TRACK_ID", inplace=True)
        frames = list(df_s["FRAME"].unique())
        frames.sort()

        for frame in frames:
            df_s_f = df_s[df_s["FRAME"] == frame]
            for spot_id, row in df_s_f.iterrows():
                track_id = str(row["TRACK_ID"])
                if track_id not in tracks:
                    tracks[track_id] = Track(track_id)
                spot = Spot(
                    str(spot_id),
                    int(frame),
                    (row["POSITION_Z"], row["POSITION_Y"], row["POSITION_X"]),
                )
                tracks[track_id].add(spot)
    else:
        # Read the spots and tracks CSV files
        df_s = pd.read_csv(path_spots, index_col=0, skiprows=skiprows)
        df_t = pd.read_csv(path_tracks, index_col=2, skiprows=skiprows)

        # Sort the spots by frame
        df_s = df_s.sort_values(by=["FRAME"])

        # Loop through the spots
        for spotid, row_s in df_s.iterrows():
            # Check if the spot is in the tracks
            if row_s["TRACK_ID"] in df_t.index:
                row_t = df_t.loc[row_s["TRACK_ID"]]
                if row_t["LABEL"] not in tracks:
                    tracks[row_t["LABEL"]] = Track(row_t["LABEL"])

                # Append the spot to the list of spots
                spot = Spot(
                    str(spotid),
                    row_s["FRAME"],
                    (row_s["POSITION_Z"], row_s["POSITION_Y"], row_s["POSITION_X"]),
                )
                tracks[row_t["LABEL"]].add(spot)

    logging.debug("Finished reading spots and tracks CSV files: %s" % path_spots)
    return tracks


def save_tracks_napari(tracks: dict[str, Track], path_output: str):
    """
    Save the tracks to a CSV file in the format used by Napari.

    Args:
        tracks: Dictionary of tracks
        path_output: Path to the output CSV file
    """
    with open(path_output, "w") as f:
        f.write("TrackID,ObjectID,t,z,y,x\n")
        for track_id, track in tracks.items():
            for spot in track:
                if len(spot.coord) == 3:
                    f.write(
                        f"{track_id},{spot.id},{spot.t},{spot.coord[0]},{spot.coord[1]},{spot.coord[2]}\n"
                    )
                elif len(spot.coord) == 2:
                    f.write(f"{track_id},{spot.id},{spot.t},{0},{spot.coord[0]},{spot.coord[1]}\n")
                else:
                    raise Exception(f"len(spot.coord) must be 2 or 3. {len(spot.coord)} is given.")


def read_tracks_napari(path_tracks: str) -> dict[str, Track]:
    """
    Read the tracks CSV file and return a dictionary of tracks.

    Args:
        path_tracks: Path to the tracks CSV file

    Returns:
        Dictionary of tracks
    """
    tracks = {}
    with open(path_tracks, "r") as f:
        next(f)  # Skip the header
        for line in f:
            track_id, spot_id, t, z, y, x = line.strip().split(",")
            if track_id not in tracks:
                tracks[track_id] = Track(track_id)
            spot = Spot(spot_id, int(t), (float(z), float(y), float(x)))
            tracks[track_id].add(spot)
    return tracks


def detection_to_trackmate_xml(path_in_coords, path_out):
    df_objects = pd.read_csv(path_in_coords)
    list_t = df_objects.t.unique()
    list_t.sort()

    xml_text = f'    <AllSpots nspots="{len(df_objects)}">'
    for t in list_t:
        df_t = df_objects[df_objects.t == t]
        xml_text += f'\n      <SpotsInFrame frame="{t}">'
        for idx, row in df_t.iterrows():
            xml_text += f'\n        <Spot ID="{idx}" name="{row.object_id}" VISIBILITY="1" RADIUS="3" QUALITY="1.0" POSITION_T="{t}" POSITION_X="{row.x:.1f}" POSITION_Y="{row.y:.1f}" POSITION_Z="{row.z:.1f}" FRAME="{row.t:.0f}" />'
        xml_text += "\n      </SpotsInFrame>"
    xml_text += "\n    </AllSpots>"

    with open(path_out, "w") as f:
        f.write(xml_text)


def tracks_napari_to_trackmate_xml(path_tracks, path_out):
    tracks = read_tracks_napari(path_tracks)
    list_t = []
    for track in tracks.values():
        list_t.extend([spot.t for spot in track])
    list_t = list(set(list_t))
    list_t.sort()

    # Spots
    spot_index = 0
    map_spot_id_index = {}
    xml_text = f'    <AllSpots nspots="{sum([len(track) for track in tracks.values()])}">'
    for t in list_t:
        xml_text += f'\n      <SpotsInFrame frame="{t}">'
        for track_id, track in tracks.items():
            for spot in track:
                if spot.t == t:
                    xml_text += f'\n        <Spot ID="{spot_index}" name="{spot.id}" VISIBILITY="1" RADIUS="3" QUALITY="1.0" FRAME="{spot.t:.0f}" '
                    xml_text += f'POSITION_T="{spot.t}" POSITION_X="{spot.coord[2]:.1f}" POSITION_Y="{spot.coord[1]:.1f}" POSITION_Z="{spot.coord[0]:.1f}" />'
                    map_spot_id_index[spot.id] = spot_index
                    spot_index += 1
        xml_text += "\n      </SpotsInFrame>"
    xml_text += "\n    </AllSpots>"

    # Tracks
    xml_text += "\n    <AllTracks>"
    list_track_index = []
    for i, (track_id, track) in enumerate(tracks.items()):
        track_start = track.head.t
        track_stop = track.tail.t
        track_duration = track_stop - track_start
        xml_text += f'\n      <Track name="{track_id}" TRACK_ID="{i}" TRACK_INDEX="{i}" NUMBER_SPOTS="{len(track)}" '
        xml_text += f'TRACK_DURATION="{track_duration:.1f}" TRACK_START="{track_start:.1f}" TRACK_STOP="{track_stop:.1f}" >'
        list_track_index.append(i)
        for spot in track:
            if spot.next is not None:
                curr_spot_index = map_spot_id_index[spot.id]
                next_spot_index = map_spot_id_index[spot.next.id]
                edge_x = (spot.coord[2] + spot.next.coord[2]) / 2
                edge_y = (spot.coord[1] + spot.next.coord[1]) / 2
                edge_z = (spot.coord[0] + spot.next.coord[0]) / 2
                edge_t = (spot.t + spot.next.t) / 2
                xml_text += f'\n        <Edge SPOT_SOURCE_ID="{curr_spot_index}" SPOT_TARGET_ID="{next_spot_index}" '
                xml_text += f'EDGE_TIME="{edge_t}" EDGE_X_LOCATION="{edge_x}" EDGE_Y_LOCATION="{edge_y}" EDGE_Z_LOCATION="{edge_z}" />'
        xml_text += "\n      </Track>"
    xml_text += "\n    </AllTracks>"

    xml_text += "\n    <FilteredTracks>"
    for i in list_track_index:
        xml_text += f'\n      <TrackID TRACK_ID="{i}" />'
    xml_text += "\n    </FilteredTracks>"

    with open(path_out, "w") as f:
        f.write(xml_text)


def print_track_errors(track_error, log_file, print_detail=True):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    total_mismatch = track_error["total_mismatch"]
    total_missing = track_error["total_missing"]
    total_errors = total_mismatch + total_missing
    MOTA = track_error["MOTA"]
    mismatch_ratio = track_error["mismatch_ratio"]
    missing_ratio = track_error["missing_ratio"]
    details_by_frame = track_error["details_by_frame"]
    with open(log_file, "w") as f:
        f.write("Tracking evaluation\n")
        f.write(
            f"Total errors: {total_errors} (mismatch: {total_mismatch}, missing: {total_missing})\n"
        )
        f.write(
            f"MOTA: {MOTA:.4f}, Mismatch ratio: {mismatch_ratio:.4f}, Missing ratio: {missing_ratio:.4f}\n"
        )

    if print_detail:
        log_file_basename = os.path.basename(log_file)
        log_file_ext = os.path.splitext(log_file_basename)[1]
        log_file_dir = os.path.dirname(log_file)
        log_file_name = os.path.splitext(log_file_basename)[0]
        log_file_detail = os.path.join(log_file_dir, log_file_name + "_detail" + log_file_ext)
        with open(log_file_detail, "w") as f:
            # Write the details by tracks
            f.write("\n" + "-" * 30 + "Track by track detail (GT)" + "-" * 30 + "\n")
            tid_val_list = [(tid, val) for tid, val in track_error["details_by_track_gt"].items()]
            tid_val_list.sort(key=lambda x: x[1]["total"], reverse=True)
            have_pred_track_id = len(tid_val_list) > 0 and "pred_track_id" in tid_val_list[0][1]
            if have_pred_track_id:
                f.write("GT Track\tPred Track\tTotal\t\tMismatch\tMissing\n")
            else:
                f.write("GT Track\tTotal\t\tMismatch\tMissing\n")
            for track_id, val in tid_val_list:
                total = val["total"]
                mismatch = val["mismatch"]
                missing = val["missing"]
                if have_pred_track_id:
                    pred_track_id = val["pred_track_id"]
                    f.write(
                        f"{track_id}\t\t{pred_track_id}\t\t{total}\t\t{mismatch}\t\t{missing}\n"
                    )
                else:
                    f.write(f"{track_id}\t\t{total}\t\t{mismatch}\t\t{missing}\n")

            # Write the log of the GT tracks
            if len(tid_val_list) > 0 and "log" in tid_val_list[0][1]:
                f.write("\n" + "-" * 30 + "GT Track log" + "-" * 30 + "\n")
                for track_id, val in tid_val_list:
                    log = val["log"]
                    pred_track_id = val["pred_track_id"]
                    f.write(f"Track {track_id} ({pred_track_id}):\n")
                    f.write(log + "\n")

            # Write the details by tracks
            f.write("\n" + "-" * 30 + "Track by track detail (Pred)" + "-" * 30 + "\n")
            tid_val_list = [(tid, val) for tid, val in track_error["details_by_track_pred"].items()]
            tid_val_list.sort(key=lambda x: x[1]["total"], reverse=True)
            have_gt_track_id = len(tid_val_list) > 0 and "gt_track_id" in tid_val_list[0][1]
            if have_gt_track_id:
                f.write("Pred Track\tGT Track\tTotal\t\tMismatch\n")
            else:
                f.write("Pred Track\tTotal\t\tMismatch\n")
            for track_id, val in tid_val_list:
                total = val["total"]
                mismatch = val["mismatch"]
                if have_gt_track_id:
                    gt_track_id = val["gt_track_id"]
                    f.write(f"{track_id}\t\t{gt_track_id}\t\t{total}\t\t{mismatch}\n")
                else:
                    f.write(f"{track_id}\t\t{total}\t\t{mismatch}\n")

            # Write the log of the pred tracks
            if len(tid_val_list) > 0 and "log" in tid_val_list[0][1]:
                f.write("\n" + "-" * 30 + "Pred Track log" + "-" * 30 + "\n")
                tid_val_list.sort(key=lambda x: x[1]["gt_track_id"])
                for track_id, val in tid_val_list:
                    log = val["log"]
                    gt_track_id = val["gt_track_id"]
                    f.write(f"Track {track_id} ({gt_track_id}): \n")
                    f.write(log + "\n")

            # Write the details by frame
            f.write("\n" + "-" * 30 + "Frame by frame detail" + "-" * 30 + "\n")
            f.write("Frame\t\tMismatch\t\tMissing\n")
            for t, val in details_by_frame.items():
                mismatch = val["mismatch"]
                missing = val["missing"]
                f.write(f"{t}\t\t{len(mismatch)}\t\t{len(missing)}\n")
            f.write("\n" + "-" * 80 + "\n")
            for t, val in details_by_frame.items():
                mismatch = val["mismatch"]
                missing = val["missing"]
                f.write(f"Frame {t}:\n")
                f.write(f"Missing spots: {', '.join(missing)}\n")
                f.write("Mismatch\n")
                for item in mismatch:
                    if "GTTrack" in item:
                        f.write(
                            f"Spot {item['Spot']} in GT track {item['GTTrack']} mismatched. True Pred track: {item['TruePredTrack']}, tracked Pred track: {item['TrackedPredTrack']}\n"
                        )
                    else:
                        f.write(
                            f"Spot {item['Spot']} in pred track {item['PredTrack']} mismatched. True GT track: {item['TrueGTTrack']}, tracked GT track: {item['TrackedGTTrack']}\n"
                        )
