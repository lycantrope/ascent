# ASCENT

*Annotation‑free Self‑supervised Contrastive Embeddings for 3‑D Neuron Tracking*

---

## 📑 Table of Contents

- [ASCENT](#ascent)
  - [📑 Table of Contents](#-table-of-contents)
  - [📦 Installation (ASCENT core)](#-installation-ascent-core)
    - [Quick Install (from Source)](#quick-install-from-source)
      - [Use **`pip/conda`**](#use-pipconda)
      - [Use **`uv`**](#use-uv)
  - [🔬 Optional: Generate Neuron Candidates with **StarDist 3-D**](#-optional-generate-neuron-candidates-with-stardist-3-d)
    - [Why a Separate Environment?](#why-a-separate-environment)
    - [1. Install StarDist](#1-install-stardist)
    - [2. Download a Pre-trained Model](#2-download-a-pre-trained-model)
    - [3. Run the Segmentation Script](#3-run-the-segmentation-script)
      - [Key CLI Flags](#key-cli-flags)
      - [Expected HDF5 Layout](#expected-hdf5-layout)
      - [Expected Zarr/Zarr-ZipStore Layout](#expected-zarrzarr-zipstore-layout)
  - [🚀 Tracking Your Own Video with a Pre-trained NETr Model](#-tracking-your-own-video-with-a-pre-trained-netr-model)
    - [How ASCENT Runs Inference](#how-ascent-runs-inference)
    - [Available Pre-trained NETr Checkpoints](#available-pre-trained-netr-checkpoints)
    - [Example: Tracking Lightsheet Video](#example-tracking-lightsheet-video)
    - [Overriding Parameters from the Command Line](#overriding-parameters-from-the-command-line)
    - [Input File Formats](#input-file-formats)
      - [1a. Raw Video (HDF5)](#1a-raw-video-hdf5)
      - [1b. Raw Video (Zarr ZipStore)](#1b-raw-video-zarr-zipstore)
      - [2. Detections CSV](#2-detections-csv)
    - [Output Files](#output-files)
    - [Recommended Parameters](#recommended-parameters)
    - [Minimal Workflow](#minimal-workflow)
  - [🧪 Training Your Own NETr Model](#-training-your-own-netr-model)
    - [Quick Start](#quick-start)
    - [Configuration Schema](#configuration-schema)
    - [Notes On Parameters](#notes-on-parameters)
    - [Tips](#tips)
  - [📜 License](#-license)

---

## 📦 Installation (ASCENT core)

ASCENT runs on Python **3.10–3.12** (Linux/macOS/Windows). It’s a standard PyTorch project; install either the CPU build or CUDA build depending on your machine.

---

### Quick Install (from Source)

#### Use **`pip/conda`**
```sh
# Create a fresh environment (recommended)
conda create -n ascent python=3.12 && conda activate ascent
# or: python -m venv .venv && source .venv/bin/activate

# Clone the repository
git clone https://github.com/lu-lab/ascent.git
cd ascent

# Install ASCENT
pip install -e .

# Sanity check
python -c "import ascent, torch; print('ASCENT', ascent.__version__, '| CUDA available:', torch.cuda.is_available())"
```

#### Use **`uv`**
```sh
# Clone the repository
git clone https://github.com/lu-lab/ascent.git
cd ascent

# Install ASCENT with .venv
uv sync
 
# Sanity check
uv run python -c "import ascent, torch; print('ASCENT', ascent.__version__, '| CUDA available:', torch.cuda.is_available())"
```

---

## 🔬 Optional: Generate Neuron Candidates with **StarDist 3-D**

If you already have detections (centroids) from a separate pipeline, you can skip this part.

### Why a Separate Environment?
StarDist relies on TensorFlow 2.x, which often conflicts with the CUDA/PyTorch stack used by ASCENT. Keeping them in separate conda/pip environments prevents these issues.

---

### 1. Install StarDist

Follow the official guide → [https://github.com/stardist/stardist](https://github.com/stardist/stardist)
*(create a fresh environment first!)*

---

### 2. Download a Pre-trained Model

| Model ID                  | Download                                                                                                                              | Training volumes | Voxel size (µm)       | Target tissue      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | --------------------- | ------------------ |
| `celegans-free-NeRVE`     | [link](https://www.dropbox.com/scl/fo/dxcikcgwgi96yw5lefokq/AH8gG4qmRP86jTjiy4t-6GA?rlkey=8673p9td73sb1cnvmdu4phxlr&st=wfoj46mu&dl=0) | 2 (NeRVE)        | 0.3226 × 0.3226 × 1.5 | *C. elegans* brain |
| `celegans-device-Opterra` | [link](https://www.dropbox.com/scl/fo/djsxhdxfco9xhijdxpk6w/ALbtald5Lnh2p1dOibEs4Q4?rlkey=t07vmppwwm04gtg9eww0wvgdl&st=qljcb71i&dl=0) | 2 (in-house)     | 0.243 × 0.243 × 1.5   | *C. elegans* brain |

Dataset details are in the bioRxiv preprint (see “Datasets and ground truth”):
[https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1.full](https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1.full)

---

### 3. Run the Segmentation Script

`examples/scripts/stardist_segment.py` converts a 4-D HDF5 movie into:

* **Per-frame instance masks** (`--output_mask`)
* A **centroid table** (`--output_centroids`) that ASCENT can use

```bash
# Activate your StarDist environment
conda activate stardist

python examples/scripts/stardist_segment.py \
    --input            path/to/input.h5 \
    --input_channel    0 \
    --input_axis_order ZYX \
    --modelpath        path/to/model \
    --normalize        1 99.99 \
    --output_mask      path/to/output_mask.h5 \
    --output_centroids path/to/output_centroids.csv
```

#### Key CLI Flags

| Flag                      | Description                                                  | Default   |
| ------------------------- | ------------------------------------------------------------ | --------- |
| `--input`                 | Path to the HDF5 movie                                       | —         |
| `--input_channel`         | Channel index to segment                                     | `0`       |
| `--input_axis_order`      | Axis order of each 3-D stack (`Z`, `Y`, `X` in any order)    | `ZYX`     |
| `--modelpath`             | Folder containing StarDist 3-D `config.json` + weights `.h5` | —         |
| `--normalize P_MIN P_MAX` | Percentile range for intensity normalization                 | `1 99.99` |
| `--output_mask`           | HDF5 file to store label volumes (one dataset per frame)     | —         |
| `--output_centroids`      | CSV with `object_id,t,z,y,x` columns                         | —         |

Run:

```bash
python stardist_segment.py --help
```

for the full list of options.

#### Expected HDF5 Layout

```
root
└── t{frame}            # HDF5 group
    └── c{channel}      # 3-D dataset (Z × Y × X)
```
Example: `t250/c0` holds the Z-stack for frame 250, channel 0.

#### Expected Zarr/Zarr-ZipStore Layout

```
root
└── Zarr Array (T × C × Z x Y x X float32/uint16)
...
```

**After segmentation you’ll have:**

* `output_mask.h5` – datasets named `"0"`, `"1"`, … each containing a label volume (`uint32`)
* `output_centroids.csv` – centroid coordinates and bounding-box radii for each detected neuron

You can now feed these outputs, along with the raw movie, into ASCENT’s tracking pipeline.

---

## 🚀 Tracking Your Own Video with a Pre-trained NETr Model

### How ASCENT Runs Inference

ASCENT’s `run` mode performs:

1. **Feature extraction** – NETr encodes each neuron candidate into a high‑dimensional **embedding vector** that captures appearance and context.
2. **Tracking** – **HungarianTracker** links candidates across frames using cosine similarity in embedding space, producing continuous neuron identities.

Both steps run sequentially in `ascent run` (see [`tools/run_ascent.py`](src/ascent/tools/run_ascent.py)).

---

### Available Pre-trained NETr Checkpoints

All checkpoints share the **same NETr architecture hyperparameters** as in `examples/configs/track_template.py`.

| Model ID          | Download |Trained on (dataset)                                   | Microscope / Modality          | Voxel size (µm)       | Sample |
| ----------------- | --- | ------------------------------------------------------ | ------------------------------ | -------------------------------- | --- |
| `NETr-lightsheet` | [link](https://www.dropbox.com/scl/fi/jju3gb44f8wtwi0mqewqe/NETr-lightsheet.pth?rlkey=lboxgt7a4iq7ez1x99t2ax22a&st=5vbuordg&dl=0) | Lightsheet microscopy recordings                       | Light-sheet microscope             | 0.36 x 0.36 x 1.26                                | Head-fixed *C. elegans* |
| `NETr-NeRVE`      | [link](https://www.dropbox.com/scl/fi/tj8oxcuvefz3fvjg18jhq/NETr-NeRVE.pth?rlkey=wee4aqpi04tny88f5r4e17ann&st=q3v3s1hn&dl=0) | **NeRVE** dataset        | Spinning-disk confocal         | 0.3226 × 0.3226 × 1.5     | Freely moving *C. elegans* |
| `NETr-Opterra`    | [link](https://www.dropbox.com/scl/fi/japv9qzzhxer63vdxg714/NETr-Opterra.pth?rlkey=nl1ryrgzfgets2i2ljha9ljiv&st=ut8zabuu&dl=0) | **In-house** dataset | Swept-field confocal (Opterra) | 0.243 × 0.243 × 1.5  | Immobilized *C. elegans* |

* Use these checkpoints directly with the workflow below in [Example: Tracking Lightsheet Video](#example-tracking-lightsheet-video).
* To point ASCENT to a checkpoint, either edit `model_ckpt` in the config or **override from the CLI** (see [Overriding Parameters from the Command Line](#overriding-parameters-from-the-command-line)).

**Usage**

```python
# examples/configs/track_template.py
model_ckpt = "/path/to/checkpoints/NETr-NeRVE.pth"  # or NETr-lightsheet / NETr-Opterra
```

or from the command line:

```bash
python -m ascent run \
  --config examples/configs/track_template.py \
  --model_ckpt /path/to/checkpoints/NETr-Opterra.pth
```

---

### Example: Tracking Lightsheet Video

```bash
python -m ascent run --config examples/configs/track_template.py
```



The config file defines:

* **Dataset paths** (`dataset_file_image`, `dataset_file_coord`)
* **Dataset parameters** (`dataset_image_channel`, `dataset_axis_order`, `dataset_normalize`, `dataset_norm_p_low`, `dataset_norm_p_high`, `dataset_spacing`)
* **Model checkpoint & architecture** (`model_ckpt`, NETr params, patch size)
* **Tracking parameters** (`tracking_momentum`, `tracking_temperature`, etc.)
* **Runtime settings** (`runtime_output_dir`, `runtime_output_prefix`, etc.)

---

### Overriding Parameters from the Command Line

Any config value can be overrideen at runtime:

```bash
python -m ascent run \
  --config examples/configs/track_template.py \
  --dataset_file_image /path/to/raw.h5 \
  --dataset_file_coord /path/to/centroids.csv \
  --dataset_axis_order ZYX \
  --dataset_do_normalize true \
  --dataset_norm_p_low 1 \
  --dataset_norm_p_high 99.9 \
  --model_ckpt /path/to/checkpoints/NETr-lightsheet.pth \
  --runtime_output_dir ./outputs \
  --runtime_output_prefix sample
```

---

### Input File Formats

#### 1a. Raw Video (HDF5)

```
root
└── t0
    └── c0    (Z × Y × X float32/uint16)
└── t1
    └── c0
...
```


#### 1b. Raw Video (Zarr ZipStore)
```
root
└── Zarr Array (T × C × Z x Y x X float32/uint16)
...
```

* Axis order in each dataset is set via `dataset_axis_order` (`"ZYX"`, `"YXZ"`, etc.).
* `dataset_image_channel` selects the channel to load.
* For 1b, you can convert HDF5 sequence (FileType in 1a) to Zarr ZipStore (1b.) by [examples/scripts/convert_hdf_to_zarr.py](examples/scripts/convert_hdf_to_zarr.py)

#### 2. Detections CSV

Comma-separated with at least:

```
object_id,t,z,y,x
0,0,15.2,64.1,33.8
1,0,28.7,50.2,70.1
...
```

Coordinates are in voxel units, matching the raw video.

---

### Output Files

When `runtime_output_dir="outdir"` and `runtime_output_prefix="sample"`:

* `outdir/sample_pred_z.pt` – NETr embeddings (PyTorch tensor)
* `outdir/sample_pred_object_ids.pt` – Object IDs corresponding to rows in the embedding tensor
* `outdir/sample_tracks.csv` – Napari-compatible track table:
  `TrackID,ObjectID,t,z,y,x`

---

### Recommended Parameters

* Start with the default `examples/configs/track_template.py`.

| Parameter                  | Recommended Value | Notes                                                                 |
|----------------------------|-------------------|----------------------------------------------------------------------|
| `dataset_normalize`        | `percentile`      | Chooses per-frame intensity normalization method (`none` or `percentile`). |
| `dataset_norm_p_low`       | `1.0`             | Lower percentile bound for normalization.                           |
| `dataset_norm_p_high`      | `99.99`           | Upper percentile bound for normalization.                           |
| `tracking_momentum`        | `0.5`             | Momentum factor for updating track embeddings over time.            |
| `tracking_temperature`     | `0.05`            | Softmax temperature for similarity scoring in the tracker.          |
| `tracking_max_gap_frames`  | `9999`            | Maximum allowed gap (in frames) when linking detections into tracks.|
| `tracking_w_within`        | `0`               | Weight for within-track distance when estimating cutoffs.           |
| `runtime_batch_size_frame` | `64`              | Number of frames processed per batch; increase if memory allows.    |
| `runtime_device`           | `"cuda"`          | Device for computation (`"cuda"` for GPU, `"cpu"` for CPU).          |

---

### Minimal Workflow

1. (Optional) Generate detections with StarDist 3‑D.
2. Point `model_ckpt` to one of the pre‑trained NETr checkpoints.
3. Run `ascent run --config examples/configs/track_template.py`.
4. Inspect `*_tracks.csv` in Napari.

---

## 🧪 Training Your Own NETr Model

### Quick Start

Train NETr from scratch or fine‑tune using a single **config file** and the training entry point:

```bash
python -m ascent train \
  --config examples/configs/train_NETr_template.py
```

ASCENT supports multi‑GPU training via PyTorch DistributedDataParallel (DDP) and automatically uses **all visible GPUs**.

### Configuration Schema

A training config is a small Python (or YAML) file that declares the **model**, **dataset**, **transforms**, **dataloader**, **losses**, **optimizer**, and run settings.

See the [example](./examples/configs/train_NETr_template.py) for an example config.

### Notes On Parameters
* `device`: `"cpu"`, `"cuda"`, `"mps"`, or `"auto"`. `"auto"` picks an appropriate GPU device if available.
* `port`: Internal port number used by DDP. Not used for single‑GPU or CPU runs.
* `image_file`, `coord_file`: Paths to the image and detection data used for self‑supervised NETr training. See [Input File Formats](#input-file-formats) for details.
* `dataloader`: Key–value dictionary passed to PyTorch’s `DataLoader` (e.g., `num_workers`, `batch_size`). `batch_size` has the largest impact on GPU memory usage and training speed—pick the largest value that fits without OOM.
* `losses`: A list of losses. Each item specifies a `"class"` with `"params"`. The total loss is the **weighted sum** of items using each entry’s `"weight"`.
* `optimizer`, `scheduler`: Defaults to Adam with learning rate **1e‑3** and **no scheduler** if not specified in the config.
* `epochs`, `time_limit`: Maximum epochs and/or wall‑clock time (seconds). Training stops when either limit is reached. If unspecified, no limit is applied.
* `save_every_n_epochs`, `save_time_span`: Control checkpoint frequency by epoch count or elapsed time (seconds).
* `continue_training`: If `True`, resumes from the most recent checkpoint found alongside `model_save_path`.

### Tips

* Keep `axis_order`, `spacing`, and `image_channel` consistent with your data. Use percentile normalization to stabilize training across recordings.
* Start with `batch_size=4`; increase if memory allows. If you see too few objects per frame, reduce crop sizes or jitter ranges.
* You can list **multiple training datasets** (as a list) to mix sources; batches are interleaved across loaders each epoch.
* For custom learning‑rate schedules or per‑layer LRs, add a `scheduler` or `optimizer.layer_lrs` to the config.

---

## 📜 License

ASCENT is released under the MIT license. © Haejun Han.
