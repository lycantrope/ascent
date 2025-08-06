# ASCENT

*Annotation‑free Self‑supervised Contrastive Embeddings for 3‑D Neuron Tracking*

---

## 📦 Installation (ASCENT core)

TODO

---
## 🔬 Optional: Generate neuron candidates with **StarDist 3-D**

### Why a separate environment?
StarDist relies on TensorFlow 2.x, which often clashes with the CUDA/PyTorch stack used by ASCENT. Keeping them in different conda / pip environments avoids those headaches.

### 1. Install StarDist
Follow the official guide → <https://github.com/stardist/stardist>  
*(create a fresh environment first!)*

### 2. Grab a pre-trained model
| Model ID | Download | Training volumes | Voxel size (µm) | Target tissue |
|----------|----------|------------------|-----------------|---------------|
| `celegans-free-NeRVE` | [link](https://www.dropbox.com/scl/fo/dxcikcgwgi96yw5lefokq/AH8gG4qmRP86jTjiy4t-6GA?rlkey=8673p9td73sb1cnvmdu4phxlr&st=wfoj46mu&dl=0) | 2 (NeRVE) | 0.3226 × 0.3226 × 1.5 | *C. elegans* brain |
| `celegans-device-Opterra` | [link](https://www.dropbox.com/scl/fo/djsxhdxfco9xhijdxpk6w/ALbtald5Lnh2p1dOibEs4Q4?rlkey=t07vmppwwm04gtg9eww0wvgdl&st=qljcb71i&dl=0) | 2 (in-house) | 0.243 × 0.243 × 1.5 | *C. elegans* brain |

Dataset details are in the bioRxiv preprint (see “Datasets and ground-truth”):  
<https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1.full>

### 3. Run the segmentation script

`examples/scripts/stardist_segment.py` turns a 4-D HDF5 movie into

* **per-frame instance masks** (`--output_mask`) and
* a **centroid table** (`--output_centroids`) that ASCENT can ingest.

```bash
# activate your StarDist env
conda activate stardist

python examples/scripts/stardist_segment.py \
    --input            your-path-to-input-data.h5 \
    --input_channel    0 \
    --input_axis_order ZYX \
    --modelpath        your-path-to-model-directory \
    --normalize        1 99.99 \
    --output_mask      your-path-to-output-mask.h5 \
    --output_centroids your-path-to-output-centroids.csv
```

#### Key CLI flags

| Flag                      | Meaning                                                      | Default   |
| ------------------------- | ------------------------------------------------------------ | --------- |
| `--input`                 | Path to the HDF5 movie                                       | —         |
| `--input_channel`         | Channel index to segment                                     | `0`       |
| `--input_axis_order`      | Axis order of each 3-D stack (`Z`, `Y`, `X` in any order)    | `ZYX`     |
| `--modelpath`             | Folder containing StarDist 3-D `config.json` + weights `.h5` | —         |
| `--normalize P_MIN P_MAX` | Percentile range for intensity normalisation                 | `1 99.99` |
| `--output_mask`           | HDF5 file to store label volumes (one dataset per frame)     | —         |
| `--output_centroids`      | CSV with `object_id,t,z,y,x` columns             | —         |

Run `python stardist_segment.py --help` for the full list.

#### Expected HDF5 layout

```
root
└── t{frame}            # HDF5 group
    └── c{channel}      # 3-D dataset (Z × Y × X)
```

Example: `t250/c0` holds the Z-stack for frame 250, channel 0.

After segmentation you’ll have:

* `your-path-to-model-directory.h5` – datasets named `"0"`, `"1"`, … each containing a label volume (`uint32`)
* `your-path-to-output-centroids.csv` – centroid coordinates and bounding-box radii for every detected neuron

Feed these outputs, together with the raw movie, into ASCENT’s tracking pipeline.


---

## 🚀 Quick‑start (once you have detections)

TODO

---

## 📜 License

ASCENT is released under the MIT license. © Haejun Han.