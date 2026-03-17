# Unified Graph Weather (Regional, Multi‑Scale, Lightweight)

> Graph‑based regional weather forecasting pipeline: from **mesh + static features** to **monthly shards**, **training**, and **evaluation**.  
> Includes a general GAT backbone with **vertical self‑attention** + **horizontal graph attention**, and task‑specific heads for **precipitation** and **near‑surface wind**.

---

## Highlights

- **End‑to‑end pipeline** (notebooks `01`–`06`):
  1. Mesh creation for your ROI (with optional ring/outer bands)
  2. Static geophysical features on nodes (land/sea, orography, etc.)
  3. Channel **statistics** (means/std; `tp` in log-domain)
  4. Single **static graph package** (`.npz`) for the dataloader
  5. Monthly **graph shards** (`graph_YYYY-MM.pt`) with `[T, C, N]` tensors
  6. **Evaluation** notebook (1‑step and optional AR multi‑step)
- **Lightweight backbone**: axial design = vertical SDPA (channels) + horizontal GAT (nodes).
- **Task heads**: precipitation (sparse‑aware loss) and wind (vector loss with magnitude/angle terms).
- **Production‑ready dataloader**: **prefetch**, deterministic per‑file shuffling, Tin/Tout override.

---

## Repository Layout (recommended)

```
repo/
├─ model/
│  ├─ unified_graph_weather_model.py           # GAT backbone + Lightning
│  ├─ unified_graph_weather_model_tp.py        # TP head (precipitation)
│  └─ unified_graph_weather_model_uv10.py      # UV10 head (u10, v10)
├─ utils/
│  └─ unified_graph_weather_dataloader.py      # DataModule + Dataset + Prefetch
├─ notebooks/
│  ├─ 01_mesh_creation.ipynb
│  ├─ 02_mesh_with_geo_factor.ipynb
│  ├─ 03_stats.ipynb
│  ├─ 04_build_static _graph.ipynb
│  ├─ 05_shard_creation.ipynb
│  └─ 06_model_evaluation.ipynb
├─ data/
│  ├─ graph/                                   # static graph package
│  │  └─ graph_static_from_stats_vgeo_aligned.npz
│  ├─ shards_graph/                            # monthly shards
│  │  ├─ graph_1980-10.pt
│  │  ├─ graph_1980-11.pt
│  │  └─ ...
│  └─ stats/
│     └─ stats_025deg_1980_2013.npz
├─ train/
│  ├─ train_unified_graph_weather_general_model_1step.py
│  ├─ train_unified_graph_weather_precip_1step_finetune.py
│  └─ train_unified_graph_weather_wind_model_1step.py
└─ README.md
```

> English‑annotated copies of the notebooks are also available (suffix `_en.ipynb`).

---

## Installation

### 1) Python & PyTorch
- Python **3.10+** recommended
- PyTorch **2.1+** with CUDA (if using GPU)

```bash
# Example (adjust versions to your CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
# If you don't maintain requirements.txt, the minimal set is:
pip install pytorch-lightning numpy scipy xarray netCDF4 pandas rasterio shapely pyproj \
            matplotlib tqdm pyarrow
```

**Optional (mesh):** If you generate unstructured meshes with **JIGSAW**, install `jigsawpy` and its dependencies. The provided notebooks also work with regular lat‑lon grids.

### 2) Reproducibility & speed
The scripts enable **TF32** on Ampere+ and set `torch.set_float32_matmul_precision("high")`. For deterministic behavior you can add `deterministic=True` to the Lightning `Trainer` (slower).

---

## Data Preparation

> ERA5 (or similar) is expected as the source. The pipeline operates on **node‑aligned** tensors lifted onto your mesh.

### Step A — Build the mesh *(notebooks/01_... and 02_...)*
- Define your **ROI** and optional **ring/outer** zones with coarser resolution.
- Choose a spacing (e.g., ROI `0.25°`, ring `0.5°`, outer `1.0°`).
- Save node coordinates `coord_lon[N]`, `coord_lat[N]`.
- Attach static channels to each node (e.g., land/sea mask, orography, distance‑to‑coast, roughness) and optional 2D positional encodings.

### Step B — Compute channel stats *(notebooks/03_stats.ipynb)*
- Produce a `.npz` with:
  - `order` — list[str], channel names (e.g., `["t2m@sfc","u10@sfc","v10@sfc","tp_log@sfc", ...]`)
  - `mean` — float32 vector
  - `std` — float32 vector (clamped to ≥1e‑6)
- For precipitation use **log1p**: store that channel as `tp_log` in `order`.

### Step C — Build the static graph package *(notebooks/04_build_static _graph.ipynb)*
Create a single `.npz` consumed by the dataloader:
- `coord_lon[N]`, `coord_lat[N]`
- `static_geo_mesh[N,4]` (already normalized if desired)
- `static_pos2d_mesh[4,N]` *(optional)*
- **Edges** (one of):
  - `edge_index_horiz_base[2, Eh]` **(preferred)**
  - or `edge_index_horiz[2, E_total]` + `horiz_edge_ptr[L+1]` (slice level 0)
  - or legacy `edges_h_src`/`edges_h_dst` or `edges_horiz`
- Optional meta: `levels`, `level_offsets`, `mesh_hash`

Path used by scripts: `./data/graph/graph_static_from_stats_vgeo_aligned.npz`

### Step D — Create monthly shards *(notebooks/05_shard_creation.ipynb)*
For each month, write `graph_YYYY-MM.pt` with keys:
- `values` : **[T, C, N]**
- `Tin`, `Tout` : ints used during slicing
- `stride` : usually `1`
- `sample_starts` : vector of starting indices
- `channel_names` : list of channel names (same order as training)
- *(optional)* dtype float16 to save space

---

## Training

### General 1‑step model (all channels)
```
python train/train_unified_graph_weather_general_model_1step.py \
  -- (edit paths & hyperparams inside the script)
```
- Dataloader: `GraphWeatherDataModule`
- Model: `UnifiedGraphWeatherGATLightning` (backbone = vertical SDPA + horizontal GAT)
- Outputs checkpoints under `./lightning_logs/graph_gat_unified_1step/`

### Precipitation fine‑tune (tp_log@sfc)
```
python train/train_unified_graph_weather_precip_1step_finetune.py
```
- Loads a **general** checkpoint into the base net and trains a TP head
- Loss = Huber (log‑space) + α·L1 in **mm** with **wet/dry** re‑weighting

**Key knobs:** `wet_thr_mm`, `wet_weight`, `alpha_mm`, `gamma_focal`

### Wind fine‑tune (u10, v10)
```
python train/train_unified_graph_weather_wind_model_1step.py
```
- Loads a general checkpoint; trains a UV head
- Loss combines component MSE/Huber with optional magnitude/direction terms

---

## Evaluation

Use `notebooks/06_model_evaluation.ipynb`:
- Lead‑1 metrics (MAE/RMSE) on ROI
- Optional **autoregressive** rollout for multiple leads (e.g., K=4) and plots
- Spatial error maps, histograms (esp. for precipitation)

> You can also script evaluation: load the checkpoint with the same DataModule and call `trainer.validate` or run custom inference loops.

---

## Dataloader & File Specs (quick reference)

### Static graph `.npz`
- **Required:** `coord_lon[N]`, `coord_lat[N]`, `static_geo_mesh[N,4]`
- **Optional:** `static_pos2d_mesh[4,N]`, edges as listed above, metadata

### Shards `graph_YYYY-MM.pt`
- `values` **[T, C, N]**
- `Tin`, `Tout`, `stride`, `sample_starts`
- `channel_names` (list[str])

### Batch (to the model)
- `((x_all, geo, pos2d), y_all)` with shapes:
  - `x_all`: **[B, Tin,  C, N]**
  - `y_all`: **[B, Tout, C, N]**
  - `geo`:   **[B, 4, N]**
  - `pos2d`: **[B, 4, N]**

---

## Tips & Troubleshooting

- **edge_index missing**: ensure `edge_index_horiz_base` exists in the static `.npz` (or provide `edge_index_horiz` + `horiz_edge_ptr` to slice level‑0).
- **Channel mismatch**: keep **`channel_names`** consistent across shards, stats, and training. For `tp`, use `tp_log` in stats.
- **Memory**: use `float16` shards; start with `B=1–2`. Enable `prefetch` and set a sensible `num_workers`.
- **Speed**: try `torch.compile` (PyTorch 2.1+). If it fails, the script falls back gracefully.

---

## Acknowledgments

- ERA5 reanalysis (Hersbach et al., 2020)
- Natural Earth data (Kelso & Patterson, 2010) for land/sea masks & river centerlines
- GAT / Transformers literature and recent ML weather models (GraphCast, etc.)

If you use this code, please consider citing the relevant datasets and papers.

---

## License
This repository is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 📄 Paper

[![arXiv](https://img.shields.io/badge/arXiv-2603.13563-b31b1b.svg)](https://arxiv.org/abs/2603.13563)

**MR-GNF: Multi-Resolution Graph Neural Forecasting on Ellipsoidal Earth Meshes for Efficient Regional Weather Prediction**

> A multi-resolution graph neural network for efficient regional weather forecasting on ellipsoidal Earth meshes.

📑 https://arxiv.org/abs/2603.13563  
📄 https://arxiv.org/pdf/2603.13563.pdf

If you find this work useful, please consider citing:

```bibtex
@article{shchur2026mrgnf,
  title={MR-GNF: Multi-Resolution Graph Neural Forecasting on Ellipsoidal Earth Meshes for Efficient Regional Weather Prediction},
  author={Shchur, Andrii and Skarga-Bandurova, Inna},
  year={2026},
  journal={arXiv preprint arXiv:2603.13563}
}
