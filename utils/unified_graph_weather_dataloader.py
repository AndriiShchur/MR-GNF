# graph_weather_dataloader.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import re
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# ==========================
# Utils
# ==========================
def _parse_ym_from_name(p: Path):
    """Extract year and month from a shard name ``graph_YYYY-MM.pt``.


    Returns
    -------
    (year, month) as integers, or ``(None, None)`` if the pattern does not match.
    """
    m = re.search(r"(\d{4})-(\d{2})\.pt$", p.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _sincos_pos2d_nodes(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Node‑wise 2D positional encoding ``[4, N]``.


    We use ``sin(ϕ), cos(ϕ), sin(λ·cos(\bar{ϕ})), cos(λ·cos(\bar{ϕ}))`` where
    ``\bar{ϕ}`` is the mean latitude of the domain (kept stable as in the mesh
    build script). This encoding provides rotationally aware longitudinal terms
    and avoids extreme stretching near the poles for regional meshes.
    """
    lat = np.deg2rad(lat_deg.astype(np.float32, copy=False))
    lon = np.deg2rad(lon_deg.astype(np.float32, copy=False))
    lat0 = float(lat.mean())
    lon_adj = lon * np.cos(lat0)
    pos = np.stack([np.sin(lat), np.cos(lat), np.sin(lon_adj), np.cos(lon_adj)], axis=0)
    return pos.astype(np.float32, copy=False)


def _load_graph_static(npz_path: Path):
    """Load the static multi‑scale graph description from ``.npz``.


    Required keys
    -------------
    - ``coord_lon[N]``, ``coord_lat[N]``: node coordinates in degrees.
    - ``static_geo_mesh[N,4]``: 4 geophysical channels (already normalized).


    Optional keys for edges (any one of the following is accepted)
    ----------------------------------------------------------------
    - ``edge_index_horiz_base[2, Eh]`` (preferred)
    - ``edge_index_horiz[2, E_total]`` + ``horiz_edge_ptr[L+1]`` (slice level 0)
    - legacy: ``edges_h_src`` + ``edges_h_dst`` or ``edges_horiz``


    Optional metadata
    -----------------
    - ``levels`` (list‑like), ``level_offsets`` (np.int32), ``mesh_hash`` (str)


    Returns
    -------
    dict with fields ``lon``, ``lat``, ``geo`` (``[4,N]``), ``pos2d`` (``[4,N]``),
    ``edge_index_base`` (``torch.LongTensor[2, Eh]`` or ``None``), and ``meta``.
    """
    d = np.load(npz_path, allow_pickle=True)

    # --- coords
    lon = d["coord_lon"].astype(np.float32, copy=False)
    lat = d["coord_lat"].astype(np.float32, copy=False)
    N = int(lon.shape[0])

    # --- static geo 
    if "static_geo_mesh" not in d:
        raise KeyError(f"{npz_path} must contain 'static_geo_mesh'")
    static_geo = d["static_geo_mesh"].astype(np.float32, copy=False)  # [N,4]
    if static_geo.shape[0] != N:
        raise ValueError("static_geo_mesh first dim must match coord arrays")

    # --- pos2d
    if "static_pos2d_mesh" in d:
        pos2d = d["static_pos2d_mesh"].astype(np.float32, copy=False).T  # [4,N]
    else:
        pos2d = _sincos_pos2d_nodes(lat, lon)  # [4,N]

    # --- levels/meta
    levels = [str(x) for x in (d["levels"].tolist() if "levels" in d else [])]
    level_offsets = d["level_offsets"].astype(np.int32, copy=False) if "level_offsets" in d else None
    mesh_hash = None
    if "mesh_hash" in d:
        mh = d["mesh_hash"].item()
        mesh_hash = (mh.decode() if isinstance(mh, (bytes, np.bytes_)) else str(mh))

    # --- edge_index_base
    edge_index = None

    if "edge_index_horiz_base" in d:
        ei = d["edge_index_horiz_base"].astype(np.int64, copy=False)
        edge_index = torch.from_numpy(ei)

    elif "edge_index_horiz" in d and "horiz_edge_ptr" in d:
        e_all = d["edge_index_horiz"]   # [2, E_total] або [E_total, 2]
        if e_all.ndim == 2 and e_all.shape[0] == 2:
            e_all = e_all
        elif e_all.ndim == 2 and e_all.shape[1] == 2:
            e_all = e_all.T
        else:
            raise ValueError("edge_index_horiz must be [2, E] or [E, 2]")

        ptr = d["horiz_edge_ptr"].astype(np.int64, copy=False)  # [L+1]
        if ptr.size < 2:
            raise ValueError("horiz_edge_ptr must have at least 2 entries")
        e0 = e_all[:, int(ptr[0]):int(ptr[1])].astype(np.int64, copy=False)  # slice рівня 0

        if level_offsets is not None and int(level_offsets[0]) != 0:
            e0 = e0 - int(level_offsets[0])

        if e0.min() < 0 or e0.max() >= N:
            raise RuntimeError(f"edge_index_base out of range after slice: min={e0.min()}, max={e0.max()}, N={N}")

        edge_index = torch.from_numpy(e0.copy())

    elif "edges_h_src" in d and "edges_h_dst" in d:
        src = torch.from_numpy(d["edges_h_src"].astype(np.int64, copy=False))
        dst = torch.from_numpy(d["edges_h_dst"].astype(np.int64, copy=False))
        edge_index = torch.stack([src, dst], dim=0)
    elif "edges_horiz" in d:
        eh = d["edges_horiz"]
        if eh.ndim == 2 and eh.shape[1] == 2:
            edge_index = torch.from_numpy(eh.astype(np.int64, copy=False)).t().contiguous()
        elif eh.ndim == 2 and eh.shape[0] == 2:
            edge_index = torch.from_numpy(eh.astype(np.int64, copy=False))

    meta = {
        "N_mesh": N,
        "levels": levels,
        "level_offsets": level_offsets,
        "mesh_hash": mesh_hash,
    }
    geo = static_geo.T  # [4,N]
    return {
        "lon": lon, "lat": lat, "geo": geo, "pos2d": pos2d,      # [4,N] і [4,N]
        "edge_index_base": edge_index,                            # torch.Long[2,Eh] або None
        "meta": meta,
    }


# ==========================
# Dataset
# ==========================
class GraphShardDataset(Dataset):
    """Read monthly graph shards and emit samples for a GNN/GAT model.

    ``__getitem__`` returns ``((x_all, geo, pos2d), y_all)`` with shapes:
      - ``x_all``: ``[Tin,  C, N]``
      - ``y_all``: ``[Tout, C, N]``
      - ``geo``  : ``[4, N]``
      - ``pos2d``: ``[4, N]``

    Parameters
    ----------
    shards_root : str
        Directory with files named ``graph_YYYY-MM.pt``.
    years : range
        Years to include (e.g., ``range(1980, 2011)``).
    static_graph_npz : str
        Path to the static mesh ``.npz`` (see ``_load_graph_static``).
    allowed_months : list[int] | None
        If provided, restricts months (1..12). ``None`` means all months.
    force_tin_tout : tuple[int, int] | None
        If set, overrides ``Tin``/``Tout`` in the shard and re‑slices time windows
        with ``stride=1`` during loading.
    dtype_out : {"float16", "float32"}
        Output dtype for tensors returned by ``__getitem__``.
    shuffle_within_file : bool
        If ``True``, per‑file sample starts are deterministically shuffled using
        a RNG seeded from ``year*100 + month``.
    """

    def __init__(self,
                 shards_root: str,
                 years: range,
                 static_graph_npz: str = "./data/graph/graph_static_from_stats_vgeo_aligned.npz",
                 allowed_months: Optional[List[int]] = None,  # None → all 1..12
                 force_tin_tout: Optional[Tuple[int, int]] = None,  # None → as in shard
                 dtype_out: str = "float32",
                 shuffle_within_file: bool = True):
        super().__init__()
        self.root = Path(shards_root)
        self.years = set(int(y) for y in years)
        self.allowed_months = None if allowed_months is None else {int(m) for m in allowed_months}
        self.force_tin_tout = force_tin_tout
        assert dtype_out in {"float16", "float32"}
        self.dtype_out = torch.float16 if dtype_out == "float16" else torch.float32
        self.shuffle_within_file = bool(shuffle_within_file)

        # ---- зчитуємо статичні фічі графа один раз ----
        sg = _load_graph_static(Path(static_graph_npz))
        self.geo   = torch.from_numpy(sg["geo"].copy())     # [4,N]
        self.pos2d = torch.from_numpy(sg["pos2d"].copy())   # [4,N]
        self.N = int(sg["geo"].shape[1])
        self.edge_index_base = sg["edge_index_base"]  # torch.Long[2,Eh] або None
        self.static_meta = sg["meta"]

        files = []
        for p in sorted(self.root.glob("graph_*.pt")):
            y, m = _parse_ym_from_name(p)
            if y is None:
                continue
            if (y in self.years) and (self.allowed_months is None or m in self.allowed_months):
                files.append(p)
        if not files:
            raise FileNotFoundError(f"No graph_YYYY-MM.pt shards in {self.root}")

        self.entries: List[Tuple[str, int]] = []  # (path, s_start)
        self._file_meta: Dict[str, Dict] = {}     # path -> {T,C,N,Tin,Tout,stride,S}
        self._file_counts: Dict[str, int] = {}

        for p in files:
            d = torch.load(p, map_location="cpu")
            values = d["values"]         # [T,C,N]
            T, C, N = values.shape
            if N != self.N:
                raise ValueError(f"{p.name}: N({N}) != static N({self.N})")

            Tin  = int(d["Tin"])
            Tout = int(d["Tout"])
            stride = int(d["stride"])
            starts = d["sample_starts"].cpu().numpy().astype(np.int64)  # [S]
            if self.force_tin_tout is not None:

                fTin, fTout = self.force_tin_tout
                max_start = T - (fTin + fTout)
                if max_start < 0:
                    continue
                starts = np.arange(0, max_start + 1, 1, dtype=np.int64)
                Tin, Tout = int(fTin), int(fTout)

            if self.shuffle_within_file:
                rng = np.random.default_rng(int(y) * 100 + int(m))
                rng.shuffle(starts)

            for s in starts:
                self.entries.append((str(p), int(s)))

            self._file_meta[str(p)] = {
                "T": T, "C": C, "N": N, "Tin": Tin, "Tout": Tout, "stride": stride, "S": int(len(starts))
            }
            self._file_counts[str(p)] = int(len(starts))

        # cache last shard
        self._last_path: Optional[str] = None
        self._last_data = None

        self.C = next(iter(self._file_meta.values()))["C"]
        self.Tin = next(iter(self._file_meta.values()))["Tin"]
        self.Tout = next(iter(self._file_meta.values()))["Tout"]

    def __len__(self) -> int:
        return len(self.entries)

    def _load_shard(self, path: str):
        if self._last_path != path:
            self._last_data = torch.load(path, map_location="cpu")
            self._last_path = path
        return self._last_data

    def __getitem__(self, i: int):
        path, s = self.entries[i]
        meta = self._file_meta[path]
        Tin, Tout = meta["Tin"], meta["Tout"]

        d = self._load_shard(path)
        vals = d["values"]  # torch Tensor [T,C,N], float16
        x_all = vals[s:s+Tin].to(dtype=self.dtype_out)        # [Tin,C,N]
        y_all = vals[s+Tin:s+Tin+Tout].to(dtype=self.dtype_out)  # [Tout,C,N]

        geo   = self.geo.to(dtype=self.dtype_out)     # [4,N]
        pos2d = self.pos2d.to(dtype=self.dtype_out)   # [4,N]

        return (x_all, geo, pos2d), y_all

    # ---------- diagnostics ----------
    @property
    def file_counts(self) -> Dict[str, int]:
        return dict(self._file_counts)

    @property
    def N_mesh(self) -> int:
        return self.N


# ==========================
# Prefetch (GPU)
# ==========================
class PrefetchLoader:
    """DataLoader wrapper with optional CUDA stream prefetch.

    If ``device`` starts with ``"cuda"``, batches are staged to GPU on a
    background stream and yielded one step behind to overlap host‑to‑device
    copies with compute. On CPU, this acts as a thin transparent wrapper.
    """
    def __init__(self, loader: DataLoader, device: str = 'cuda'):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.device.startswith("cuda"):
            stream = torch.cuda.Stream()
            first = True
            for (x_all, geo, pos2d), y_all in self.loader:
                with torch.cuda.stream(stream):
                    x_all = x_all.to(self.device, non_blocking=True)
                    geo   = geo.to(self.device, non_blocking=True)
                    pos2d = pos2d.to(self.device, non_blocking=True)
                    y_all = y_all.to(self.device, non_blocking=True)
                    next_in, next_tgt = (x_all, geo, pos2d), y_all
                if not first:
                    yield cur_in, cur_tgt
                else:
                    first = False
                torch.cuda.current_stream().wait_stream(stream)
                cur_in, cur_tgt = next_in, next_tgt
            yield cur_in, cur_tgt
        else:
            # CPU → прозора прокладка
            for batch in self.loader:
                yield batch


# ==========================
# Lightning DataModule
# ==========================
class GraphWeatherDataModule(pl.LightningDataModule):
    """Lightning ``DataModule`` for graph‑sharded weather time series.

    Exposes ``edge_index_base`` and ``N_mesh`` so the model can initialize its
    topology (e.g., build a GAT). Also publishes convenient properties like
    ``train_steps_per_epoch`` and ``val_steps``.

    Batch structure
    ---------------
    ``((x_all, geo, pos2d), y_all)`` with shapes
      - ``x_all``: ``[B, Tin,  C, N]``
      - ``y_all``: ``[B, Tout, C, N]``
      - ``geo``  : ``[B, 4, N]``
      - ``pos2d``: ``[B, 4, N]``
    """
    def __init__(self,
                 shards_root_train: str = "./data/shards_graph",
                 shards_root_val: str   = "./data/shards_graph",
                 years_train: range = range(1980, 2011),
                 years_val:   range = range(2011, 2013),
                 static_graph_npz: str = "./data/graph/graph_static_from_stats_vgeo_aligned.npz",
                 allowed_months: Optional[List[int]] = None,
                 force_tin_tout: Optional[Tuple[int,int]] = None,
                 dtype_out: str = "float32",
                 shuffle_within_file_train: bool = True,
                 shuffle_within_file_val:   bool = False,
                 batch_size_train: int = 1,
                 batch_size_val:   int = 1,
                 num_workers: Optional[int] = None,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 device: str = "cuda"):
        super().__init__()
        self.paths = {"train": shards_root_train, "val": shards_root_val}
        self.years = {"train": years_train, "val": years_val}
        self.kw_ds = dict(
            static_graph_npz=static_graph_npz,
            allowed_months=allowed_months,
            force_tin_tout=force_tin_tout,
            dtype_out=dtype_out,
        )
        self.shuffle = {"train": shuffle_within_file_train, "val": shuffle_within_file_val}
        self.bs = {"train": batch_size_train, "val": batch_size_val}
        self.num_workers = num_workers if num_workers is not None else max((torch.get_num_threads() or 1) - 1, 0)
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.device = device

        self._train_ds = None
        self._val_ds   = None
        self._train_loader = None
        self._val_loader   = None

        self.edge_index_base: Optional[torch.Tensor] = None  # Long[2,Eh], None
        self.N_mesh: Optional[int] = None
        self.C: Optional[int] = None
        self.Tin: Optional[int] = None
        self.Tout: Optional[int] = None

    def setup(self, stage=None):
        self._train_ds = GraphShardDataset(
            shards_root=self.paths["train"],
            years=self.years["train"],
            shuffle_within_file=self.shuffle["train"],
            **self.kw_ds
        )
        self._val_ds = GraphShardDataset(
            shards_root=self.paths["val"],
            years=self.years["val"],
            shuffle_within_file=self.shuffle["val"],
            **self.kw_ds
        )

        self.N_mesh = self._train_ds.N_mesh
        self.edge_index_base = self._train_ds.edge_index_base
        self.C = self._train_ds.C
        self.Tin = self._train_ds.Tin
        self.Tout = self._train_ds.Tout

    def _make_loader(self, ds: Dataset, bs: int, shuffle: bool):
        base = DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
        return PrefetchLoader(base, device=self.device)

    def train_dataloader(self):
        self._train_loader = self._make_loader(self._train_ds, self.bs["train"], shuffle=True)
        return self._train_loader

    def val_dataloader(self):
        self._val_loader = self._make_loader(self._val_ds, self.bs["val"], shuffle=False)
        return self._val_loader

    # handy stats
    @property
    def train_samples(self):
        return len(self._train_ds) if self._train_ds is not None else None

    @property
    def val_samples(self):
        return len(self._val_ds) if self._val_ds is not None else None

    @property
    def train_steps_per_epoch(self):
        if self._train_ds is None:
            return None
        return math.ceil(len(self._train_ds) / self.bs["train"])

    @property
    def val_steps(self):
        if self._val_ds is None:
            return None
        return math.ceil(len(self._val_ds) / self.bs["val"])