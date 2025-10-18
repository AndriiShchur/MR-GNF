"""
One‑step fine‑tuning for near‑surface wind (u10, v10) from a general checkpoint.
- Selects only (u10, v10) channels via the UV head
- Supports component/magnitude/direction losses (see model file)
"""

import os
import re
import json
from pathlib import Path
import numpy as np
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.unified_graph_weather_dataloader import GraphWeatherDataModule
from model.unified_graph_weather_model_uv10 import WindUV10Lightning

# --- Repro & TF32 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SHARDS_TRAIN = "./data/shards_graph"   # graph_YYYY-MM.pt
SHARDS_VAL   = "./data/shards_graph"
STATIC_GRAPH = "./data/graph/graph_static_from_stats_vgeo_aligned.npz"

YEARS_TRAIN = range(1980, 2019)
YEARS_VAL   = range(2019, 2024)

USE_SEASONAL_MONTHS = False
SEASONAL_MONTHS = [10, 11, 12, 1, 2, 3]
ALLOWED_MONTHS = SEASONAL_MONTHS if USE_SEASONAL_MONTHS else None
FORCE_TIN_TOUT = None 

GENERAL_CKPT      = "./lightning_logs/graph_gat_unified_1step/checkpoints/graph-gat-epoch=11-val_loss=0.0789.ckpt"

OUT_DIR           = "./lightning_logs/wind_uv10_1step_ft"
MAX_EPOCHS        = 10
BATCH_SIZE_TRAIN  = 4
BATCH_SIZE_VAL    = 4
NUM_WORKERS       = 6
PIN_MEMORY        = True
PERSISTENT        = True
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME      = "graph_gat_wind_1step"

def _coerce_channel_list(names):
    if isinstance(names, (list, tuple, np.ndarray)):
        out = []
        for n in names:
            if isinstance(n, (bytes, np.bytes_)):
                n = n.decode("utf-8", "ignore")
            out.append(str(n))
        return out
    if isinstance(names, (bytes, np.bytes_)):
        names = names.decode("utf-8", "ignore")
    if isinstance(names, str):
        s = names.strip()
        # try JSON
        try:
            arr = json.loads(s)
            if isinstance(arr, (list, tuple)):
                return [str(x) for x in arr]
        except Exception:
            pass
        # fallback: comma-separated or [a,b,c]
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip(" '\"") for p in s.split(",") if p.strip(" '\"")]
        return parts if parts else [s]
    return [str(names)]

def _find_shards_by_year(root: Path, year: int):
    """
    Return shards for a specific year; supports names:
      graph_YYYY-MM.pt, graph_YYYY_MM.pt, graph_YYYYMM.pt
    """
    cand = sorted(root.glob("graph_*.pt"))
    by_year = []
    for p in cand:
        n = p.name
        # graph_YYYY-MM.pt
        m = re.match(r"^graph_(\d{4})-(\d{2})\.pt$", n)
        if m and int(m.group(1)) == year:
            by_year.append(p); continue
        # graph_YYYY_MM.pt
        m = re.match(r"^graph_(\d{4})_(\d{2})\.pt$", n)
        if m and int(m.group(1)) == year:
            by_year.append(p); continue
        # graph_YYYYMM.pt
        m = re.match(r"^graph_(\d{4})(\d{2})\.pt$", n)
        if m and int(m.group(1)) == year:
            by_year.append(p); continue
    return sorted(by_year)

def read_channel_order_from_any_shard(root: str, year: int | None = None):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root.resolve()}")

    all_shards = sorted(root.glob("graph_*.pt"))
    if not all_shards:
        raise FileNotFoundError(f"No 'graph_*.pt' found under {root.resolve()}")

    if year is not None:
        by_year = _find_shards_by_year(root, year)
        chosen = by_year[0] if by_year else None
        if chosen is None:
            print(f"[warn] No shards for year {year} in {root.resolve()}. "
                  f"Using the first available shard to read channel_names.")
            chosen = all_shards[0]
    else:
        chosen = all_shards[0]

    print(f"[info] Using shard for channel_names: {chosen.name}")
    d = torch.load(chosen, map_location="cpu")
    names_raw = d.get("channel_names", None)
    if names_raw is None:
        raise KeyError(f"'{chosen.name}' has no 'channel_names'. Keys: {list(d.keys())[:10]}...")
    return _coerce_channel_list(names_raw)

def main():

    # ---------- Data ----------
    dm = GraphWeatherDataModule(
        shards_root_train=SHARDS_TRAIN,
        shards_root_val=SHARDS_VAL,
        years_train=YEARS_TRAIN,
        years_val=YEARS_VAL,
        static_graph_npz=STATIC_GRAPH,
        allowed_months=ALLOWED_MONTHS,
        force_tin_tout=FORCE_TIN_TOUT, 
        dtype_out="float32" ,
        shuffle_within_file_train=True,
        shuffle_within_file_val=False,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT,
        device=DEVICE,
    )
    dm.setup()

    print(f"Train samples: {dm.train_samples}")
    print(f"Val   samples: {dm.val_samples}")
    print(f"Steps/epoch (train): {dm.train_steps_per_epoch}")
    print(f"N_mesh={dm.N_mesh}, C={dm.C}, Tin={dm.Tin}, Tout={dm.Tout}")
    print(f"Edge index present: {dm.edge_index_base is not None}")

    channel_order = read_channel_order_from_any_shard(SHARDS_VAL, 2023)
    print("Channels:", channel_order)
    dm.channel_order = channel_order

    # -------------------- Модель (UV) --------------------
    C, Tin, Tout, N_mesh = dm.C, dm.Tin, dm.Tout, dm.N_mesh
    if dm.edge_index_base is None:
        raise RuntimeError("edge_index_base not found. Ensure it is present in the static graph or pass manually.")
    
    lit = WindUV10Lightning(
        C=C, Tin=Tin, Tout=Tout,
        edge_index_base=dm.edge_index_base, N_mesh=N_mesh,
        embed_dim=192, blocks=4, heads_v=4, heads_xy=4, attn_drop_xy=0.0,
        lr=3e-4, weight_decay=1e-4,
        loss_components=1.0,
        loss_magnitude=0.25,
        loss_direction=0.10,
        robust_loss=False,
        chan_u_name="u10@sfc",
        chan_v_name="v10@sfc",
        channel_order=channel_order,
        diffusion_max_step=0,
    )
    
    # -------------------- init from general checkpoint --------------------
    ckpt = torch.load(GENERAL_CKPT, map_location="cpu")
    
    state = ckpt.get("state_dict", ckpt)
    
    missing, unexpected = lit.net.base.load_state_dict(
        {k.replace("net.", "").replace("base.", ""): v for k, v in state.items()},
        strict=False
    )
    
    print("Loaded from general ckpt → base net. missing:", len(missing), "unexpected:", len(unexpected))

    # try:
    #     # lit = torch.compile(lit)
    #     lit.net.base= torch.compile(lit.net.base) # за потреби вимкни, якщо на твоїй збірці є issue
    #     print("✅ torch.compile enabled")
    # except Exception as e:
    #     print(f"⚠️ torch.compile skipped: {e}")
    
    # -------------------- Trainer --------------------

    logger = CSVLogger(save_dir="./lightning_logs", name=RUN_NAME)
    
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=3,
        filename="graph-gat-{epoch:02d}-{val_loss:.4f}"
    )
    es_cb = EarlyStopping(monitor="val/loss", patience=8, mode="min")
    pbar_cb = TQDMProgressBar(refresh_rate=10)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[ckpt_cb, es_cb, pbar_cb],
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        logger=logger,
        enable_model_summary=False
        # deterministic=True,
    )
    
    trainer.fit(lit, dm)

if __name__ == "__main__":
    main()