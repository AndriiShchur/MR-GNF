"""
One‑step precipitation fine‑tuning over a general checkpoint (tp_log@sfc).

Guidance for the sparse‑precip loss (TP):
- wet_thr_mm:   0.1–0.2 mm is typical; increase to emphasize significant rain
- wet_weight:   4–8; larger → higher weight for wet pixels
- alpha_mm:     0.3–0.7 mixes log‑space Huber and mm‑space L1
- gamma_focal:  0–1.0 to up‑weight heavier precipitation
"""

import os, re, json, random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor

from utils.unified_graph_weather_dataloader import GraphWeatherDataModule
from model.unified_graph_weather_model_tp import PrecipTPLightning

# --- Repro & TF32 ---
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SHARDS_TRAIN = "./data/shards_graph"
SHARDS_VAL   = "./data/shards_graph"
STATIC_GRAPH = "./data/graph/graph_static_from_stats_vgeo_aligned.npz"

STATS_025_NPZ = "./data/stats/stats_025deg_1980_2013.npz"

YEARS_TRAIN = range(1980, 2019)
YEARS_VAL   = range(2019, 2024)

ALLOWED_MONTHS = None
FORCE_TIN_TOUT = None

GENERAL_CKPT = "./lightning_logs/graph_gat_unified_1step/checkpoints/graph-gat-epoch=11-val_loss=0.0789.ckpt"

OUT_DIR          = "./lightning_logs/precip_tp_1step_ft"
RUN_NAME         = "graph_gat_precip_1step"
MAX_EPOCHS       = 12
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL   = 4
NUM_WORKERS      = 6
PIN_MEMORY       = True
PERSISTENT       = True
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- helpers -----------------
def _coerce_channel_list(names):
    if isinstance(names, (list, tuple, np.ndarray)):
        out = []
        for n in names:
            if isinstance(n, (bytes, np.bytes_)): n = n.decode("utf-8", "ignore")
            out.append(str(n))
        return out
    if isinstance(names, (bytes, np.bytes_)):
        names = names.decode("utf-8", "ignore")
    if isinstance(names, str):
        s = names.strip()
        try:
            arr = json.loads(s)
            if isinstance(arr, (list, tuple)):
                return [str(x) for x in arr]
        except Exception:
            pass
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip(" '\"") for p in s.split(",") if p.strip(" '\"")]
        return parts if parts else [s]
    return [str(names)]

def _find_shards_by_year(root: Path, year: int):
    cand = sorted(root.glob("graph_*.pt"))
    by_year = []
    for p in cand:
        n = p.name
        m = re.match(r"^graph_(\d{4})-(\d{2})\.pt$", n) or re.match(r"^graph_(\d{4})_(\d{2})\.pt$", n) or re.match(r"^graph_(\d{4})(\d{2})\.pt$", n)
        if m and int(m.group(1)) == year:
            by_year.append(p)
    return sorted(by_year)

def read_channel_order_from_any_shard(root: str, year: int | None = None):
    root = Path(root); all_shards = sorted(root.glob("graph_*.pt"))
    if not all_shards:
        raise FileNotFoundError(f"Не знайдено 'graph_*.pt' у {root.resolve()}")
    chosen = None
    if year is not None:
        by_year = _find_shards_by_year(root, year)
        chosen = by_year[0] if by_year else None
    if chosen is None:
        chosen = all_shards[0]
    print(f"[info] Зчитую channel_names із: {chosen.name}")
    d = torch.load(chosen, map_location="cpu")
    names_raw = d.get("channel_names", None)
    if names_raw is None:
        raise KeyError(f"У {chosen.name} немає 'channel_names'. Ключі: {list(d.keys())[:10]}...")
    return _coerce_channel_list(names_raw)

def load_stats(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    order = list(z["order"])
    if order and isinstance(order[0], (bytes, np.bytes_)):
        order = [o.decode("utf-8") for o in order]
    mu = z["mean"].astype(np.float32, copy=False)
    sd = z["std"].astype(np.float32, copy=False); sd[sd < 1e-6] = 1e-6
    return order, mu, sd

def main():
    # ---------- DataModule ----------
    dm = GraphWeatherDataModule(
        shards_root_train=SHARDS_TRAIN,
        shards_root_val=SHARDS_VAL,
        years_train=YEARS_TRAIN,
        years_val=YEARS_VAL,
        static_graph_npz=STATIC_GRAPH,
        allowed_months=ALLOWED_MONTHS,
        force_tin_tout=FORCE_TIN_TOUT,
        dtype_out="float32",
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
    print(f"N_mesh={dm.N_mesh}, C={dm.C}, Tin={dm.Tin}, Tout={dm.Tout}")

    channel_order = read_channel_order_from_any_shard(SHARDS_VAL, 2023)
    dm.channel_order = channel_order

    stats_order, mu_vec, sd_vec = load_stats(STATS_025_NPZ)

    # ---------- Model (TP) ----------
    C, Tin, Tout, N_mesh = dm.C, dm.Tin, dm.Tout, dm.N_mesh
    lit = PrecipTPLightning(
        C=C, Tin=Tin, Tout=Tout,
        edge_index_base=dm.edge_index_base, N_mesh=N_mesh,
        embed_dim=192, blocks=4, heads_v=4, heads_xy=4, attn_drop_xy=0.0,
        lr=3e-4, weight_decay=1e-4,
        huber_delta=0.5,
        wet_thr_mm=0.1,
        wet_weight=6.0,
        dry_weight=0.5,
        alpha_mm=0.5,
        gamma_focal=0.0,
        diffusion_max_step=0,
        channel_order=channel_order,
        stats_order=stats_order, stats_mu=mu_vec, stats_sd=sd_vec,
    )

    if GENERAL_CKPT and Path(GENERAL_CKPT).exists():
        ckpt = torch.load(GENERAL_CKPT, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = lit.net.base.load_state_dict(
            {k.replace("net.", "").replace("base.", ""): v for k, v in state.items()},
            strict=False
        )
        print(f"[ckpt] init from general: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("[ckpt] general checkpoint not found — starting from scratch")

    # ---------- Trainer ----------
    logger = CSVLogger(save_dir="./lightning_logs", name=RUN_NAME)
    ckpt_cb = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3,
                              filename="graph-gat-tp-{epoch:02d}-{val_loss:.4f}")
    es_cb   = EarlyStopping(monitor="val/loss", patience=8, mode="min")
    pbar_cb = TQDMProgressBar(refresh_rate=10)
    lrmon   = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        precision="bf16-mixed" if DEVICE == "cuda" else 32,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=[ckpt_cb, es_cb, pbar_cb, lrmon],
        enable_model_summary=False,
        num_sanity_val_steps=2,
    )

    trainer.fit(lit, dm)


if __name__ == "__main__":
    main()