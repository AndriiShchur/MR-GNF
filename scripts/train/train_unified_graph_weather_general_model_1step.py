import os, math, json, random
import numpy as np
import torch

"""
General 1‑step training script for the Unified Graph Weather GAT model.


What it does
------------
• Builds a GraphWeatherDataModule over monthly graph shards (graph_YYYY-MM.pt)
• Instantiates the UnifiedGraphWeatherGATLightning model
• Optionally compiles the model with torch.compile
• Trains with CSV logging, ModelCheckpoint, EarlyStopping, and a TQDM progress bar


Key knobs
---------
- SHARDS_TRAIN / SHARDS_VAL / STATIC_GRAPH: paths to your data
- YEARS_TRAIN / YEARS_VAL: time split
- ALLOWED_MONTHS: set to [10,11,12,1,2,3] to restrict to Oct–Mar season
- EMBED_DIM, BLOCKS, HEADS_V, HEADS_XY: architecture
- LR, WEIGHT_DECAY, MAX_EPOCHS, PRECISION: optimization


Outputs
-------
- Logs at ./lightning_logs/<RUN_NAME>/
- Checkpoints under ./lightning_logs/<RUN_NAME>/checkpoints/
"""

# --- Repro & TF32 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import pytorch_lightning as pl

from utils.unified_graph_weather_dataloader import GraphWeatherDataModule
from model.unified_graph_weather_model import UnifiedGraphWeatherGATLightning

# ==== Cell 2: Config =========================================================

SHARDS_TRAIN = "./data/shards_graph"   # graph_YYYY-MM.pt
SHARDS_VAL   = "./data/shards_graph"
STATIC_GRAPH = "./data/graph/graph_static_from_stats_vgeo_aligned.npz"

YEARS_TRAIN = range(1980, 2019)
YEARS_VAL   = range(2019, 2024)

USE_SEASONAL_MONTHS = False
SEASONAL_MONTHS = [10, 11, 12, 1, 2, 3]
ALLOWED_MONTHS = SEASONAL_MONTHS if USE_SEASONAL_MONTHS else None

DATA_DTYPE = "float32"

FORCE_TIN_TOUT = None 

BATCH_TRAIN = 2
BATCH_VAL   = 2
NUM_WORKERS = 6
PIN_MEMORY  = True
PERSISTENT  = True
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM  = 192
BLOCKS     = 4
HEADS_V    = 4
HEADS_XY   = 4
ATTN_DROP_XY = 0.0

LR            = 3e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 50
PRECISION     = "16-mixed"          # "32-true", "16-mixed", "bf16-mixed"
GRAD_CLIP_VAL = 1.0
ACCELERATOR   = "gpu" if DEVICE == "cuda" else "cpu"
DEVICES       = 1
RUN_NAME      = "graph_gat_unified_1step"

RESUME_FROM = None

dm = GraphWeatherDataModule(
    shards_root_train=SHARDS_TRAIN,
    shards_root_val=SHARDS_VAL,
    years_train=YEARS_TRAIN,
    years_val=YEARS_VAL,
    static_graph_npz=STATIC_GRAPH,
    allowed_months=ALLOWED_MONTHS,
    force_tin_tout=FORCE_TIN_TOUT,
    dtype_out=DATA_DTYPE,
    shuffle_within_file_train=True,
    shuffle_within_file_val=False,
    batch_size_train=BATCH_TRAIN,
    batch_size_val=BATCH_VAL,
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

# ==== Build model ====================================================
if dm.edge_index_base is None:
    raise RuntimeError(
        "edge_index_base not found in the static graph. "
        "Add 'edge_index_horiz_base' (Long[2, Eh]) to the NPZ or pass edges manually."
    )

lit = UnifiedGraphWeatherGATLightning(
    C=dm.C, Tin=dm.Tin, Tout=dm.Tout,
    edge_index_base=dm.edge_index_base,   # Long[2,Eh]
    N_mesh=dm.N_mesh,
    embed_dim=EMBED_DIM, blocks=BLOCKS,
    heads_v=HEADS_V, heads_xy=HEADS_XY, attn_drop_xy=ATTN_DROP_XY,
    lr=LR, weight_decay=WEIGHT_DECAY,
    loss="mse",
)

try:
    lit = torch.compile(lit)
    print("✅ torch.compile enabled")
except Exception as e:
    print(f"⚠️ torch.compile skipped: {e}")


# ==== Logger, callbacks, trainer, fit ================================
logger = CSVLogger(save_dir="./lightning_logs", name=RUN_NAME)

ckpt_cb = ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=3,
    filename="graph-gat-{epoch:02d}-{val_loss:.4f}"
)
es_cb = EarlyStopping(monitor="val_loss", patience=8, mode="min")
pbar_cb = TQDMProgressBar(refresh_rate=10)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    callbacks=[ckpt_cb, es_cb, pbar_cb],
    accelerator=ACCELERATOR, devices=DEVICES,
    precision=PRECISION,
    gradient_clip_val=GRAD_CLIP_VAL,
    logger=logger,
    enable_model_summary=False,
    # deterministic=True,
)

trainer.fit(lit, dm, ckpt_path=RESUME_FROM)