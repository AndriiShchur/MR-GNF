# model/unified_graph_weather_model_tp.py
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Base network (keep your original import path to fit your project layout)
from model.unified_graph_weather_model import UnifiedGraphSpatioVerticalGATNet


# ----------------------- small helpers -----------------------

def _coerce_list_str(x) -> list[str]:
    if isinstance(x, (list, tuple)):
        return [str(v.decode("utf-8")) if isinstance(v, (bytes, bytearray)) else str(v) for v in x]
    if isinstance(x, str):
        return [x]
    return [str(x)]


def _find_tp_index(order: Sequence[str]) -> int:
    """Locate the precipitation channel index in `channel_order`.
    Tries candidates in priority: 'tp_log@sfc', 'tp@sfc', 'tp_log', 'tp'.
    """
    order = [str(v) for v in order]
    low   = [s.lower() for s in order]
    for cand in ("tp_log@sfc", "tp@sfc", "tp_log", "tp"):
        if cand.lower() in low:
            return low.index(cand.lower())
    raise KeyError(f"'tp' channel not found in channel_order. Example: {order[:6]} ...")


def _stats_key_for_tp(name: str) -> str:
    # In statistics we use log‑precipitation
    return "tp_log"


# ---------- conversions (normalized log ↔ physical mm) ----------
class TPDenormHelper:
    """Convert between normalized 'tp_log' and physical mm using ROI statistics.

    Expects mean/std for 'tp_log' in the same order as `stats_order`.
    """
    def __init__(self, stats_order: Sequence[str], mu: torch.Tensor, sd: torch.Tensor):
        if isinstance(mu, (list, tuple)):
            mu = torch.tensor(mu, dtype=torch.float32)
        if isinstance(sd, (list, tuple)):
            sd = torch.tensor(sd, dtype=torch.float32)

        key = _stats_key_for_tp("tp")
        idx = _coerce_list_str(stats_order).index(key)
        self.mu = float(mu[idx])
        self.sd = float(sd[idx])

    def lognorm_to_mm(self, arr_norm: torch.Tensor) -> torch.Tensor:
        # arr_norm → arr_log → mm
        arr_log = arr_norm * self.sd + self.mu
        mm = torch.expm1(arr_log)
        return torch.clamp(mm, min=0.0)

    def mm_to_lognorm(self, mm: torch.Tensor) -> torch.Tensor:
        arr_log = torch.log1p(torch.clamp(mm, min=0.0))
        return (arr_log - self.mu) / max(self.sd, 1e-6)


# ----------------------- loss tailored for precipitation -----------------------
class SparsePrecipLoss(nn.Module):
    """Composite loss for sparse precipitation targets.

    L = huber(log_norm) + α · w · |mm_pred − mm_true|

    where `w` up‑weights *wet* pixels (mm_true ≥ wet_thr_mm) and can keep lower
    weights for near‑dry pixels.

    Args
    ----
    huber_delta : δ for SmoothL1 in normalized log space
    wet_thr_mm  : threshold for "wet" mask (in millimeters)
    wet_weight  : weight multiplier for wet pixels
    dry_weight  : weight multiplier for dry pixels
    alpha_mm    : weight for the mm‑space L1 term
    gamma_focal : optional focal factor to emphasize larger mm_true
    """

    def __init__(self,
                 huber_delta: float = 0.5,
                 wet_thr_mm: float = 0.1,
                 wet_weight: float = 5.0,
                 dry_weight: float = 0.5,
                 alpha_mm: float = 0.5,
                 gamma_focal: float = 0.0):
        super().__init__()
        self.delta = float(huber_delta)
        self.wet_thr = float(wet_thr_mm)
        self.w_wet = float(wet_weight)
        self.w_dry = float(dry_weight)
        self.alpha = float(alpha_mm)
        self.gamma = float(gamma_focal)
        self.huber = nn.SmoothL1Loss(beta=self.delta, reduction="none")  # PyTorch Huber v2

    def forward(self,
                y_pred_norm: torch.Tensor,   # [B, N]  (normalized tp_log)
                y_true_norm: torch.Tensor,   # [B, N]
                denorm_helper: TPDenormHelper) -> torch.Tensor:

        # 1) Huber in normalized log space
        L_h = self.huber(y_pred_norm, y_true_norm)  # [B, N]

        if self.alpha <= 0.0:
            return L_h.mean()

        # 2) Additional term in physical mm
        mm_pred = denorm_helper.lognorm_to_mm(y_pred_norm)  # [B,N]
        mm_true = denorm_helper.lognorm_to_mm(y_true_norm)  # [B,N]
        L_mm = torch.abs(mm_pred - mm_true)

        # Weights: wet/dry
        wet_mask = (mm_true >= self.wet_thr).float()
        dry_mask = 1.0 - wet_mask
        w = self.w_wet * wet_mask + self.w_dry * dry_mask

        if self.gamma > 0.0:
            # Optional focalization to stress large values
            w = w * torch.pow(1.0 + mm_true, self.gamma)

        L = L_h + self.alpha * (w * L_mm)
        return L.mean()


# ------------------- TP adapter over the base model -------------------
class UnifiedGraphSpatioVerticalGATNetTP(nn.Module):
    """Thin adapter that returns only the precipitation channel (tp_log@sfc).

    Internally uses the full base network and slices the output, so compute cost
    is close to the base model (saves a tiny bit at the head).
    """

    def __init__(self,
                 C: int, Tin: int, Tout: int,
                 edge_index_base: torch.Tensor,
                 N_mesh: Optional[int] = None,
                 embed_dim: int = 192, blocks: int = 4, heads_v: int = 4, heads_xy: int = 4,
                 attn_drop_xy: float = 0.0,
                 channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        self.base = UnifiedGraphSpatioVerticalGATNet(
            C=C, Tin=Tin, Tout=Tout,
            edge_index_base=edge_index_base, N_mesh=N_mesh,
            embed_dim=embed_dim, blocks=blocks, heads_v=heads_v, heads_xy=heads_xy,
            attn_drop_xy=attn_drop_xy,
        )
        self.channel_order = list(channel_order) if channel_order is not None else None
        self.idx_tp = None
        if self.channel_order is not None:
            self.idx_tp = _find_tp_index(self.channel_order)

    def set_channel_order(self, order: Sequence[str]):
        self.channel_order = list(order)
        self.idx_tp = _find_tp_index(self.channel_order)

    def forward(self, x_tuple, diffusion_step: Optional[torch.Tensor] = None,
                channel_order: Optional[Sequence[str]] = None):
        """
        x_tuple: (x_all, geo, pos2d)
          x_all: [B, Tin, C, N]
          geo:   [B, 4,   N]
          pos2d: [B, 4,   N]
        return: [B, Tout, 1, N]  (normalized tp_log)
        """
        if (self.idx_tp is None) and (channel_order is not None):
            self.set_channel_order(channel_order)

        y_full = self.base(x_tuple, diffusion_step=diffusion_step)  # [B,Tout,C,N]
        if self.idx_tp is None:
            raise RuntimeError("tp index is undefined. Provide channel_order in forward() or call set_channel_order().")
        tp = y_full[:, :, self.idx_tp].unsqueeze(2)  # [B, Tout, 1, N]
        return tp


# ------------------------- Lightning for precipitation (TP) -------------------------
class PrecipTPLightning(pl.LightningModule):
    """
    One‑step precipitation head (tp_log@sfc) with a loss tailored to sparsity.
    Supports fine‑tuning from a general checkpoint via `.net.base.load_state_dict()`.
    """

    def __init__(self,
                 C: int, Tin: int, Tout: int,
                 edge_index_base: torch.Tensor,
                 N_mesh: Optional[int] = None,
                 embed_dim: int = 192, blocks: int = 4, heads_v: int = 4, heads_xy: int = 4,
                 attn_drop_xy: float = 0.0,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 # loss params
                 huber_delta: float = 0.5,
                 wet_thr_mm: float = 0.1,
                 wet_weight: float = 5.0,
                 dry_weight: float = 0.5,
                 alpha_mm: float = 0.5,
                 gamma_focal: float = 0.0,
                 diffusion_max_step: int = 0,
                 channel_order: Optional[Sequence[str]] = None,
                 stats_order: Optional[Sequence[str]] = None,
                 stats_mu: Optional[Sequence[float]] = None,
                 stats_sd: Optional[Sequence[float]] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["edge_index_base", "channel_order", "stats_order", "stats_mu", "stats_sd"])
        self.net = UnifiedGraphSpatioVerticalGATNetTP(
            C=C, Tin=Tin, Tout=Tout,
            edge_index_base=edge_index_base, N_mesh=N_mesh,
            embed_dim=embed_dim, blocks=blocks, heads_v=heads_v, heads_xy=heads_xy,
            attn_drop_xy=attn_drop_xy,
            channel_order=channel_order,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.diffusion_max_step = int(diffusion_max_step)

        self.crit = SparsePrecipLoss(
            huber_delta=huber_delta,
            wet_thr_mm=wet_thr_mm,
            wet_weight=wet_weight,
            dry_weight=dry_weight,
            alpha_mm=alpha_mm,
            gamma_focal=gamma_focal,
        )

        # ROI statistics for the mm‑space term
        if (stats_order is None) or (stats_mu is None) or (stats_sd is None):
            self.denorm_helper = None
        else:
            self.register_buffer("_stats_mu", torch.tensor(stats_mu, dtype=torch.float32))
            self.register_buffer("_stats_sd", torch.tensor(stats_sd, dtype=torch.float32))
            self.stats_order = list(stats_order)
            self.denorm_helper = TPDenormHelper(self.stats_order, self._stats_mu, self._stats_sd)

    # ----------- shared step -----------
    def _shared_step(self, batch, stage: str):
        (x_all, geo, pos2d), y_full = batch            # y_full: [B, Tout, C, N]
        B = x_all.size(0)
        # diffusion step (same convention as the general model)
        steps = torch.zeros((B,), device=x_all.device, dtype=torch.float32)
        if self.diffusion_max_step > 0:
            steps = torch.randint(0, self.diffusion_max_step, (B,), device=x_all.device, dtype=torch.float32)

        # channel order from DataModule
        order = getattr(self.trainer.datamodule, "channel_order", None)
        if order is None:
            raise RuntimeError("DataModule did not provide channel_order.")
        # prediction [B,Tout,1,N]
        y_hat = self.net((x_all, geo, pos2d), diffusion_step=steps, channel_order=order)
        y_true_tp = y_full[:, :, self.net.idx_tp].unsqueeze(2)  # [B,Tout,1,N]

        y_pred = y_hat[:, 0, 0]   # [B,N] normalized tp_log
        y_true = y_true_tp[:, 0, 0]

        if self.denorm_helper is None:
            loss = nn.SmoothL1Loss(beta=0.5)(y_pred, y_true)  # fallback if no stats
        else:
            loss = self.crit(y_pred, y_true, self.denorm_helper)

        self.log_dict({f"{stage}/loss": loss}, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # ----------- optim -----------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.trainer.max_epochs, 1))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}