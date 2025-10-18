# model_unified_graph_uv10.py
import math
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import pytorch_lightning as pl

# IMPORTANT: keep your original import path to match project structure
from model.unified_graph_weather_model import UnifiedGraphSpatioVerticalGATNet


# ------------------------- Vector wind loss -------------------------
class WindVectorLoss(nn.Module):
    """Loss for near‑surface wind: components plus optional magnitude/angle.

    y_pred, y_true: [B, 2, N] (u, v) in the **normalized** scale (same as training).
    Component MSE/Huber is usually sufficient; to inject a bit of "physics"
    you can also penalize magnitude and/or direction (cosine).
    """

    def __init__(self,
                 w_components: float = 1.0,
                 w_magnitude: float = 0.0,
                 w_direction: float = 0.0,
                 robust: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.wc = float(w_components)
        self.wm = float(w_magnitude)
        self.wd = float(w_direction)
        self.eps = float(eps)
        self.comp_loss = nn.SmoothL1Loss(reduction="mean") if robust else nn.MSELoss(reduction="mean")
        self.mag_loss  = nn.SmoothL1Loss(reduction="mean") if robust else nn.MSELoss(reduction="mean")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # component loss
        Lc = self.comp_loss(y_pred, y_true)

        if (self.wm == 0.0) and (self.wd == 0.0):
            return self.wc * Lc

        # magnitude
        mag_p = torch.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2 + self.eps)  # [B, N]
        mag_t = torch.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2 + self.eps)
        Lm = self.mag_loss(mag_p, mag_t)

        # direction via cosine similarity: 1 − cos(Δθ)
        dot = (y_pred[:, 0]*y_true[:, 0] + y_pred[:, 1]*y_true[:, 1])  # [B, N]
        cos_sim = dot / (mag_p * mag_t + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        Ld = (1.0 - cos_sim).mean()

        return self.wc * Lc + self.wm * Lm + self.wd * Ld


# ----------------- UV adapter over the base Graph GAT model ----------------
class UnifiedGraphSpatioVerticalGATNetUV(nn.Module):
    """Thin adapter around the general network.

    Returns only two channels (u10, v10). Internally the full backbone is used;
    savings come from a small output head, preserving checkpoint compatibility.

    Parameters
    ----------
    chan_u_name, chan_v_name : names in your shard order (e.g., "u10@sfc", "v10@sfc")
    channel_order            : list from shards to locate indices
    """

    def __init__(self,
                 C: int, Tin: int, Tout: int,
                 edge_index_base: torch.Tensor,
                 N_mesh: Optional[int] = None,
                 embed_dim: int = 192, blocks: int = 4, heads_v: int = 4, heads_xy: int = 4,
                 attn_drop_xy: float = 0.0,
                 chan_u_name: str = "u10@sfc",
                 chan_v_name: str = "v10@sfc",
                 channel_order: Optional[Sequence[str]] = None):
        super().__init__()
        self.chan_u_name = chan_u_name
        self.chan_v_name = chan_v_name
        self.channel_order = list(channel_order) if channel_order is not None else None

        # Base network as in the general model (C_out = C)
        self.base = UnifiedGraphSpatioVerticalGATNet(
            C=C, Tin=Tin, Tout=Tout,
            edge_index_base=edge_index_base, N_mesh=N_mesh,
            embed_dim=embed_dim, blocks=blocks, heads_v=heads_v, heads_xy=heads_xy,
            attn_drop_xy=attn_drop_xy,
        )

        # Determine indices if channel_order provided at init
        if self.channel_order is not None:
            self.idx_u, self.idx_v = self._find_uv_indices(self.channel_order)
        else:
            self.idx_u, self.idx_v = None, None

    @staticmethod
    def _find_uv_indices(order: Sequence[str],
                         u_candidates=("u10@sfc", "u10"),
                         v_candidates=("v10@sfc", "v10")) -> Tuple[int, int]:
        lower = [str(x).lower() for x in order]
        iu, iv = None, None
        for cand in u_candidates:
            if cand.lower() in lower:
                iu = lower.index(cand.lower()); break
        for cand in v_candidates:
            if cand.lower() in lower:
                iv = lower.index(cand.lower()); break
        if iu is None or iv is None:
            raise KeyError(f"u/v indices not found in channel_order. Example: {order[:5]} ...")
        return int(iu), int(iv)

    def set_channel_order(self, order: Sequence[str]):
        self.channel_order = list(order)
        self.idx_u, self.idx_v = self._find_uv_indices(self.channel_order)

    @torch.no_grad()
    def _lazy_indices_from_order(self, order: Sequence[str]):
        if (self.idx_u is None) or (self.idx_v is None):
            self.set_channel_order(order)

    def forward(self, x_tuple, diffusion_step: Optional[torch.Tensor] = None,
                channel_order: Optional[Sequence[str]] = None):
        """
        x_tuple: (x_all, geo, pos2d)
          x_all: [B, Tin, C, N]
          geo:   [B, 4,   N]
          pos2d: [B, 4,   N]
        return: [B, Tout, 2, N]  (u, v)
        """
        if (self.idx_u is None or self.idx_v is None) and (channel_order is not None):
            self._lazy_indices_from_order(channel_order)

        y_full = self.base(x_tuple, diffusion_step=diffusion_step)  # [B, Tout, C, N]
        if self.idx_u is None or self.idx_v is None:
            raise RuntimeError("UV indices are undefined (provide channel_order in forward() or call set_channel_order()).")
        uv = torch.stack([y_full[:, :, self.idx_u], y_full[:, :, self.idx_v]], dim=2)  # [B, Tout, 2, N]
        return uv


# ----------------------- Lightning: UV fine‑tuning ----------------------------
class WindUV10Lightning(pl.LightningModule):
    """
    Lightning module for a (u10, v10) fine‑tuning head.
      • Returns only 2‑channel output
      • Loss: components + optional magnitude/direction terms
      • Can warm‑start from a general model checkpoint
    """

    def __init__(self,
                 C: int, Tin: int, Tout: int,
                 edge_index_base: torch.Tensor,
                 N_mesh: Optional[int] = None,
                 embed_dim: int = 192, blocks: int = 4, heads_v: int = 4, heads_xy: int = 4,
                 attn_drop_xy: float = 0.0,
                 lr: float = 3e-4, weight_decay: float = 1e-4,
                 loss_components: float = 1.0,
                 loss_magnitude: float = 0.0,
                 loss_direction: float = 0.0,
                 robust_loss: bool = False,
                 chan_u_name: str = "u10@sfc",
                 chan_v_name: str = "v10@sfc",
                 channel_order: Optional[Sequence[str]] = None,
                 diffusion_max_step: int = 100):
        super().__init__()
        self.save_hyperparameters(ignore=["edge_index_base", "channel_order"])
        self.net = UnifiedGraphSpatioVerticalGATNetUV(
            C=C, Tin=Tin, Tout=Tout,
            edge_index_base=edge_index_base, N_mesh=N_mesh,
            embed_dim=embed_dim, blocks=blocks, heads_v=heads_v, heads_xy=heads_xy,
            attn_drop_xy=attn_drop_xy,
            chan_u_name=chan_u_name, chan_v_name=chan_v_name,
            channel_order=channel_order,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.diffusion_max_step = int(diffusion_max_step)
        self.criterion = WindVectorLoss(
            w_components=loss_components,
            w_magnitude=loss_magnitude,
            w_direction=loss_direction,
            robust=robust_loss,
        )

    # -------- helper: select GT (u, v) from the full tensor ----------
    def _select_uv_from_full(self, y_full: torch.Tensor, order: Sequence[str]) -> torch.Tensor:
        """y_full: [B, Tout, C, N] → [B, Tout, 2, N]"""
        self.net._lazy_indices_from_order(order)
        iu, iv = self.net.idx_u, self.net.idx_v
        return torch.stack([y_full[:, :, iu], y_full[:, :, iv]], dim=2)

    # ----------------------- train/val step -----------------------
    def _shared_step(self, batch, stage: str):
        (x_all, geo, pos2d), y_full = batch                  # y_full: [B, Tout, C, N]
        # diffusion step (as in the general model)
        steps = torch.randint(
            low=0, high=max(self.diffusion_max_step, 1),
            size=(x_all.size(0),), device=x_all.device, dtype=torch.float32
        )
        # forward → [B, Tout, 2, N]
        y_hat = self.net((x_all, geo, pos2d), diffusion_step=steps,
                         channel_order=getattr(self.trainer.datamodule, "channel_order", None))

        # GT only (u, v)
        order = getattr(self.trainer.datamodule, "channel_order", None)
        if order is None:
            raise RuntimeError("DataModule did not provide channel_order.")
        y_uv = self._select_uv_from_full(y_full, order)     # [B, Tout, 2, N]

        # lead‑1 (Tout=1)
        y_pred = y_hat[:, 0]                                # [B, 2, N]
        y_true = y_uv[:, 0]                                 # [B, 2, N]

        loss = self.criterion(y_pred, y_true)
        self.log_dict({f"{stage}/loss": loss}, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    # ----------------------- optim -----------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.trainer.max_epochs, 1))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
