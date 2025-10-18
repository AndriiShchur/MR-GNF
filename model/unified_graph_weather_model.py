# unified_graph_weather_model_gat.py
# Графова версія: Vertical (C) attention + Horizontal GAT over edges
from __future__ import annotations
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

"""
Unified Graph Weather Model (GAT-based)
--------------------------------------
This module implements a lightweight spatio‑vertical network for regional weather
forecasting on graphs. The model couples:
  1) **Vertical self‑attention** across feature channels per node; and
  2) **Horizontal graph attention** (GAT) over the base‑mesh edges.

Expected batch structure (from the dataloader):
  inputs = (x_all, geo, pos2d)
    - x_all: [B, Tin,  C, N]
    - geo  : [B, 4,    N]  (static geophysical features)
    - pos2d: [B, 4,    N]  (2D sinusoidal positional encodings)
  output = y: [B, Tout, C, N]

Where:
  C      – number of variables/channels
  Tin    – input history length
  Tout   – prediction horizon length
  N      – number of mesh nodes (base horizontal level)

The Lightning wrapper exposes a standard training interface with MSE or Huber loss.
"""

# -----------------------------
# SDPA helpers (explicitly disable Flash if not available)
# -----------------------------
def _sdpa_no_flash_ctx():
    """Return a context manager that uses math/mem‑efficient kernels, not Flash.
    Works across torch versions: tries the new API and falls back to the old one.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(backends=[SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
    except Exception:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        )


class EfficientScaledDotProductAttention(nn.Module):
    """Self‑attention along a sequence using torch SDPA.

    Input:  x ∈ [B, N, E]  (N = sequence length)
    Output: same shape, with residual + LayerNorm.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim  = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, E]
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,h,N,d]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        with _sdpa_no_flash_ctx():
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, N, self.embed_dim)                 # [B,N,E]
        out = self.out_proj(attn)
        return self.norm(out + x)


# -----------------------------
# GAT mixer (multi‑head, no external libs)
# -----------------------------
class GraphGATMixer(nn.Module):
    """
    Multi‑head Graph Attention over horizontal edges. Operates independently
    for each vertical token (channel) and each batch item.

    Input:  H ∈ [B, C, N, E]
    Output: H' of the same shape

    edge_index: Long[2, Eh] with source/target indices in [0..N‑1].
    Tip: add self‑loops (i→i) outside the model if desired; otherwise the residual
    path preserves identity information.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, attn_drop: float = 0.0, neg_slope: float = 0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.E = embed_dim
        self.H = num_heads
        self.Dh = embed_dim // num_heads
        self.attn_drop = attn_drop
        self.leaky = nn.LeakyReLU(neg_slope)

        # Linear projection to head space
        self.lin = nn.Linear(self.E, self.E, bias=False)

        # a_src / a_dst parameters for each head: [H, Dh]
        self.a_src = nn.Parameter(torch.empty(self.H, self.Dh))
        self.a_dst = nn.Parameter(torch.empty(self.H, self.Dh))

        # Output projection + normalization
        self.out_proj = nn.Linear(self.E, self.E, bias=True)
        self.norm = nn.LayerNorm(self.E)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))

    def forward(self, H: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        H: [B, C, N, E]
        edge_index: [2, Eh]
        """
        B, C, N, E = H.shape
        src = edge_index[0].to(H.device, non_blocking=True)
        dst = edge_index[1].to(H.device, non_blocking=True)
        Eh = src.numel()

        # Reshape to [B*C, N, H, Dh]
        X = H.view(B * C, N, E)
        X = self.lin(X)                                 # [BC, N, E]
        X = X.view(B * C, N, self.H, self.Dh)           # [BC, N, H, Dh]

        # Accumulator for outputs
        out = torch.zeros(B * C, N, self.H, self.Dh, device=H.device, dtype=H.dtype)

        # Prepare attention vectors
        a_src = self.a_src.to(device=H.device, dtype=H.dtype)
        a_dst = self.a_dst.to(device=H.device, dtype=H.dtype)

        for bc in range(B * C):
            Xbc = X[bc]                                 # [N, H, Dh]
            # Gather features for edges
            x_i = Xbc.index_select(0, src)              # [Eh, H, Dh] (source features)
            x_j = Xbc.index_select(0, dst)              # [Eh, H, Dh] (target features)

            # e_ij = Leaky(a_src·x_i + a_dst·x_j), per head
            e_src = torch.einsum("ehd,hd->eh", x_i, a_src)   # [Eh, H]
            e_dst = torch.einsum("ehd,hd->eh", x_j, a_dst)   # [Eh, H]
            e = self.leaky(e_src + e_dst)                    # [Eh, H]

            # Stabilize softmax
            e = torch.clamp(e, -10.0, 10.0)
            exp_e = torch.exp(e)                             # [Eh, H]
            if self.attn_drop > 0:
                exp_e = F.dropout(exp_e, p=self.attn_drop, training=self.training)

            # Sum over incoming edges of each dst (per head)
            sum_dst = torch.zeros(self.H, N, device=H.device, dtype=H.dtype)  # [H, N]
            # transpose -> [H, Eh], add along dim=1 at indices 'dst'
            sum_dst.index_add_(1, dst, exp_e.transpose(0, 1))
            # вибрати суму для кожного ребра/голови
            denom = sum_dst[:, dst].transpose(0, 1) + 1e-12                     # [Eh, H]
            alpha = exp_e / denom                                               # [Eh, H]

            # Message: alpha * x_i (classic GAT v1 style
            m = alpha.unsqueeze(-1) * x_i                                       # [Eh, H, Dh]

            # Aggregate into dst nodes per head
            for h in range(self.H):
                out[bc, :, h, :].index_add_(0, dst, m[:, h, :])                 # [N, Dh]

        # Concatenate heads → [BC, N, E]
        out = out.reshape(B * C, N, self.H * self.Dh)
        out = self.out_proj(out)                                                # [BC, N, E]
        out = self.act(out)
        # Back to [B, E, C, N]
        out = out.view(B, C, N, E).permute(0, 3, 1, 2).contiguous()             # [B, E, C, N]
        return out


# -----------------------------
# One block: Vertical SDPA + GAT + FFN
# -----------------------------
class GraphAxialGATBlock(nn.Module):
    """
    Operates in the layout [B, E, C, N].
      1) Vertical attention across channels C at each node
      2) Graph attention across horizontal edges, per channel
      3) 1×1 Conv FFN with residual + GroupNorm
    """

    def __init__(self, E: int, heads_v: int = 4, heads_xy: int = 4, attn_drop_xy: float = 0.0):
        super().__init__()
        self.attn_v = EfficientScaledDotProductAttention(E, heads_v)
        self.gat    = GraphGATMixer(E, num_heads=heads_xy, attn_drop=attn_drop_xy)
        self.ffn = nn.Sequential(
            nn.Conv2d(E, 2 * E, kernel_size=1), nn.GELU(),
            nn.Conv2d(2 * E, E, kernel_size=1),
        )
        self.norm = nn.GroupNorm(8, E)

    def forward(self, F: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # (1) Vertical attention
        B, E, C, N = F.shape
        V = F.permute(0, 3, 2, 1).reshape(B * N, C, E)    # [B*N, C, E]
        V = self.attn_v(V)                                # [B*N, C, E]
        V = V.reshape(B, N, C, E).permute(0, 3, 2, 1)     # [B, E, C, N]

        # (2) GAT mixing (per channel)
        G_in  = V.permute(0, 2, 3, 1).contiguous()        # [B, C, N, E]
        G_out = self.gat(G_in, edge_index=edge_index)     # [B, E, C, N]

        S = V + G_out                                     # residual after GAT

        # (3) FFN + residual + norm
        U = self.ffn(S)
        return self.norm(U + S)


class UnifiedGraphSpatioVerticalGATNet(nn.Module):
    """
    Core network that fuses temporal projection, static encoders, vertical SDPA,
    horizontal GAT mixing, and a 1×1 decoder to Tout.

    Inputs
    ------
    (x_all, geo, pos2d): see shapes above
    diffusion_step: optional scalar per batch item for diffusion‑style conditioning

    Output
    ------
    y ∈ [B, Tout, C, N]

    Requires a base‑level edge list `edge_index_base: Long[2, Eh]`.
    """

    def __init__(self,
                 C: int,
                 Tin: int,
                 Tout: int,
                 edge_index_base: torch.Tensor,
                 N_mesh: int,
                 static_geo_ch: int = 4,
                 static_pos2d_ch: int = 4,
                 embed_dim: int = 192,
                 blocks: int = 4,
                 heads_v: int = 4,
                 heads_xy: int = 4,
                 attn_drop_xy: float = 0.0):
        super().__init__()
        self.C = int(C)
        self.Tin = int(Tin)
        self.Tout = int(Tout)
        self.N = int(N_mesh)
        self.static_in = int(static_geo_ch + static_pos2d_ch)
        E = int(embed_dim)

        # Register base edges as buffers
        if edge_index_base.dtype != torch.long:
            edge_index_base = edge_index_base.long()
        self.register_buffer("edge_src", edge_index_base[0].contiguous())
        self.register_buffer("edge_dst", edge_index_base[1].contiguous())

        # Diffusion‑step embedding (optional)
        self.diffusion_embed = nn.Sequential(
            nn.Linear(1, E), nn.ReLU(inplace=True),
            nn.Linear(E, E)
        )

        # Temporal projection: [B, Tin, C, N] -> [B, E, C, N]
        self.temporal_proj = nn.Conv2d(self.Tin, E, kernel_size=1)

        # Static encoder: [B, static_in, 1, N] -> [B, E, 1, N] (broadcast over C)
        self.static_encoder = nn.Sequential(
            nn.Conv2d(self.static_in, 64, kernel_size=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, E, kernel_size=1),
            nn.GroupNorm(8, E), nn.ReLU(inplace=True),
        )

        # Blocks
        self.blocks = nn.ModuleList([
            GraphAxialGATBlock(E, heads_v=heads_v, heads_xy=heads_xy, attn_drop_xy=attn_drop_xy)
            for _ in range(blocks)
        ])

        # Decoder: [B, E, C, N] -> [B, Tout, C, N]
        self.decoder = nn.Sequential(
            nn.Conv2d(E, E, kernel_size=1), nn.GELU(),
            nn.Conv2d(E, self.Tout, kernel_size=1)
        )

    def forward(self,
                inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                diffusion_step: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_all, geo, pos2d = inputs
        x_all = x_all.float()  # [B, Tin, C, N]
        geo   = geo.float()    # [B, G,   N]
        pos2d = pos2d.float()  # [B, P,   N]

        B, Tin, C, N = x_all.shape
        assert Tin == self.Tin and C == self.C and N == self.N, \
            f"Shape mismatch: x_all={x_all.shape} expected Tin={self.Tin}, C={self.C}, N={self.N}"

        # (1) Temporal projection
        F = self.temporal_proj(x_all)   # [B, E, C, N]

        # (2) Diffusion step embedding (optional)
        if diffusion_step is None:
            diffusion_step = torch.zeros(B, device=x_all.device, dtype=x_all.dtype)
        d_emb = self.diffusion_embed(diffusion_step.unsqueeze(-1))          # [B, E]
        d_map = d_emb.view(B, -1, 1, 1).expand(-1, -1, C, N)                # [B, E, C, N]
        F = F + d_map

        # (3) Static features (broadcast per channel)
        static = torch.cat([geo, pos2d], dim=1)          # [B, static_in, N]
        static_e = self.static_encoder(static.unsqueeze(2))     # [B, E, 1, N]
        F = F + static_e.expand(-1, -1, C, -1)                  # [B, E, C, N]

        # (4) Axial GAT blocks
        edge_index = torch.stack([self.edge_src, self.edge_dst], dim=0)
        for blk in self.blocks:
            F = blk(F, edge_index=edge_index)

        # (5) Decode to Tout
        y = self.decoder(F)  # [B, Tout, C, N]
        return y


# -----------------------------
# Lightning wrapper
# -----------------------------
class UnifiedGraphWeatherGATLightning(pl.LightningModule):
    """
    Lightning module expecting batches as:
        ((x_all, geo, pos2d), y_all)
      where
        x_all: [B, Tin, C, N]
        y_all: [B, Tout, C, N]

    Uses MSE (default) or Huber loss; cosine annealing LR schedule.
    """

    def __init__(self,
                 C: int,
                 Tin: int,
                 Tout: int,
                 edge_index_base: torch.Tensor,
                 N_mesh: int,
                 embed_dim: int = 192,
                 blocks: int = 4,
                 heads_v: int = 4,
                 heads_xy: int = 4,
                 attn_drop_xy: float = 0.0,
                 static_geo_ch: int = 4,
                 static_pos2d_ch: int = 4,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 loss: Literal["mse", "huber"] = "mse"):
        super().__init__()
        self.save_hyperparameters(ignore=["edge_index_base"])
        self.net = UnifiedGraphSpatioVerticalGATNet(
            C=C, Tin=Tin, Tout=Tout, N_mesh=N_mesh,
            edge_index_base=edge_index_base,
            static_geo_ch=static_geo_ch, static_pos2d_ch=static_pos2d_ch,
            embed_dim=embed_dim, blocks=blocks,
            heads_v=heads_v, heads_xy=heads_xy, attn_drop_xy=attn_drop_xy,
        )
        self.criterion = nn.MSELoss() if loss == "mse" else nn.HuberLoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x_tuple, diffusion_step: Optional[torch.Tensor] = None):
        return self.net(x_tuple, diffusion_step=diffusion_step)

    def _step(self, batch, stage: str):
        (x_all, geo, pos2d), y_true = batch
        B = x_all.size(0)
        t = torch.zeros(B, device=x_all.device, dtype=x_all.dtype)  # zero diffusion‑step
        y_pred = self.forward((x_all, geo, pos2d), diffusion_step=t)
        loss = self.criterion(y_pred, y_true)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}