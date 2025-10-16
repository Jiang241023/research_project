# grit_like_framework/encoder/lfpe.py
import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.graphgym.config import cfg

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        super().__init__()
        assert F_dim % 2 == 0, "F_dim must be even."
        assert D % G == 0, "D must be divisible by G."
        self.G, self.M, self.F_dim, self.H_dim, self.D, self.gamma = G, M, F_dim, H_dim, D, float(gamma)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G),
        )
        nn.init.normal_(self.Wr.weight, mean=0.0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, G, M) -> (N, D)
        N, G, M = x.shape
        proj = self.Wr(x)                                   # (N,G,F/2)
        F = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(self.F_dim)  # (N,G,F)
        Y = self.mlp(F)                                     # (N,G,D/G)
        return Y.reshape(N, self.D)                         # (N,D)

@register_node_encoder('LFPE')
class LFPEEncoder(nn.Module):
    """
    Ensures batch.x -> emb_dim, then fuses LFPE:
      - projects raw x (N, in_dim) -> (N, emb_dim) if needed
      - builds LFPE from coords (batch.pos or batch.node_coords)
      - fuse = 'add' (default) : x += Proj(pe)
        or 'concat': x = Proj([x || pe]) -> emb_dim
    YAML:
      dataset:
        node_encoder: true
        node_encoder_name: LFPE
      posenc_LFPE:
        D: 64
        F_dim: 64
        H_dim: 32
        gamma: 10.0
        coord_attr: node_coords   # or 'pos'
        fuse: add                 # or 'concat'
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        p = cfg.posenc_LFPE
        self.coord_attr = getattr(p, 'coord_attr', 'pos')
        self.fuse = getattr(p, 'fuse', 'add')
        self._target_emb_dim = emb_dim

        self._F_dim = int(getattr(p, 'F_dim', 64))
        self._H_dim = int(getattr(p, 'H_dim', 32))
        self._D     = int(getattr(p, 'D', emb_dim))
        self._gamma = float(getattr(p, 'gamma', 10.0))

        # Lazily created once we see input dims/devices
        self._xproj: nn.Linear | None = None   # raw x -> emb_dim
        self._lfpe:  LearnableFourierPositionalEncoding | None = None
        self._peproj: nn.Linear | None = None  # pe -> emb_dim (for 'add') or [x||pe] -> emb_dim (for 'concat')

    def _ensure_modules(self, x_dim: int, M: int, device: torch.device):
        # 1) raw feature projection if needed
        if x_dim != self._target_emb_dim and self._xproj is None:
            self._xproj = nn.Linear(x_dim, self._target_emb_dim).to(device)
        # 2) LFPE builder over coords of size M
        if self._lfpe is None:
            self._lfpe = LearnableFourierPositionalEncoding(G=1, M=M, F_dim=self._F_dim,
                                                            H_dim=self._H_dim, D=self._D, gamma=self._gamma).to(device)
        # 3) projection for PE fusion
        if self.fuse == 'add':
            if self._peproj is None or self._peproj.in_features != self._D:
                self._peproj = nn.Linear(self._D, self._target_emb_dim).to(device)
        else:  # 'concat'
            in_dim = self._target_emb_dim + self._D
            if self._peproj is None or self._peproj.in_features != in_dim:
                self._peproj = nn.Linear(in_dim, self._target_emb_dim).to(device)

    def forward(self, batch):
        x = batch.x
        # get coords
        coords = getattr(batch, self.coord_attr, None)
        if coords is None:
            coords = getattr(batch, 'node_coords', None)
        # ensure modules
        x_dim = x.size(-1)
        M = int(coords.size(-1)) if coords is not None else 0
        self._ensure_modules(x_dim, M if M > 0 else 1, x.device)

        # 1) project raw features to emb_dim if needed
        if self._xproj is not None:
            x = self._xproj(x)

        # 2) add LFPE if coords exist
        if coords is not None:
            N = coords.size(0)
            pe = self._lfpe(coords.float().to(x.device).view(N, 1, -1))  # (N, D)
            if self.fuse == 'add':
                x = x + self._peproj(pe)
            else:  # concat
                x = self._peproj(torch.cat([x, pe], dim=-1))

        batch.x = x
        return batch
