import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_edge_encoder
from torch_geometric.graphgym.config import cfg

# -----------------------------------------------------------------------------
# Learnable Fourier Positional Encoding (generic; unchanged)
# -----------------------------------------------------------------------------
class LearnableFourierPositionalEncoding(nn.Module):
    """
    Input:  X ∈ R^{E, G, M}  (E items, G groups, M raw-coord dims per group)
    Output: Y ∈ R^{E, D}
    """
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        super().__init__()
        assert F_dim % 2 == 0, "F_dim must be even."
        assert D % G == 0, "D must be divisible by G."
        self.G, self.M, self.F_dim, self.H_dim, self.D = int(G), int(M), int(F_dim), int(H_dim), int(D)
        self.gamma = float(gamma)

        # Random Fourier features (learnable projection)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight, mean=0.0, std=self.gamma ** -2)

        # Small MLP to map Fourier features to target dim per group, then concat groups
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (E, G, M)  ->  (E, D)
        """
        E, G, M = x.shape
        proj = self.Wr(x)  # (E, G, F_dim/2)
        fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (E, G, F_dim)
        fourier = fourier / math.sqrt(self.F_dim)
        y = self.mlp(fourier)                      # (E, G, D/G)
        return y.reshape(E, self.D)                # (E, D)


# -----------------------------------------------------------------------------
# Edge LFPE Encoder
# -----------------------------------------------------------------------------
@register_edge_encoder('LFPE_Edge')
class EdgeLFPEEncoder(nn.Module):
    """
    Edge encoder that fuses Learnable Fourier Positional Encoding (LFPE) built
    from simple edge geometry:
      - Δ = pos[v] - pos[u]     (3)
      - mid = (pos[u] + pos[v]) / 2  (3)
      - len = ||Δ||2            (1)
      => edge_geom ∈ R^{7} by default (settable)

    Fusion:
      - 'add'    : edge_attr += Proj( LFPE(edge_geom) )
      - 'concat' : edge_attr  = Proj( [edge_attr || LFPE(edge_geom)] )

    YAML (example):
      dataset:
        edge_encoder: true
        edge_encoder_name: LFPE_Edge
      posenc_LFPE_Edge:
        D: 64
        F_dim: 64
        H_dim: 32
        gamma: 10.0
        coord_attr: node_coords    # or 'pos'
        use_len: true
        use_mid: true
        fuse: add                  # or 'concat'
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        p = getattr(cfg, 'posenc_LFPE_Edge', cfg)  # fallback to cfg if block missing

        # Config
        self.coord_attr = getattr(p, 'coord_attr', 'pos')     # where to read node coords from batch
        self.fuse       = getattr(p, 'fuse', 'add')            # 'add' or 'concat'
        self.use_len    = bool(getattr(p, 'use_len', True))
        self.use_mid    = bool(getattr(p, 'use_mid', True))

        # Target embedding dim for edges after encoding
        self.target_emb_dim = int(emb_dim)

        # LFPE hyperparams
        self.F_dim = int(getattr(p, 'F_dim', 64))
        self.H_dim = int(getattr(p, 'H_dim', 32))
        self.D     = int(getattr(p, 'D', emb_dim))
        self.gamma = float(getattr(p, 'gamma', 10.0))

        # Lazy modules (created once we know input dims & device)
        self.edge_proj   = None  # projects raw edge_attr -> target_emb_dim (if needed)
        self.lfpe        = None  # LearnableFourierPositionalEncoding(G=1, M=?, ...)
        self.fuse_proj   = None  # projection for PE fusion (add/concat)

    # ------------------------------ helpers ---------------------------------
    @staticmethod
    def _gather_node_coords(batch, coord_attr: str):
        coords = getattr(batch, coord_attr, None)
        if coords is None:
            coords = getattr(batch, 'node_coords', None)
        if coords is None:
            coords = getattr(batch, 'pos', None)
        return coords  # shape: (N, C)

    @staticmethod
    def _build_edge_geometry(edge_index: torch.Tensor, node_coords: torch.Tensor,
                             use_mid: bool, use_len: bool) -> torch.Tensor:
        """
        Build per-edge geometry features from endpoints.
        Returns: (E, M_geom)
        """
        u, v = edge_index  # (E,), (E,)
        Cu = node_coords[u]          # (E, C)
        Cv = node_coords[v]          # (E, C)
        delta = Cv - Cu              # (E, C)
        feats = [delta]

        if use_mid:
            mid = 0.5 * (Cu + Cv)    # (E, C)
            feats.append(mid)
        if use_len:
            length = torch.norm(delta, dim=-1, keepdim=True)  # (E, 1)
            feats.append(length)

        return torch.cat(feats, dim=-1)  # (E, M_geom)

    def _ensure_modules(self, in_edge_dim: int, M_geom: int, device: torch.device):
        # 1) project raw edge_attr to target dim if needed
        if in_edge_dim != self.target_emb_dim and self.edge_proj is None:
            self.edge_proj = nn.Linear(in_edge_dim, self.target_emb_dim).to(device)

        # 2) LFPE builder over edge-geometry of size M_geom
        if self.lfpe is None:
            self.lfpe = LearnableFourierPositionalEncoding(
                G=1, M=M_geom, F_dim=self.F_dim, H_dim=self.H_dim, D=self.D, gamma=self.gamma
            ).to(device)

        # 3) fusion projection
        if self.fuse == 'add':
            in_dim = self.D
        else:  # 'concat'
            in_dim = self.target_emb_dim + self.D

        if (self.fuse_proj is None) or (self.fuse_proj.in_features != in_dim):
            self.fuse_proj = nn.Linear(in_dim, self.target_emb_dim).to(device)

    # ------------------------------- forward --------------------------------
    def forward(self, batch):
        """
        Expects:
          batch.edge_index : (2, E)
          batch.edge_attr  : (E, d_in)  (if missing, will be created as zeros)
          batch.{pos|node_coords}: (N, C) node coordinates
        """
        edge_index = batch.edge_index
        edge_attr  = getattr(batch, 'edge_attr', None)
        if edge_attr is None:
            # if dataset had no edge features, start from zeros
            E = edge_index.size(1)
            edge_attr = torch.zeros(E, self.target_emb_dim, device=edge_index.device)
        in_edge_dim = edge_attr.size(-1)

        # Node coordinates
        node_coords = self._gather_node_coords(batch, self.coord_attr)
        if node_coords is None:
            raise ValueError("EdgeLFPEEncoder: node coordinates not found on batch "
                             f"('{self.coord_attr}', 'node_coords', or 'pos').")

        # Build per-edge geometry and ensure modules
        edge_geom = self._build_edge_geometry(edge_index, node_coords, self.use_mid, self.use_len)  # (E, M_geom)
        E, M_geom = edge_geom.size()
        self._ensure_modules(in_edge_dim=in_edge_dim, M_geom=M_geom, device=edge_attr.device)

        # 1) project raw edges if needed
        if self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)

        # 2) LFPE over edge geometry
        pe = self.lfpe(edge_geom.view(E, 1, M_geom))  # (E, D)

        # 3) fuse PE into edge features
        if self.fuse == 'add':
            edge_attr = edge_attr + self.fuse_proj(pe)
        else:  # 'concat'
            edge_attr = self.fuse_proj(torch.cat([edge_attr, pe], dim=-1))

        batch.edge_attr = edge_attr
        return batch
