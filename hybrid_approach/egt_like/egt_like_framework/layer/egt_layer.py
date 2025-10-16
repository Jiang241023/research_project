import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.graphgym import register as gg_register

# ---- your official EGT building blocks (unchanged) ----
# If these are already importable from your project, you can import them instead.
class Graph(dict):
    def __dir__(self): return super().__dir__() + list(self.keys())
    def __getattr__(self, k): 
        try: return self[k]
        except KeyError: raise AttributeError('No such attribute: '+k)
    def __setattr__(self, k, v): self[k] = v
    def copy(self): return self.__class__(self)


class EGT_Layer(nn.Module):
    @staticmethod
    @torch.jit.script
    def _egt(scale_dot: bool, scale_degree: bool, num_heads: int, dot_dim: int,
             clip_logits_min: float, clip_logits_max: float, attn_dropout: float,
             attn_maskout: float, training: bool, num_vns: int,
             QKV: torch.Tensor, G: torch.Tensor, E: torch.Tensor,
             mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shp = QKV.shape
        Q, K, V = QKV.view(shp[0], shp[1], -1, num_heads).split(dot_dim, dim=2)
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E

        if mask is None:
            if attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
                gates = torch.sigmoid(G)
                A_tild = F.softmax(H_hat + rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G)
                A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            if attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
                gates = torch.sigmoid(G + mask)
                A_tild = F.softmax(H_hat + mask + rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G + mask)
                A_tild = F.softmax(H_hat + mask, dim=2) * gates

        if attn_dropout > 0:
            A_tild = F.dropout(A_tild, p=attn_dropout, training=training)

        V_att = torch.einsum('blmh,bmkh->blkh', A_tild, V)

        if scale_degree:
            degrees = torch.sum(gates, dim=2, keepdim=True)
            degree_scalers = torch.log(1 + degrees)
            degree_scalers[:, :num_vns] = 1.
            V_att = V_att * degree_scalers

        V_att = V_att.reshape(shp[0], shp[1], num_heads * dot_dim)
        return V_att, H_hat

    @staticmethod
    @torch.jit.script
    def _egt_edge(scale_dot: bool, num_heads: int, dot_dim: int,
                  clip_logits_min: float, clip_logits_max: float,
                  QK: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        shp = QK.shape
        Q, K = QK.view(shp[0], shp[1], -1, num_heads).split(dot_dim, dim=2)
        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim ** -0.5)
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E
        return H_hat

    def __init__(self,
                 node_width, edge_width, num_heads,
                 node_mha_dropout=0, edge_mha_dropout=0,
                 node_ffn_dropout=0, edge_ffn_dropout=0,
                 attn_dropout=0, attn_maskout=0,
                 activation='elu', clip_logits_value=[-5, 5],
                 node_ffn_multiplier=2.0, edge_ffn_multiplier=2.0,
                 scale_dot=True, scale_degree=False,
                 node_update=True, edge_update=True):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        self.node_mha_dropout = node_mha_dropout
        self.edge_mha_dropout = edge_mha_dropout
        self.node_ffn_dropout = node_ffn_dropout
        self.edge_ffn_dropout = edge_ffn_dropout
        self.attn_dropout = attn_dropout
        self.attn_maskout = attn_maskout
        self.activation = activation
        self.clip_logits_value = clip_logits_value
        self.node_ffn_multiplier = node_ffn_multiplier
        self.edge_ffn_multiplier = edge_ffn_multiplier
        self.scale_dot = scale_dot
        self.scale_degree = scale_degree
        self.node_update = node_update
        self.edge_update = edge_update

        assert not (self.node_width % self.num_heads)
        self.dot_dim = self.node_width // self.num_heads

        self.mha_ln_h = nn.LayerNorm(self.node_width)
        self.mha_ln_e = nn.LayerNorm(self.edge_width)
        self.lin_E = nn.Linear(self.edge_width, self.num_heads)
        if self.node_update:
            self.lin_QKV = nn.Linear(self.node_width, self.node_width * 3)
            self.lin_G = nn.Linear(self.edge_width, self.num_heads)
        else:
            self.lin_QKV = nn.Linear(self.node_width, self.node_width * 2)

        self.ffn_fn = getattr(F, self.activation)
        if self.node_update:
            self.lin_O_h = nn.Linear(self.node_width, self.node_width)
            if self.node_mha_dropout > 0:
                self.mha_drp_h = nn.Dropout(self.node_mha_dropout)

            node_inner_dim = round(self.node_width * self.node_ffn_multiplier)
            self.ffn_ln_h = nn.LayerNorm(self.node_width)
            self.lin_W_h_1 = nn.Linear(self.node_width, node_inner_dim)
            self.lin_W_h_2 = nn.Linear(node_inner_dim, self.node_width)
            if self.node_ffn_dropout > 0:
                self.ffn_drp_h = nn.Dropout(self.node_ffn_dropout)

        if self.edge_update:
            self.lin_O_e = nn.Linear(self.num_heads, self.edge_width)
            if self.edge_mha_dropout > 0:
                self.mha_drp_e = nn.Dropout(self.edge_mha_dropout)

            edge_inner_dim = round(self.edge_width * self.edge_ffn_multiplier)
            self.ffn_ln_e = nn.LayerNorm(self.edge_width)
            self.lin_W_e_1 = nn.Linear(self.edge_width, edge_inner_dim)
            self.lin_W_e_2 = nn.Linear(edge_inner_dim, self.edge_width)
            if self.edge_ffn_dropout > 0:
                self.ffn_drp_e = nn.Dropout(self.edge_ffn_dropout)

    def forward(self, g: Graph) -> Graph:
        h, e, mask = g.h, g.e, g.mask
        h_r1, e_r1 = h, e

        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)

        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)

        if self.node_update:
            G = self.lin_G(e_ln)
            V_att, H_hat = self._egt(self.scale_dot, self.scale_degree,
                                     self.num_heads, self.dot_dim,
                                     self.clip_logits_value[0], self.clip_logits_value[1],
                                     self.attn_dropout, self.attn_maskout, self.training,
                                     0 if 'num_vns' not in g else g.num_vns,
                                     QKV, G, E, mask)
            h = self.lin_O_h(V_att)
            if self.node_mha_dropout > 0:
                h = self.mha_drp_h(h)
            h.add_(h_r1)

            h_r2 = h
            h_ln = self.ffn_ln_h(h)
            h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
            if self.node_ffn_dropout > 0:
                h = self.ffn_drp_h(h)
            h.add_(h_r2)
        else:
            H_hat = self._egt_edge(self.scale_dot, self.num_heads, self.dot_dim,
                                   self.clip_logits_value[0], self.clip_logits_value[1],
                                   QKV, E)

        if self.edge_update:
            e = self.lin_O_e(H_hat)
            if self.edge_mha_dropout > 0:
                e = self.mha_drp_e(e)
            e.add_(e_r1)

            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.edge_ffn_dropout > 0:
                e = self.ffn_drp_e(e)
            e.add_(e_r2)

        out = g.copy()
        out.h, out.e = h, e
        return out


class VirtualNodes(nn.Module):
    def __init__(self, node_width, edge_width, num_virtual_nodes=1):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_virtual_nodes = num_virtual_nodes
        self.vn_node_embeddings = nn.Parameter(torch.empty(num_virtual_nodes, node_width))
        self.vn_edge_embeddings = nn.Parameter(torch.empty(num_virtual_nodes, edge_width))
        nn.init.normal_(self.vn_node_embeddings)
        nn.init.normal_(self.vn_edge_embeddings)

    def forward(self, g: Graph) -> Graph:
        h, e, mask = g.h, g.e, g.mask

        # prepend VN nodes
        node_emb = self.vn_node_embeddings.unsqueeze(0).expand(h.shape[0], -1, -1)
        h = torch.cat([node_emb, h], dim=1)

        # build VN edge blocks (top-left + first rows/cols)
        B, L, _, Ew = e.shape
        evr = self.vn_edge_embeddings.unsqueeze(1)      # (V,1,Ew)
        evc = self.vn_edge_embeddings.unsqueeze(0)      # (1,V,Ew)
        ebox = 0.5 * (evr + evc)                        # (V,V,Ew)

        evr = evr.unsqueeze(0).expand(B, -1, L, -1)     # (B,V,L,Ew)
        evc = evc.unsqueeze(0).expand(B, L, -1, -1)     # (B,L,V,Ew)
        ebox = ebox.unsqueeze(0).expand(B, -1, -1, -1)  # (B,V,V,Ew)

        e = torch.cat([evr, e], dim=1)                  # rows
        e = torch.cat([torch.cat([ebox, evc], dim=1), e], dim=2)  # cols

        out = g.copy()
        out.h, out.e = h, e
        out.num_vns = self.num_virtual_nodes

        if mask is not None:
            # pad (VNs do not add any -inf; they’re fully connectable)
            out.mask = F.pad(mask, (0, 0, self.num_virtual_nodes, 0, self.num_virtual_nodes, 0),
                             mode='constant', value=0)
        return out
# ---- end official blocks ----


@gg_register.register_layer('egt_layer')
class EGTGraphGymLayer(nn.Module):
    """
    GraphGym wrapper for the official EGT layer.
    Expects a PyG batch with:
      - batch.x:        (N, in_dim)
      - batch.edge_index (2, E)
      - batch.edge_attr or batch.edge_feat: (E, edge_dim) [optional]
      - batch.batch:    (N,) graph-id per node

    Builds dense tensors (B, L, ·), runs EGT(+optional VNs), then maps back to
    sparse (N, ·) / (E, ·) in the original order.
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=False, residual=True, cfg=None):
        super().__init__()
        # cfg here is expected to be cfg.gt (passed by your network ctor)
        self.gt = cfg

        # dims
        self.node_dim = in_dim
        self.edge_dim = getattr(self.gt, 'dim_edge', in_dim)  # fallback
        assert out_dim == in_dim, "EGT keeps hidden dim fixed (in_dim == out_dim)."

        # behavior toggles
        self.mask_non_edges = getattr(self.gt, 'mask_non_edges', True)
        self.num_vns = int(getattr(self.gt, 'num_vns', 0))
        self.scale_degree = bool(getattr(self.gt, 'scale_degree', False))
        self.node_update = bool(getattr(self.gt, 'node_update', True))
        self.edge_update = bool(getattr(self.gt, 'edge_update', True))
        self.activation = getattr(self.gt, 'activation', 'elu')
        self.attn_maskout = float(getattr(self.gt, 'attn_maskout', 0.0))
        self.clip_logits = list(getattr(self.gt, 'clip_logits', [-5.0, 5.0]))
        self.node_ffn_mult = float(getattr(self.gt, 'node_ffn_multiplier', 2.0))
        self.edge_ffn_mult = float(getattr(self.gt, 'edge_ffn_multiplier', 2.0))
        self.scale_dot = bool(getattr(self.gt, 'scale_dot', True))

        # core
        self.egt = EGT_Layer(
            node_width=self.node_dim,
            edge_width=self.edge_dim,
            num_heads=num_heads,
            node_mha_dropout=dropout,
            edge_mha_dropout=dropout,
            node_ffn_dropout=dropout,
            edge_ffn_dropout=dropout,
            attn_dropout=attn_dropout,
            attn_maskout=self.attn_maskout,
            activation=self.activation,
            clip_logits_value=self.clip_logits,
            node_ffn_multiplier=self.node_ffn_mult,
            edge_ffn_multiplier=self.edge_ffn_mult,
            scale_dot=self.scale_dot,
            scale_degree=self.scale_degree,
            node_update=self.node_update,
            edge_update=self.edge_update,
        )
        self.vn = None
        if self.num_vns > 0:
            self.vn = VirtualNodes(self.node_dim, self.edge_dim, num_virtual_nodes=self.num_vns)

        # optional extra norms to match GraphGym style flags (kept lightweight)
        self.post_ln = nn.LayerNorm(self.node_dim) if layer_norm else None
        self.post_bn = nn.BatchNorm1d(self.node_dim) if batch_norm else None
        self.use_residual = bool(residual)

    @staticmethod
    def _denseify(batch, node_dim, edge_dim, mask_non_edges: bool):
        """
        Returns:
          h_dense: (B, L, node_dim)
          e_dense: (B, L, L, edge_dim)  (zero where no edge)
          mask_logits: (B, L, L, 1) additive logits mask: 0 for valid, -1e9 for invalid
          node_mask: (B, L) bool
          aux for re-sparsify: (batch_vec, local_index, ptr)
        """
        x = batch.x                                    # (N, node_dim)
        device = x.device
        batch_vec = batch.batch                        # (N,)
        B = int(batch_vec.max().item()) + 1

        # Dense nodes
        h_dense, node_mask = to_dense_batch(x, batch_vec)       # (B, L, D), (B, L)

        # Dense edge features (zeros if missing)
        # Try edge_attr then edge_feat
        edge_attr = getattr(batch, 'edge_attr', None)
        if edge_attr is None and hasattr(batch, 'edge_feat'):
            edge_attr = batch.edge_feat
        if edge_attr is not None:
            e_dense = to_dense_adj(batch.edge_index, batch=batch_vec,
                                   max_num_nodes=h_dense.size(1),
                                   edge_attr=edge_attr)         # (B, L, L, edge_dim)
        else:
            e_dense = to_dense_adj(batch.edge_index, batch=batch_vec,
                                   max_num_nodes=h_dense.size(1))  # (B, L, L)
            e_dense = e_dense.unsqueeze(-1).expand(-1, -1, -1, edge_dim).contiguous()

        # Build attention mask (padding + optional non-edge mask)
        pad_pair = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)     # (B, L, L)
        mask = ~pad_pair
        if mask_non_edges:
            # any non-existing edge also masked out
            edge_exist = to_dense_adj(batch.edge_index, batch=batch_vec,
                                      max_num_nodes=h_dense.size(1)) > 0  # (B, L, L) bool
            mask = mask | (~edge_exist)

        # Convert to additive logits mask shape that broadcasts to heads: (B, L, L, 1)
        mask_logits = mask.unsqueeze(-1).to(h_dense.dtype) * (-1e9)

        # Aux mapping to sparse order
        num_nodes_per_g = torch.bincount(batch_vec, minlength=B)
        ptr = torch.zeros(B + 1, device=device, dtype=torch.long)
        ptr[1:] = torch.cumsum(num_nodes_per_g, dim=0)
        arange_N = torch.arange(x.size(0), device=device)
        local_index = arange_N - ptr[batch_vec]  # position 0..(n_b-1) per graph

        return h_dense, e_dense, mask_logits, node_mask, batch_vec, local_index, ptr

    @staticmethod
    def _flatten_nodes_back(h_dense, node_mask):
        # (B, L, D) -> (N, D) in original packed order using mask
        return h_dense[node_mask]

    @staticmethod
    def _gather_edges_back(e_dense, edge_index, batch_vec, local_index):
        # Vectorized gather: pick e_dense[b, ru, rv, :] for each edge (u,v)
        row, col = edge_index
        b = batch_vec[row]           # (E,)
        ru = local_index[row]        # (E,)
        rv = local_index[col]        # (E,)
        return e_dense[b, ru, rv, :] # (E, edge_dim)

    def forward(self, batch):
        # Dense-ify
        h_dense, e_dense, mask_logits, node_mask, batch_vec, local_index, _ = \
            self._denseify(batch, self.node_dim, self.edge_dim, self.mask_non_edges)

        # Build Graph object for EGT
        g = Graph(h=h_dense, e=e_dense, mask=mask_logits)

        # Optional virtual nodes inside the block
        if self.vn is not None:
            g = self.vn(g)

        # Run EGT
        g = self.egt(g)

        # If VNs were added, drop them before mapping back to sparse
        if self.vn is not None:
            V = self.num_vns
            g.h = g.h[:, V:, :]
            g.e = g.e[:, V:, V:, :]

        # Map back to sparse
        x_new = self._flatten_nodes_back(g.h, node_mask)                 # (N, D)
        if self.edge_update:
            e_new = self._gather_edges_back(g.e, batch.edge_index, batch_vec, local_index)  # (E, Ew)

        # Optional post norms / residual (node side)
        if self.use_residual and hasattr(batch, 'x') and batch.x.shape == x_new.shape:
            x_new = x_new + batch.x
        if self.post_ln is not None:
            x_new = self.post_ln(x_new)
        if self.post_bn is not None:
            x_new = self.post_bn(x_new)

        batch.x = x_new
        if self.edge_update:
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                batch.edge_attr = e_new
            else:
                batch.edge_attr = e_new  # or use edge_feat if that's your convention

        return batch
