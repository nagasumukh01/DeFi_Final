"""
src/model.py
============
ETGT-FRD: Explainable Temporal Graph Transformer for Fraud Ring Detection.

Architecture Overview
---------------------
Input: Node features [raw (165) + wavelet_emb (32)] = 197-dim
       Edge features: [time_delta, same_time_flag]  = 2-dim

1. Input Projection layer  (linear + LayerNorm)
2. N × Temporal Graph Transformer (TGT) blocks:
      - Multi-head attention with edge biases (EdgeConv-style)
      - Feed-forward sub-layer (FFN)
      - Residual + LayerNorm (Pre-LN style for stability)
      - Dropout
3. Classification head → softmax over {licit, illicit}

Novelty
-------
- Edge features are injected into the attention score computation:
      score(i,j) = (Q_i · K_j) / √d_k  +  W_e · e_ij
  so the model differentiates same-step vs. cross-step money flows.
- Attention weights are stored after each layer for XAI purposes.
- Built-in `get_attention_maps()` API used by explain.py and dashboard.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary / multi-class classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    alpha : float
        Weighting factor for the rare class.
    gamma : float
        Focusing parameter. γ=0 → cross-entropy.
    reduction : str
        'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits  : (N, C)  unnormalized scores
        targets : (N,)    integer class labels
        """
        # Mask out ignored / unknown nodes
        valid = targets != self.ignore_index
        logits  = logits[valid]
        targets = targets[valid]

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t     = torch.exp(-ce_loss)                          # prob of correct class
        focal   = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Temporal Graph Transformer Layer (single block)
# ---------------------------------------------------------------------------

class TemporalGraphTransformerLayer(MessagePassing):
    """
    One Temporal Graph Transformer block with edge-feature-enhanced attention.

    The attention score between node i (query) and node j (key) is:

        a_ij = (Q_i · K_j) / √d_k  +  φ(e_ij)     (before softmax)

    where φ(e_ij) = W_edge · e_ij is a learned scalar bias from edge features.
    The aggregated context is then:

        z_i = Σ_j  softmax(a_ij) · V_j

    Followed by a 2-layer FFN with GELU activation.

    Parameters
    ----------
    in_dim      : input node feature dimension
    out_dim     : output node feature dimension (= hidden_dim for intermediate)
    edge_dim    : edge feature dimension (default 2)
    num_heads   : number of attention heads
    dropout     : dropout probability
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout   = dropout

        # Q, K, V projections (concatenated for efficiency)
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)

        # Edge-feature bias projection → one scalar per head
        self.W_edge = nn.Linear(edge_dim, num_heads, bias=True)

        # Output projection
        self.W_o = nn.Linear(out_dim, out_dim)

        # Pre-LN
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # FFN sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
        )

        # Input adapter (if dim changes)
        self.residual_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        self.drop = nn.Dropout(dropout)

        # ---- Storage for XAI ----
        self._attention_weights: Optional[Tensor] = None  # (E, num_heads)

    # ------------------------------------------------------------------
    # MessagePassing interface
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        x          : (N, in_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_dim)

        Returns
        -------
        out : (N, out_dim)
        """
        # Pre-LN for attention sub-layer
        h = self.norm1(x)
        q = self.W_q(h).view(-1, self.num_heads, self.head_dim)  # (N, H, d_k)
        k = self.W_k(h).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(h).view(-1, self.num_heads, self.head_dim)

        # Edge bias: (E, H)
        edge_bias = self.W_edge(edge_attr)  # (E, num_heads)

        # Propagate
        out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            edge_attr=edge_attr,
            edge_bias=edge_bias,
            num_nodes=x.size(0),
        )                                          # (N, H, d_k)
        out = out.view(-1, self.out_dim)           # (N, out_dim)
        out = self.drop(self.W_o(out))

        # Residual + Pre-LN for FFN
        res = self.residual_proj(x)
        out = self.norm2(out + res)
        out = out + self.drop(self.ffn(out))

        return out

    def message(
        self,
        q_i: Tensor,    # (E, H, d_k)  query from target node i
        k_j: Tensor,    # (E, H, d_k)  key   from source node j
        v_j: Tensor,    # (E, H, d_k)  value from source node j
        edge_bias: Tensor,  # (E, H)
        index: Tensor,  # target node indices for softmax normalisation
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        """Compute attention-weighted messages."""
        # Scaled dot-product per head: (E, H)
        attn_score = (q_i * k_j).sum(dim=-1) / self.scale  # (E, H)
        attn_score = attn_score + edge_bias                 # add edge bias

        # Softmax normalised per target node, per head
        attn_weight = softmax(attn_score, index, num_nodes=size_i)  # (E, H)
        attn_weight = F.dropout(attn_weight, p=self.dropout, training=self.training)

        # Store for XAI
        self._attention_weights = attn_weight.detach()

        # Weighted sum: (E, H, d_k)
        return attn_weight.unsqueeze(-1) * v_j

    def get_attention_weights(self) -> Optional[Tensor]:
        """Return stored attention weights from last forward pass."""
        return self._attention_weights


# ---------------------------------------------------------------------------
# ETGT-FRD: Full Model
# ---------------------------------------------------------------------------

class ETGT_FRD(nn.Module):
    """
    Explainable Temporal Graph Transformer for Fraud Ring Detection.

    Parameters (all read from config.yaml via `from_config`):
    -----------
    node_feature_dim  : int  — raw node feature dim (e.g. 197 after concat with wavelet)
    hidden_dim        : int  — hidden dimension throughout transformer layers
    num_heads         : int  — number of attention heads
    num_layers        : int  — number of TGT blocks (4-6 recommended)
    dropout           : float
    edge_feature_dim  : int  — edge feature dimension
    num_classes       : int  — 2 (licit / illicit)
    focal_alpha       : float
    focal_gamma       : float
    """

    def __init__(
        self,
        node_feature_dim: int = 197,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 5,
        dropout: float = 0.3,
        edge_feature_dim: int = 2,
        num_classes: int = 2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers  = num_layers
        self.use_residual = use_residual

        # ---- Input projection ----
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ---- Stack of TGT layers ----
        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TemporalGraphTransformerLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_dim=edge_feature_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ---- Loss ----
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # ---- Weight initialisation ----
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight Init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "ETGT_FRD":
        """Construct model from config.yaml dict."""
        m = cfg["model"]
        wv = cfg["wavelet"]
        node_feat_dim = m["node_feature_dim"] + wv["embedding_dim"]  # 165 + 32 = 197
        return cls(
            node_feature_dim=node_feat_dim,
            hidden_dim=m["hidden_dim"],
            num_heads=m["num_heads"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            edge_feature_dim=m["edge_feature_dim"],
            num_classes=m["num_classes"],
            focal_alpha=m["focal_alpha"],
            focal_gamma=m["focal_gamma"],
            use_residual=m["use_residual"],
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x          : (N, node_feature_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_feature_dim)
        mask       : (N,) bool mask — if provided, loss computed only on masked nodes

        Returns
        -------
        logits : (N, num_classes)
        probs  : (N, num_classes)  (softmax)
        """
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        logits = self.classifier(h)
        probs  = torch.softmax(logits, dim=-1)
        return logits, probs

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute focal loss, optionally restricting to `mask` nodes."""
        if mask is not None:
            logits  = logits[mask]
            targets = targets[mask]
        return self.criterion(logits, targets)

    # ------------------------------------------------------------------
    # XAI helpers
    # ------------------------------------------------------------------

    def get_attention_maps(self) -> List[Dict]:
        """
        Retrieve attention weights from all TGT layers after a forward pass.

        Returns
        -------
        List of dicts, one per layer:
          {
            'layer': int,
            'weights': Tensor (E, num_heads)   # per-edge, per-head attention
          }
        """
        maps = []
        for layer_idx, layer in enumerate(self.layers):
            w = layer.get_attention_weights()
            if w is not None:
                maps.append({"layer": layer_idx, "weights": w})
        return maps

    def get_node_embeddings(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """Return the final hidden-layer embeddings (before classification head)."""
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h

    def predict_node(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, node_idx: int
    ) -> Dict:
        """Predict fraud probability for a single node."""
        self.eval()
        with torch.no_grad():
            logits, probs = self.forward(x, edge_index, edge_attr)
        return {
            "node_idx": node_idx,
            "fraud_prob": probs[node_idx, 1].item(),
            "licit_prob": probs[node_idx, 0].item(),
            "predicted_class": probs[node_idx].argmax().item(),
        }

    def predict_with_uncertainty(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        num_forward_passes: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate predictions with uncertainty estimates via MC-Dropout.

        Performs multiple forward passes with dropout enabled to capture
        model uncertainty. This estimates epistemic uncertainty (model uncertainty).

        Parameters
        ----------
        x                  : (N, node_feature_dim)  node features
        edge_index         : (2, E)                 edge indices
        edge_attr          : (E, edge_feature_dim)  edge attributes
        num_forward_passes : int                    number of MC dropout samples

        Returns
        -------
        mean_probs : (N, num_classes)  mean probability across passes
        std_probs  : (N, num_classes)  standard deviation across passes
        """
        self.train()  # Enable dropout even during inference
        probabilities = []

        with torch.no_grad():
            for _ in range(num_forward_passes):
                _, probs = self.forward(x, edge_index, edge_attr)
                probabilities.append(probs.unsqueeze(0))  # (1, N, num_classes)

        self.eval()  # Reset to eval mode

        # Stack and compute statistics
        stacked = torch.cat(probabilities, dim=0)  # (num_passes, N, num_classes)
        mean_probs = stacked.mean(dim=0)           # (N, num_classes)
        std_probs = stacked.std(dim=0)             # (N, num_classes)

        return mean_probs, std_probs


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = ETGT_FRD.from_config(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅  ETGT-FRD model created — {total_params:,} parameters")

    # Dummy forward
    N, E = 100, 300
    x          = torch.randn(N, 197)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr  = torch.randn(E, 2)
    y          = torch.randint(0, 2, (N,))
    mask       = torch.ones(N, dtype=torch.bool)

    logits, probs = model(x, edge_index, edge_attr)
    loss = model.compute_loss(logits, y, mask)
    attn = model.get_attention_maps()

    print(f"   Logits  shape : {logits.shape}")
    print(f"   Probs   shape : {probs.shape}")
    print(f"   Loss          : {loss.item():.4f}")
    print(f"   Attention maps: {len(attn)} layers, each (E, H) = {attn[0]['weights'].shape}")
