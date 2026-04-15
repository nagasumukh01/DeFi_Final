"""
src/baselines.py
================
Baseline models for comparison against ETGT-FRD.

Baselines implemented
---------------------
1. XGBoost         — classical tree-based, no graph structure
2. GraphSAGE       — inductive graph convolution (Hamilton et al., 2017)
3. GAT             — Graph Attention Network (Veličković et al., 2018)
4. TGAT            — Temporal Graph Attention (Xu et al., 2020) simplified

All baselines:
  - Use the same chronological train/val/test split
  - Are evaluated with the same metric suite
  - Results saved to outputs/results/baselines_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor

from src.data_loader import EllipticDataLoader, load_config
from src.utils import (
    compute_metrics,
    get_device,
    plot_comparison_bar,
    save_metrics,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. XGBoost Baseline
# ---------------------------------------------------------------------------

def run_xgboost(data: any, splits: Dict, cfg: dict) -> Dict:
    """XGBoost on raw node features (no graph structure)."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Skipping.")
        return {}

    xc = cfg["baselines"]["xgboost"]
    train_mask = splits["train"].numpy()
    val_mask   = splits["val"].numpy()
    test_mask  = splits["test"].numpy()

    X = data.x.numpy()
    y = data.y.numpy()

    X_train = X[train_mask];  y_train = y[train_mask]
    X_val   = X[val_mask];    y_val   = y[val_mask]
    X_test  = X[test_mask];   y_test  = y[test_mask]

    # Remove unknown labels (-1)
    valid_train = y_train >= 0
    X_train = X_train[valid_train]; y_train = y_train[valid_train]
    valid_val = y_val >= 0
    X_val = X_val[valid_val]; y_val = y_val[valid_val]
    valid_test = y_test >= 0
    X_test = X_test[valid_test]; y_test = y_test[valid_test]

    model = xgb.XGBClassifier(
        n_estimators=xc["n_estimators"],
        max_depth=xc["max_depth"],
        learning_rate=xc["learning_rate"],
        scale_pos_weight=xc["scale_pos_weight"],
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=cfg["project"]["seed"],
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob, split="xgboost_test")
    logger.info("XGBoost | F1=%.4f | AUC=%.4f", metrics["f1_illicit"], metrics["roc_auc"])
    return metrics


# ---------------------------------------------------------------------------
# 2. GraphSAGE Baseline
# ---------------------------------------------------------------------------

class GraphSAGEBaseline(nn.Module):
    """
    GraphSAGE with mean aggregation.
    (Hamilton et al., 2017 — Inductive Representation Learning on Large Graphs)
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        return logits, torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# 3. GAT Baseline
# ---------------------------------------------------------------------------

class GATBaseline(nn.Module):
    """
    Graph Attention Network.
    (Veličković et al., 2018 — Graph Attention Networks)
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, num_heads: int, dropout: float
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        logits = self.classifier(x)
        return logits, torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# 4. TGAT Baseline (Simplified)
# ---------------------------------------------------------------------------

class TimeEncoding(nn.Module):
    """
    Sinusoidal time encoding used in TGAT.
    (Xu et al., 2020 — Inductive Representation Learning on Temporal Graphs)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.W = nn.Linear(1, dim)

    def forward(self, t: Tensor) -> Tensor:
        t = t.float().unsqueeze(-1)              # (N, 1)
        projected = self.W(t)                    # (N, dim)
        return torch.cos(projected)              # sinusoidal encoding


class TGATBaseline(nn.Module):
    """
    Simplified TGAT using sinusoidal time encoding + GAT.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        time_enc_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.time_enc = TimeEncoding(time_enc_dim)
        combined_dim = in_dim + time_enc_dim

        self.conv1 = GATConv(combined_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, time_step: Tensor) -> Tuple[Tensor, Tensor]:
        t_enc = self.time_enc(time_step.float())        # (N, time_enc_dim)
        x = torch.cat([x, t_enc], dim=-1)              # (N, in_dim + time_enc_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        logits = self.classifier(x)
        return logits, torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Generic GNN training loop
# ---------------------------------------------------------------------------

def train_gnn_baseline(
    model: nn.Module,
    data: any,
    splits: Dict,
    cfg: dict,
    device: torch.device,
    model_name: str,
    baseline_cfg_key: str,
    uses_time: bool = False,
) -> Dict:
    """Generic training loop for GNN baselines."""
    bc = cfg["baselines"][baseline_cfg_key]
    epochs  = bc["epochs"]
    lr      = cfg["training"]["learning_rate"]
    patience = cfg["training"]["patience"]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0]).to(device))

    data = data.to(device)
    train_mask = splits["train"].to(device)
    val_mask   = splits["val"].to(device)
    test_mask  = splits["test"].to(device)

    best_val_f1 = 0.0
    best_state  = None
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if uses_time:
            logits, _ = model(data.x, data.edge_index, data.time_step)
        else:
            logits, _ = model(data.x, data.edge_index)

        # Only compute loss on labelled training nodes
        valid_train = train_mask & (data.y >= 0)
        loss = criterion(logits[valid_train], data.y[valid_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if uses_time:
                _, probs_val = model(data.x, data.edge_index, data.time_step)
            else:
                _, probs_val = model(data.x, data.edge_index)

        valid_val = val_mask & (data.y >= 0)
        y_val_true = data.y[valid_val].cpu().numpy()
        y_val_pred = probs_val[valid_val].argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val_true, y_val_pred, pos_label=1, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            logger.info("%s: Early stopping at epoch %d", model_name, epoch)
            break

        if epoch % 20 == 0:
            logger.info("%s | Epoch %d | Loss=%.4f | Val-F1=%.4f", model_name, epoch, loss.item(), val_f1)

    # Load best and test
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if uses_time:
            _, probs_test = model(data.x, data.edge_index, data.time_step)
        else:
            _, probs_test = model(data.x, data.edge_index)

    valid_test = test_mask & (data.y >= 0)
    y_test_true = data.y[valid_test].cpu().numpy()
    y_test_prob = probs_test[valid_test, 1].cpu().numpy()
    y_test_pred = probs_test[valid_test].argmax(dim=1).cpu().numpy()

    metrics = compute_metrics(y_test_true, y_test_pred, y_test_prob, split=f"{model_name}_test")
    logger.info("%s | F1=%.4f | AUC=%.4f", model_name, metrics["f1_illicit"], metrics["roc_auc"])
    return metrics


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_baselines(cfg: dict, device: torch.device) -> Dict[str, Dict]:
    """Train all baselines and return their test metrics."""
    set_seed(cfg["project"]["seed"])

    loader = EllipticDataLoader(cfg)
    data, splits = loader.load()

    in_dim = data.x.shape[1]   # 197 with wavelet

    all_metrics: Dict[str, Dict] = {}

    # 1. XGBoost
    logger.info("Running XGBoost baseline …")
    all_metrics["XGBoost"] = run_xgboost(data, splits, cfg)

    # 2. GraphSAGE
    logger.info("Running GraphSAGE baseline …")
    bc = cfg["baselines"]["graphsage"]
    sage = GraphSAGEBaseline(in_dim, bc["hidden_dim"], bc["num_layers"], bc["dropout"])
    all_metrics["GraphSAGE"] = train_gnn_baseline(
        sage, data, splits, cfg, device, "GraphSAGE", "graphsage"
    )

    # 3. GAT
    logger.info("Running GAT baseline …")
    bc = cfg["baselines"]["gat"]
    gat = GATBaseline(in_dim, bc["hidden_dim"], bc["num_heads"], bc["dropout"])
    all_metrics["GAT"] = train_gnn_baseline(
        gat, data, splits, cfg, device, "GAT", "gat"
    )

    # 4. TGAT
    logger.info("Running TGAT baseline …")
    bc = cfg["baselines"]["tgat"]
    tgat = TGATBaseline(
        in_dim, bc["hidden_dim"], bc["num_heads"], bc["time_enc_dim"], bc["dropout"]
    )
    all_metrics["TGAT"] = train_gnn_baseline(
        tgat, data, splits, cfg, device, "TGAT", "tgat", uses_time=True
    )

    # Load ETGT-FRD results if available
    etgt_path = Path(cfg["paths"]["results"]) / "test_metrics.json"
    if etgt_path.exists():
        with open(etgt_path) as f:
            all_metrics["ETGT-FRD"] = json.load(f)

    # Save combined results
    result_dir = Path(cfg["paths"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "baselines_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Plot comparison
    fig_dir = Path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["f1_illicit", "roc_auc", "avg_precision"]:
        plot_comparison_bar(
            all_metrics, metric=metric,
            save_path=fig_dir / f"comparison_{metric}.png",
        )

    # Print table
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    header = f"{'Model':<20} {'F1-Illicit':>12} {'AUC-ROC':>10} {'Avg-Prec':>12} {'Recall':>10}"
    print(header)
    print("-" * 70)
    for name, m in all_metrics.items():
        star = " ←★" if name == "ETGT-FRD" else ""
        print(
            f"{name:<20} {m.get('f1_illicit', 0):>12.4f} "
            f"{m.get('roc_auc', 0):>10.4f} {m.get('avg_precision', 0):>12.4f} "
            f"{m.get('recall_illicit', 0):>10.4f}{star}"
        )
    print("=" * 70)

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from typing import Tuple  # noqa: F401 (used in method signatures above)
    setup_logging()
    cfg    = load_config("config.yaml")
    device = get_device(cfg)
    run_all_baselines(cfg, device)
