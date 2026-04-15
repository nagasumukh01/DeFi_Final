"""
src/utils.py
============
Shared utilities: metrics, plotting, checkpoint helpers, logging setup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Configure root logger to file + console with rich formatting."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "etgt_frd.log"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt,
                        datefmt=datefmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split: str = "test",
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : (N,)  ground-truth labels  {0, 1}
    y_pred : (N,)  predicted labels
    y_prob : (N,)  probability for class 1 (illicit)
    split  : str   label for display

    Returns
    -------
    dict of metric_name → float
    """
    metrics = {}

    # Basic
    metrics["accuracy"] = float((y_true == y_pred).mean())
    metrics["f1_illicit"]  = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["f1_licit"]    = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    metrics["f1_macro"]    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # AUC (handle edge case with single class)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0

    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["avg_precision"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics["true_positives"]  = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"]  = int(tn)
    metrics["false_negatives"] = int(fn)
    metrics["precision_illicit"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["recall_illicit"]    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    logger.info("=== %s metrics ===", split.upper())
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %-25s: %.4f", k, v)
        else:
            logger.info("  %-25s: %d", k, v)

    return metrics


def save_metrics(metrics: Dict, path: str | Path) -> None:
    """Persist metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str | Path,
) -> None:
    """Save model checkpoint including optimizer state and metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info("Checkpoint saved → %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """Load model checkpoint; returns the epoch number."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    logger.info("Checkpoint loaded from %s (epoch %d)", path, epoch)
    return epoch


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}", color="#E84855")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}", color="#1A7AE8")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    layer_idx: int,
    save_path: Optional[Path] = None,
) -> None:
    """
    Visualise attention weight distribution for a given layer.

    attention_weights : (E, num_heads)
    """
    w = attention_weights.cpu().numpy()   # (E, H)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(w[:200].T, aspect="auto", cmap="hot", vmin=0, vmax=w.max())
    ax.set_xlabel("Edge index (first 200)")
    ax.set_ylabel("Attention head")
    ax.set_title(f"Layer {layer_idx} Attention Weights")
    plt.colorbar(im, ax=ax)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_f1s: list,
    save_path: Optional[Path] = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", color="#E84855")
    ax1.plot(val_losses,   label="Val Loss",   color="#1A7AE8")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Focal Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(val_f1s, label="Val F1 (Illicit)", color="#2CA02C")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 (Illicit Class)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metric: str = "f1_illicit",
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing all models on a given metric."""
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    colors = ["#95A5A6"] * (len(models) - 1) + ["#E84855"]  # highlight last (ETGT-FRD)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(cfg: dict) -> torch.device:
    pref = cfg["project"].get("device", "cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
