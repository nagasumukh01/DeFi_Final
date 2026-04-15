"""
src/train.py
============
Training pipeline for ETGT-FRD.

Modes (controlled via --mode CLI arg):
  train    — standard training + early stopping
  tune     — Optuna hyperparameter optimization
  ablation — systematic ablation study

Features
--------
- Chronological train/val/test split (no leakage)
- Focal loss for class imbalance
- Cosine / StepLR / ReduceLROnPlateau scheduler
- Gradient clipping
- Early stopping with patience
- Automatic checkpoint saving (best val F1)
- Full metrics logging to JSON
- Optuna TPE sampling with MedianPruner
- Ablation over: wavelet, residual, num_layers, focal_loss, edge_features
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import NeighborLoader

from src.data_loader import EllipticDataLoader, load_config
from src.model import ETGT_FRD, FocalLoss
from src.utils import (
    compute_metrics,
    count_parameters,
    get_device,
    load_checkpoint,
    plot_comparison_bar,
    plot_training_curves,
    save_checkpoint,
    save_metrics,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ETGT_FRD,
    optimizer: torch.optim.Optimizer,
    data: any,            # PyG Data
    train_mask: torch.Tensor,
    device: torch.device,
    clip_norm: float = 1.0,
) -> float:
    """Train for one epoch; returns mean focal loss."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    logits, _ = model(data.x, data.edge_index, data.edge_attr)
    loss = model.compute_loss(logits, data.y, train_mask.to(device))
    loss.backward()

    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: ETGT_FRD,
    data: any,
    mask: torch.Tensor,
    device: torch.device,
    split_name: str = "val",
) -> Tuple[float, Dict]:
    """Evaluate on masked nodes; returns (loss, metrics_dict)."""
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    logits, probs = model(data.x, data.edge_index, data.edge_attr)
    loss = model.compute_loss(logits, data.y, mask)

    y_true = data.y[mask].cpu().numpy()
    y_prob = probs[mask, 1].cpu().numpy()
    y_pred = probs[mask].argmax(dim=1).cpu().numpy()

    metrics = compute_metrics(y_true, y_pred, y_prob, split=split_name)
    return loss.item(), metrics


# ---------------------------------------------------------------------------
# Full Training Loop
# ---------------------------------------------------------------------------

def train(
    cfg: dict,
    device: torch.device,
    model_override: Optional[Dict] = None,   # for Optuna
    trial: Optional[optuna.trial.Trial] = None,
) -> Dict:
    """
    Main training loop.

    Parameters
    ----------
    cfg            : config dict (may be modified by model_override)
    device         : torch device
    model_override : dict of hyperparameter overrides for Optuna
    trial          : Optuna trial object (for pruning)

    Returns
    -------
    dict with best validation and test metrics
    """
    set_seed(cfg["project"]["seed"])

    # Apply hyperparameter overrides (Optuna)
    if model_override:
        for k, v in model_override.items():
            cfg["model"][k] = v

    # Paths
    ckpt_dir   = Path(cfg["paths"]["checkpoints"])
    fig_dir    = Path(cfg["paths"]["figures"])
    result_dir = Path(cfg["paths"]["results"])
    for d in [ckpt_dir, fig_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    loader = EllipticDataLoader(cfg)
    data, splits = loader.load()
    data = data.to(device)
    train_mask = splits["train"].to(device)
    val_mask   = splits["val"].to(device)
    test_mask  = splits["test"].to(device)

    # ---- Model ----
    model = ETGT_FRD.from_config(cfg).to(device)
    logger.info("Model: %s | Params: %s", cfg["model"]["name"], f"{count_parameters(model):,}")

    # ---- Optimizer ----
    tc = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )

    # ---- Scheduler ----
    sched_name = tc["lr_scheduler"]
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc["epochs"], eta_min=1e-6
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5
        )

    # ---- Training ----
    best_val_f1    = 0.0
    best_val_metrics  = {}
    patience_counter  = 0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(1, tc["epochs"] + 1):
        train_loss = train_one_epoch(
            model, optimizer, data, train_mask, device, tc["clip_grad_norm"]
        )
        val_loss, val_metrics = evaluate(model, data, val_mask, device, "val")

        val_f1 = val_metrics["f1_illicit"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        # Scheduler step
        if sched_name == "plateau":
            scheduler.step(val_f1)
        else:
            scheduler.step()

        # Early stopping & checkpoint
        if val_f1 > best_val_f1:
            best_val_f1     = val_f1
            best_val_metrics = val_metrics
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                ckpt_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(
                "Epoch %4d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | LR=%.6f",
                epoch, train_loss, val_loss, val_f1,
                optimizer.param_groups[0]["lr"],
            )

        # Optuna pruning
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if patience_counter >= tc["patience"]:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, tc["patience"])
            break

    # ---- Test evaluation (load best checkpoint) ----
    load_checkpoint(model, ckpt_dir / "best_model.pt", device=str(device))
    _, test_metrics = evaluate(model, data, test_mask, device, "test")

    # ---- Save artefacts ----
    plot_training_curves(
        train_losses, val_losses, val_f1s,
        save_path=fig_dir / "training_curves.png",
    )
    save_metrics(test_metrics, result_dir / "test_metrics.json")
    save_metrics(best_val_metrics, result_dir / "val_metrics.json")

    logger.info(
        "Training complete | Best val F1=%.4f | Test F1=%.4f | Test AUC=%.4f",
        best_val_f1,
        test_metrics["f1_illicit"],
        test_metrics["roc_auc"],
    )
    return {"val": best_val_metrics, "test": test_metrics}


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------

def tune(cfg: dict, device: torch.device, n_trials: int = 50) -> None:
    """
    Run Optuna hyperparameter search.

    Optimises `f1_illicit` on the validation set using TPE + MedianPruner.
    """
    oc = cfg["optuna"]
    ss = oc["search_space"]

    def objective(trial: optuna.trial.Trial) -> float:
        overrides = {
            "learning_rate": trial.suggest_float(
                "learning_rate", ss["learning_rate"][0], ss["learning_rate"][1], log=True
            ),
            "hidden_dim": trial.suggest_categorical("hidden_dim", ss["hidden_dim"]),
            "num_heads": trial.suggest_categorical("num_heads", ss["num_heads"]),
            "num_layers": trial.suggest_categorical("num_layers", ss["num_layers"]),
            "dropout": trial.suggest_float("dropout", ss["dropout"][0], ss["dropout"][1]),
            "focal_gamma": trial.suggest_float(
                "focal_gamma", ss["focal_gamma"][0], ss["focal_gamma"][1]
            ),
        }
        # Shallow copy of cfg to avoid mutating original
        import copy
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["training"]["learning_rate"] = overrides.pop("learning_rate")

        try:
            result = train(cfg_copy, device, model_override=overrides, trial=trial)
            return result["val"]["f1_illicit"]
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error("Trial failed: %s", e)
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=cfg["project"]["seed"])
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        direction=oc["direction"],
        sampler=sampler,
        pruner=pruner,
        study_name=cfg["project"]["name"],
    )
    study.optimize(objective, n_trials=n_trials, timeout=oc["timeout"])

    logger.info("Best Optuna trial:")
    logger.info("  Value : %.4f", study.best_value)
    for k, v in study.best_params.items():
        logger.info("  %-20s : %s", k, v)

    # Save best params
    result_dir = Path(cfg["paths"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "optuna_best_params.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    logger.info("Saved Optuna results to %s", result_dir / "optuna_best_params.json")


# ---------------------------------------------------------------------------
# Ablation Studies
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "Full ETGT-FRD": {},   # no override
    "w/o Wavelet": {
        # Replace wavelet emb with zeros by setting embedding_dim=0 proxy:
        # (handled in data_loader — here we set wavelet levels=0 trick)
        "__wavelet_levels__": 0,
    },
    "w/o Residual": {"use_residual": False},
    "w/o Focal Loss (CE)": {"focal_gamma": 0.0, "focal_alpha": 0.5},
    "w/o Edge Features": {"__no_edge__": True},
    "3-Layer": {"num_layers": 3},
    "6-Layer": {"num_layers": 6},
}


def run_ablation(cfg: dict, device: torch.device) -> None:
    """
    Run systematic ablation over predefined config variants.
    Saves a comparison bar chart and JSON summary.
    """
    import copy

    results: Dict[str, Dict] = {}
    result_dir = Path(cfg["paths"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)

    for name, overrides in ABLATION_CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Ablation: %s", name)
        logger.info("=" * 60)

        cfg_copy = copy.deepcopy(cfg)

        # Handle special ablation flags
        if "__wavelet_levels__" in overrides:
            cfg_copy["wavelet"]["levels"] = 0
            overrides = {k: v for k, v in overrides.items() if not k.startswith("__")}

        no_edge = overrides.pop("__no_edge__", False)
        if no_edge:
            cfg_copy["model"]["edge_feature_dim"] = 0

        try:
            out = train(cfg_copy, device, model_override=overrides or None)
            results[name] = out["test"]
        except Exception as e:
            logger.error("Ablation '%s' failed: %s", name, e)
            results[name] = {"f1_illicit": 0.0, "roc_auc": 0.0}

    # Save + plot
    with open(result_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig_dir = Path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["f1_illicit", "roc_auc", "avg_precision"]:
        plot_comparison_bar(
            results, metric=metric,
            save_path=fig_dir / f"ablation_{metric}.png",
        )

    # Print table
    header = f"{'Variant':<30} {'F1-Illicit':>12} {'AUC-ROC':>10} {'Avg-Prec':>10}"
    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for name, m in results.items():
        print(
            f"{name:<30} {m.get('f1_illicit',0):>12.4f} "
            f"{m.get('roc_auc',0):>10.4f} {m.get('avg_precision',0):>10.4f}"
        )
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETGT-FRD Training")
    parser.add_argument(
        "--mode", choices=["train", "tune", "ablation"], default="train",
        help="train | tune | ablation"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    args = parser.parse_args()

    setup_logging()
    cfg    = load_config(args.config)
    device = get_device(cfg)

    logger.info("Device: %s", device)
    logger.info("Mode  : %s", args.mode)

    if args.mode == "train":
        result = train(cfg, device)
        print(f"\n✅  Training done | Test F1={result['test']['f1_illicit']:.4f}")
    elif args.mode == "tune":
        tune(cfg, device, n_trials=args.trials)
    elif args.mode == "ablation":
        run_ablation(cfg, device)
