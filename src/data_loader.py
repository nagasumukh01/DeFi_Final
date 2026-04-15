"""
src/data_loader.py
==================
Data loading, preprocessing, and temporal graph construction
for the Elliptic Bitcoin Dataset.

Key responsibilities
--------------------
1. Load the three raw CSV files (features, classes, edgelist).
2. Encode labels (illicit=1, licit=0) and discard unknowns for supervised learning.
3. Build a global PyG Data object with:
   - Node feature matrix X (165 raw features + wavelet temporal embeddings)
   - Edge index + edge features [amount_proxy, time_delta]
   - Node temporal IDs (time-step 1..49)
4. Perform chronological train / val / test splits (no leakage).
5. Persist processed data to disk for fast re-use.

Novelty
-------
Multi-scale wavelet temporal encoding using PyWavelets (pywt):
  - Each node's time-step [1..49] is turned into a 49-dim indicator,
    then decomposed with a Daubechies-4 wavelet at 3 scales.
  - The approximation + detail coefficients are concatenated & projected
    to a fixed embedding_dim (default 32) capturing low/high frequency
    temporal patterns simultaneously.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load config
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Wavelet Temporal Encoder
# ---------------------------------------------------------------------------

class WaveletTemporalEncoder:
    """
    Multi-scale wavelet temporal encoder.

    Given a time-step integer t ∈ [1, T], constructs a one-hot indicator
    vector of length T, then performs DWT decomposition and returns a
    concatenated embedding of all coefficients (approximations + details).

    Parameters
    ----------
    T : int
        Total number of time-steps (49 for Elliptic).
    wavelet_name : str
        PyWavelets wavelet name (default "db4").
    levels : int
        Number of DWT decomposition levels.
    embedding_dim : int
        Output embedding dimension after linear projection.
    mode : str
        Signal extension mode for PyWavelets.
    """

    def __init__(
        self,
        T: int = 49,
        wavelet_name: str = "db4",
        levels: int = 3,
        embedding_dim: int = 32,
        mode: str = "periodization",
    ) -> None:
        self.T = T
        self.wavelet = pywt.Wavelet(wavelet_name)
        self.levels = levels
        self.embedding_dim = embedding_dim
        self.mode = mode

        # Pre-compute wavelet basis matrix: shape (T, coeff_len)
        self._basis = self._build_basis()

        # Linear projection: coeff_len → embedding_dim
        coeff_len = self._basis.shape[1]
        rng = np.random.default_rng(42)
        self._W = rng.standard_normal((coeff_len, embedding_dim)).astype(np.float32)
        self._W /= np.sqrt(coeff_len)  # Xavier-style init

    def _decompose_one(self, signal: np.ndarray) -> np.ndarray:
        """Apply DWT and concatenate all coefficient arrays."""
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels, mode=self.mode)
        # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        return np.concatenate([c for c in coeffs]).astype(np.float32)

    def _build_basis(self) -> np.ndarray:
        """Build (T, coeff_len) basis matrix from one-hot indicators."""
        rows = []
        for t in range(self.T):
            indicator = np.zeros(self.T, dtype=np.float32)
            indicator[t] = 1.0
            rows.append(self._decompose_one(indicator))
        basis = np.stack(rows, axis=0)          # (T, coeff_len)
        return basis

    def encode(self, time_steps: np.ndarray) -> np.ndarray:
        """
        Encode a 1-D array of time-steps to embeddings.

        Parameters
        ----------
        time_steps : np.ndarray of int, shape (N,)
            Values in [1, T].

        Returns
        -------
        embeddings : np.ndarray, shape (N, embedding_dim)
        """
        # Convert 1-indexed time-steps to 0-indexed
        idx = (time_steps - 1).clip(0, self.T - 1).astype(int)
        raw = self._basis[idx]               # (N, coeff_len)
        emb = raw @ self._W                  # (N, embedding_dim)
        # L2-normalize each row for stable training
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norm


# ---------------------------------------------------------------------------
# Core Data Loader
# ---------------------------------------------------------------------------

class EllipticDataLoader:
    """
    End-to-end pipeline for loading and preprocessing the Elliptic
    Bitcoin Transaction Dataset.

    Usage
    -----
    >>> loader = EllipticDataLoader(config)
    >>> data, splits = loader.load()
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.raw_dir = Path(config["paths"]["data_raw"])
        self.proc_dir = Path(config["paths"]["data_processed"])
        self.proc_dir.mkdir(parents=True, exist_ok=True)

        ds = config["dataset"]
        self.T = ds["num_time_steps"]
        self.illicit_label = ds["illicit_label"]   # 1
        self.licit_label   = ds["licit_label"]     # 0

        # Wavelet encoder
        wv = config["wavelet"]
        self.encoder = WaveletTemporalEncoder(
            T=self.T,
            wavelet_name=wv["wavelet_name"],
            levels=wv["levels"],
            embedding_dim=wv["embedding_dim"],
            mode=wv["mode"],
        )

        # Split ratios
        self.train_ratio = ds["train_ratio"]
        self.val_ratio   = ds["val_ratio"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force_reprocess: bool = False) -> Tuple[Data, Dict]:
        """
        Load (or re-process) the dataset.

        Returns
        -------
        data   : torch_geometric.data.Data
            Full graph with all nodes (labelled + unlabelled).
        splits : dict
            {"train": mask_tensor, "val": mask_tensor, "test": mask_tensor}
            Boolean masks over nodes with known labels only.
        """
        cache_path = self.proc_dir / "elliptic_graph.pkl"

        if cache_path.exists() and not force_reprocess:
            logger.info("Loading cached processed graph from %s", cache_path)
            with open(cache_path, "rb") as f:
                bundle = pickle.load(f)
            return bundle["data"], bundle["splits"]

        logger.info("Processing raw Elliptic dataset …")
        data, splits = self._process()

        with open(cache_path, "wb") as f:
            pickle.dump({"data": data, "splits": splits}, f)
        logger.info("Saved processed graph to %s", cache_path)

        return data, splits

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _load_raw_csvs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the three CSV files."""
        ds = self.cfg["dataset"]
        feat_path  = self.raw_dir / ds["features_file"]
        class_path = self.raw_dir / ds["classes_file"]
        edge_path  = self.raw_dir / ds["edgelist_file"]

        logger.info("  → Loading features …")
        # First column is txId, second is time-step, rest are features
        feat_df = pd.read_csv(feat_path, header=None)
        feat_df.columns = (
            ["txId", "time_step"]
            + [f"local_feat_{i}" for i in range(93)]
            + [f"agg_feat_{i}"   for i in range(72)]
        )

        logger.info("  → Loading classes …")
        class_df = pd.read_csv(class_path)
        class_df.columns = ["txId", "class"]

        logger.info("  → Loading edges …")
        edge_df = pd.read_csv(edge_path)
        edge_df.columns = ["txId1", "txId2"]

        return feat_df, class_df, edge_df

    def _process(self) -> Tuple[Data, Dict]:
        feat_df, class_df, edge_df = self._load_raw_csvs()

        # ---- 1. Build node index ----------------------------------------
        all_tx_ids = feat_df["txId"].values
        tx_to_idx  = {tx: i for i, tx in enumerate(all_tx_ids)}
        N = len(all_tx_ids)

        # ---- 2. Raw features (165 = 2 + 93 + 72, minus time_step) --------
        # Drop txId; keep time_step as metadata, raw feats: columns 2..166
        raw_feats = feat_df.iloc[:, 2:].values.astype(np.float32)  # (N, 165)
        # Z-score normalisation per feature (fit only on train later → approx here)
        mu = raw_feats.mean(axis=0, keepdims=True)
        sigma = raw_feats.std(axis=0, keepdims=True) + 1e-8
        raw_feats = (raw_feats - mu) / sigma

        # ---- 3. Temporal info & wavelet encoding --------------------------
        time_steps = feat_df["time_step"].values.astype(int)   # (N,)
        wavelet_emb = self.encoder.encode(time_steps)          # (N, emb_dim)

        # Concat: [raw_feats | wavelet_emb]
        x = np.concatenate([raw_feats, wavelet_emb], axis=1).astype(np.float32)

        # ---- 4. Labels ---------------------------------------------------
        # class_df: "1" = illicit, "2" = licit, "unknown" = unknown
        class_map_df = class_df.set_index("txId")["class"]
        labels = np.full(N, -1, dtype=np.int64)   # -1 = unknown
        for tx, idx in tx_to_idx.items():
            if tx in class_map_df.index:
                c = class_map_df[tx]
                if str(c) == "1":
                    labels[idx] = self.illicit_label   # 1
                elif str(c) == "2":
                    labels[idx] = self.licit_label     # 0
                # else: unknown → stays -1

        # ---- 5. Edge index + edge features --------------------------------
        src_list, dst_list = [], []
        for _, row in edge_df.iterrows():
            s = tx_to_idx.get(row["txId1"], None)
            d = tx_to_idx.get(row["txId2"], None)
            if s is not None and d is not None:
                src_list.append(s)
                dst_list.append(d)

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )  # (2, E)

        # Edge features: [time_delta, same_time_flag]
        src_ts = time_steps[src_list]
        dst_ts = time_steps[dst_list]
        time_delta = np.abs(src_ts - dst_ts).astype(np.float32) / self.T
        same_time  = (src_ts == dst_ts).astype(np.float32)
        edge_attr  = np.stack([time_delta, same_time], axis=1).astype(np.float32)

        # ---- 6. Assemble PyG Data object ---------------------------------
        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.long),
            time_step=torch.tensor(time_steps, dtype=torch.long),
            num_nodes=N,
        )

        # Store original transaction IDs as metadata
        data.tx_ids = all_tx_ids   # numpy array, index → txId

        logger.info(
            "Graph built: %d nodes, %d edges, %d labelled nodes",
            N, edge_index.shape[1], (labels >= 0).sum(),
        )

        # ---- 7. Chronological split (no data leakage) --------------------
        splits = self._chronological_split(labels, time_steps)

        return data, splits

    def _chronological_split(
        self, labels: np.ndarray, time_steps: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Chronological train / val / test split using only labelled nodes.

        Strategy
        --------
        - Sort unique time-steps; allocate first 60% → train,
          next 20% → val, last 20% → test.
        - This prevents any future-looking bias (no data leakage).
        """
        N = len(labels)
        known_mask = (labels >= 0)               # bool array
        known_idx  = np.where(known_mask)[0]     # node indices with labels
        known_ts   = time_steps[known_idx]       # their time-steps

        unique_ts = np.sort(np.unique(known_ts))
        n_ts = len(unique_ts)
        n_train_ts = int(self.train_ratio * n_ts)
        n_val_ts   = int(self.val_ratio   * n_ts)

        train_ts_set = set(unique_ts[:n_train_ts])
        val_ts_set   = set(unique_ts[n_train_ts : n_train_ts + n_val_ts])
        test_ts_set  = set(unique_ts[n_train_ts + n_val_ts :])

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask   = torch.zeros(N, dtype=torch.bool)
        test_mask  = torch.zeros(N, dtype=torch.bool)

        for node_idx, ts in zip(known_idx, known_ts):
            if ts in train_ts_set:
                train_mask[node_idx] = True
            elif ts in val_ts_set:
                val_mask[node_idx] = True
            elif ts in test_ts_set:
                test_mask[node_idx] = True

        logger.info(
            "Split sizes → train: %d | val: %d | test: %d",
            train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item(),
        )
        return {"train": train_mask, "val": val_mask, "test": test_mask}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg  = load_config("config.yaml")
    dl   = EllipticDataLoader(cfg)
    data, splits = dl.load(force_reprocess=True)

    print("\n✅  Dataset loaded successfully!")
    print(f"   Nodes          : {data.num_nodes:,}")
    print(f"   Edges          : {data.edge_index.shape[1]:,}")
    print(f"   Node feat dim  : {data.x.shape[1]}")
    print(f"   Edge feat dim  : {data.edge_attr.shape[1]}")
    print(f"   Train labelled : {splits['train'].sum().item():,}")
    print(f"   Val   labelled : {splits['val'].sum().item():,}")
    print(f"   Test  labelled : {splits['test'].sum().item():,}")
    illicit = (data.y == 1).sum().item()
    licit   = (data.y == 0).sum().item()
    print(f"   Illicit nodes  : {illicit:,}")
    print(f"   Licit nodes    : {licit:,}")
