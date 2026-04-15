# ETGT-FRD: Explainable Temporal Graph Transformer for Fraud Ring Detection

> **Research-Grade Implementation** | Elliptic Bitcoin Dataset | PyTorch Geometric | XAI

---

## 📌 Project Overview

**ETGT-FRD** (Explainable Temporal Graph Transformer for Fraud Ring Detection) is a novel deep learning framework that combines:

- 🌊 **Wavelet-enhanced temporal encoding** for multi-scale time-step feature extraction
- 🔀 **Multi-head Temporal Graph Transformer** layers with edge features
- 🔍 **Integrated XAI pipeline** including attention maps, Captum attributions, GNNExplainer, and a novel **Fraud Ring Explainer** using Louvain community detection with natural language generation
- 📊 **Full baseline comparison** (XGBoost, GraphSAGE, GAT, TGAT)
- ⚙️ **Optuna hyperparameter optimization** and ablation studies

---

## 🗂️ Project Structure

```
Mini project/
├── data/
│   ├── raw/                        # Original Elliptic CSV files
│   │   ├── elliptic_txs_features.csv
│   │   ├── elliptic_txs_classes.csv
│   │   └── elliptic_txs_edgelist.csv
│   └── processed/                  # Preprocessed graph objects
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV → Temporal graph + wavelet encoding
│   ├── model.py                    # ETGT_FRD model class
│   ├── explain.py                  # Full XAI pipeline + Fraud Ring Explainer
│   ├── train.py                    # Training + Optuna + ablation
│   ├── baselines.py                # XGBoost, GraphSAGE, GAT, TGAT
│   └── utils.py                    # Metrics, plotting, helpers
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
├── outputs/
│   ├── checkpoints/                # Saved model weights
│   ├── figures/                    # Plots and visualizations
│   ├── results/                    # Metrics JSON files
│   └── explanations/               # Per-transaction explanations
├── logs/                           # Training logs
├── app.py                          # Streamlit dashboard
├── config.yaml                     # Central configuration
├── requirements.txt
├── research_contribution.md
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <repo_url>
cd "Mini project"
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch (with CUDA 11.8 — adjust for your CUDA version)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install PyTorch Geometric
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 5. Install all other dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Data Preprocessing
```bash
python -m src.data_loader
```
Loads the 3 CSV files, builds the temporal heterogeneous graph, applies wavelet encoding, and saves processed Data objects.

### Step 2: Train ETGT-FRD (main model)
```bash
python -m src.train --mode train
```

### Step 3: Train with Optuna hyperparameter tuning
```bash
python -m src.train --mode tune --trials 50
```

### Step 4: Run ablation studies
```bash
python -m src.train --mode ablation
```

### Step 5: Train all baselines
```bash
python -m src.baselines
```

### Step 6: Run XAI analysis on a transaction
```bash
python -m src.explain --tx_id <transaction_id>
```

### Step 7: Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Key Results (Expected)

| Model | Precision | Recall | F1 (Illicit) | AUC-ROC |
|-------|-----------|--------|--------------|---------|
| XGBoost | ~0.85 | ~0.72 | ~0.78 | ~0.96 |
| GraphSAGE | ~0.87 | ~0.74 | ~0.80 | ~0.97 |
| GAT | ~0.88 | ~0.76 | ~0.82 | ~0.97 |
| TGAT | ~0.90 | ~0.78 | ~0.84 | ~0.98 |
| **ETGT-FRD** | **~0.93** | **~0.85** | **~0.89** | **~0.99** |

---

## 📁 Dataset

**Elliptic Bitcoin Dataset** — Available on [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- 203,769 transactions, 49 time-steps
- 234,355 directed payment edges
- Labels: 2% illicit, 21% licit, 77% unknown

---

## 🧪 Research Novelties

See [`research_contribution.md`](research_contribution.md) for detailed novelty analysis.

---

## 📜 Citation

If you use this work, please cite:
```bibtex
@inproceedings{etgtfrd2025,
  title={ETGT-FRD: Explainable Temporal Graph Transformer for Fraud Ring Detection in Decentralized Finance},
  author={Aneesh et al.},
  booktitle={Proceedings of ...},
  year={2025}
}
```
