# 🚀 ETGT-FRD v2.0: Explainable Temporal Graph Transformer for Fraud Ring Detection

> **Production-Grade Fraud Detection System** | Bitcoin Elliptic Dataset | PyTorch Geometric | XAI | Blockchain Integration

![Status](https://img.shields.io/badge/Status-Production%20Ready-green?style=flat-square)
![Version](https://img.shields.io/badge/Version-2.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9%2B-purple?style=flat-square)

---

## 📌 Project Overview

**ETGT-FRD v2.0** (Explainable Temporal Graph Transformer for Fraud Ring Detection) is a production-ready deep learning system combining:

### Core Features
- 🌊 **Wavelet-enhanced temporal encoding** — Multi-scale time-step feature extraction (32-dimensional)
- 🔀 **5-layer Temporal Graph Transformer** — 8 attention heads, edge feature injection, residual connections
- 🔍 **6-Method XAI Pipeline** — Attention Maps, Captum Integrated Gradients, GraphSVX Shapley, MC-Dropout Uncertainty, Fraud Ring Detection, LLM Explanations
- 📊 **8 Interactive Visualizations** — Heatmaps, bar charts, histograms, gauges, network graphs (Plotly)
- ⛓️ **Blockchain Integration** — Real-time Blockchair API for Bitcoin transaction verification
- 💾 **CSV Export** — Downloadable prediction data for audit trails
- 🚀 **Streamlit Dashboard** — 3-page web UI (About, Historical Analysis, Real-Time Prediction)

---

## 🗂️ Project Structure

```
DeFi-MiniProject-master/
├── .streamlit/
│   └── config.toml                 # Streamlit telemetry configuration
├── data/
│   ├── raw/                        # Original Elliptic CSV files
│   │   ├── elliptic_txs_features.csv        (197 dimensions)
│   │   ├── elliptic_txs_classes.csv         (labels)
│   │   └── elliptic_txs_edgelist.csv        (234K edges)
│   └── processed/                  # Preprocessed PyTorch Geometric objects
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV → Temporal graph + wavelet (32-dim)
│   ├── model.py                    # ETGT_FRD (5-layer TGT, 8 heads)
│   ├── explain.py                  # 6-method XAI pipeline (FIXED duplicate class)
│   ├── blockchain.py              # ⭐ NEW: Blockchair API + fraud verification
│   ├── train.py                    # Training + validation
│   ├── baselines.py                # XGBoost, GraphSAGE, GAT, TGAT
│   └── utils.py                    # Metrics, plotting, helpers
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
├── outputs/
│   ├── checkpoints/                # Saved model (best_model.pt)
│   ├── figures/                    # Visualizations
│   ├── results/                    # Metrics (baselines_results.json)
│   └── explanations/               # Per-transaction XAI outputs
├── scripts/
│   ├── validate_environment.py
│   ├── validate_model.py
│   ├── benchmark_performance.py
│   └── deployment_checklist.py
├── docs/
│   ├── API.md                      # API documentation
│   └── ARCHITECTURE.md             # System architecture
├── app.py                          # ⭐ Streamlit dashboard (3 pages, CSV export)
├── run_app.py                      # Helper script (telemetry disabled)
├── config.yaml                     # Central hyperparameter config
├── requirements.txt                # All dependencies (web3 added)
├── verify_requirements.py          # ⭐ Verification script (all 5 reqs passing)
├── PROJECT_LAUNCH_REPORT.md        # ⭐ Comprehensive documentation
├── QUICK_START.md
├── START_HERE.md
└── README.md
```

---

## ⚡ Quick Start (30 seconds)

```bash
# 1. Clone & setup
git clone https://github.com/nagasumukh01/DeFi_Final.git
cd DeFi-MiniProject-master
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py
# Opens: http://localhost:8501
```

---

## 🔧 Installation (Detailed)

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/nagasumukh01/DeFi_Final.git
cd DeFi-MiniProject-master
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install PyTorch
```bash
# CUDA 11.8 (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio
```

### Step 4: Install PyTorch Geometric
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
```

### Step 5: Install All Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python verify_requirements.py
# Expected output: ✓ All 5 requirements verified
```

---

## 🚀 Using the Application

### 1. Launch Streamlit Dashboard
```bash
# With telemetry disabled (recommended)
streamlit run app.py --logger.level=error --client.showErrorDetails=false

# Or use helper script
python run_app.py
```

**Access**: http://localhost:8501

### 2. Dashboard Features

#### 📊 **Page 1: About**
- Feature showcase (6 XAI methods)
- Model architecture overview
- Dataset statistics
- Technology stack

#### 📈 **Page 2: Historical Analysis** (Full XAI)
- Select any transaction (0-203,768)
- **7 Sections with Visualizations**:
  1. Prediction metrics (fraud probability, confidence)
  2. Attention heatmap (6 layers × 8 heads)
  3. Captum feature importance (top 15)
  4. GraphSVX Shapley values (top 10)
  5. Fraud ring community detection
  6. Blockchain verification (Blockchair API)
  7. Risk alert & LLM explanation
- Progress bars (6 steps)
- Computation time: 15-20s (first run), 5-6s (cached)

#### ⚡ **Page 3: Real-Time Prediction** (NEW CSV Export)
- **3 Input Modes**:
  - 📊 Load sample from dataset
  - 🔄 Random synthetic transaction
  - 🎲 Custom feature intensity
- **Results with Visualizations**:
  - Risk gauge (0-100 scale)
  - Confidence vs uncertainty bar
  - Risk assessment (CRITICAL/MODERATE/LOW)
- **⭐ CSV Export** NEW:
  - Download prediction data as CSV
  - Includes: Timestamp, predictions, all 197 features
  - Filename: `prediction_YYYYMMDD_HHMMSS.csv`
  - Preview data in expandable section

### 3. Verification Script
```bash
python verify_requirements.py
```

Output shows:
- ✓ XAI Module (6 methods)
- ✓ Blockchain Module (Blockchair API)
- ✓ Model Architecture (5-layer TGT)
- ✓ Visualizations (8 types)
- ✓ Performance (caching, progress bars)

---

## 📊 Performance Metrics

### Model Accuracy (ETGT-FRD)
| Metric | Value |
|--------|-------|
| **Precision** | 93% |
| **Recall** | 85% |
| **F1-Score** | 89% |
| **AUC-ROC** | 0.99 |
| **Inference Time** | 2-3s per transaction |
| **Batch Size** | 1 (single transaction) |

### Performance Optimization
- 🚀 **Cold Start**: 15-20s (model loading + caching)
- ⚡ **Cached Runs**: 5-6s (model in memory)
- 💾 **Memory**: ~2.5 GB (model + data)
- 📈 **Throughput**: 20-30 transactions/minute

---

## ⛓️ Blockchain Integration (NEW)

### Features
- 🔗 **Blockchair API** — Real-time Bitcoin transaction data
- 🔍 **On-Chain Fraud Verification** — Mixing signals, input/output analysis
- 💡 **Smart Contract Ready** — Ethereum integration framework
- 📊 **Cross-Chain Indicators** — Fraud ring pattern detection

### Usage
```python
from src.blockchain import BlockchainDataProvider, BlockchainFraudVerifier

# Fetch transaction from blockchain
provider = BlockchainDataProvider()
tx_data = provider.fetch_transaction("txid_here")

# Verify fraud patterns on-chain
verifier = BlockchainFraudVerifier()
fraud_score = verifier.compute_fraud_score(tx_data)
```

---

## 🔍 XAI Pipeline (6 Methods)

### 1. **Attention Maps** (Visualizer)
- Extracts per-head attention from 5 TGT layers
- Heatmap visualization (Viridis colormap)
- Shows model focus on important nodes

### 2. **Captum Integrated Gradients**
- Feature-level attribution
- Top 15 features ranked by importance
- Green (fraud) / Red (licit) color coding

### 3. **GraphSVX Shapley Values**
- Game-theoretic coalition analysis
- Top 10 Shapley values
- Explains feature interactions

### 4. **MC-Dropout Uncertainty**
- Bayesian approximation (10 forward passes)
- Confidence estimation
- Uncertainty quantification

### 5. **Fraud Ring Detection**
- Louvain community detection
- Subgraph analysis of suspicious transactions
- Ring size and structure analysis

### 6. **LLM Explanations**
- Natural language summaries
- Risk factors in plain English
- Actionable recommendations

---

## 📦 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Deep Learning** | PyTorch | 2.11.0 |
| **Graph NN** | PyTorch Geometric | 2.7.0 |
| **Visualization** | Plotly | 6.7.0 |
| **Web Framework** | Streamlit | 1.56.0 |
| **Feature Attribution** | Captum | 0.7.0 |
| **Blockchain** | Web3.py | 6.11.0 |
| **Community Detection** | NetworkX | 3.6.1 |
| **Time Series** | PyWavelets | 1.5.0 |

---

## ✅ Requirements Verification

Run to verify all 5 major requirements are met:

```bash
python verify_requirements.py
```

**Output breakdown**:
```
✓ XAI Module: 6 methods (Attention, Captum, GraphSVX, MC-Dropout, Rings, LLM)
✓ Blockchain Module: Blockchair API + fraud verification
✓ Transformer Model: 5-layer ETGT with 8 attention heads
✓ Visualizations: 8 interactive chart types (heatmaps, bars, gauges, etc.)
✓ Performance: Caching, progress bars, 5-6s cached inference
```

---

## 🐛 Bug Fixes (v2.0)

- ✅ **Fixed**: Duplicate `AttentionVisualizer` class definition (was causing AttributeError)
- ✅ **Added**: CSV export for prediction audit trails
- ✅ **Disabled**: Streamlit telemetry (no more webhook errors)
- ✅ **Optimized**: Caching strategy (@st.cache_resource)
- ✅ **Enhanced**: Progress indicators (6-step pipeline)

---

## 📌 Dataset Information

**Elliptic Bitcoin Dataset**
- **Transactions**: 203,769
- **Time Steps**: 49
- **Edges**: 234,355 (payment flows)
- **Features**: 165 original + 32 wavelet encoded = **197 total**
- **Labels**: 2% illicit, 21% licit, 77% unknown
- **Source**: [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

---

## 📊 Key Results (Baseline Comparison)

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|-----|---------|
| XGBoost | 85% | 72% | 78% | 0.96 |
| GraphSAGE | 87% | 74% | 80% | 0.97 |
| GAT | 88% | 76% | 82% | 0.97 |
| TGAT | 90% | 78% | 84% | 0.98 |
| **ETGT-FRD v2.0** | **93%** | **85%** | **89%** | **0.99** |
