# ETGT-FRD v2.0 Architecture Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ETGT-FRD v2.0 System                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐         ┌─────────────┐
   │ Streamlit│         │ FastAPI  │        │  Raw Data   │
   │Dashboard │         │ Backend  │        │  (3 CSVs)   │
   └─────────┘         └──────────┘         └─────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ ETGT_FRD Model   │
                    │ (Transformer)    │
                    └──────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
   │ Predictions │   │ Uncertainty  │   │ Attention Maps   │
   │ (logits)    │   │ (MC-Dropout) │   │ (TGT layers)     │
   └─────────────┘   └──────────────┘   └──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  XAI Pipeline      │
                    │  (6 methods)       │
                    └────────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┬──────────────┐
        │                     │                      │              │
        ▼                     ▼                      ▼              ▼
  ┌──────────┐        ┌─────────────┐      ┌──────────────┐  ┌──────────┐
  │GraphSVX  │        │Captum       │      │AttentionViz  │  │FraudRing │
  │Shapley   │        │Integrated   │      │by layer      │  │Detection │
  │Values    │        │Gradients    │      │              │  │(Louvain) │
  └──────────┘        └─────────────┘      └──────────────┘  └──────────┘
```

---

## Core Components

### 1. **Data Layer** (`src/data_loader.py`)

**Purpose**: Load and preprocess Elliptic dataset

**Key Classes**:
- `EllipticDataLoader`: Loads 3 CSV files (features, edgelist, classes)
- Returns: Graph with 203,769 nodes, ~5M edges, 165-dim features

**Flow**:
```
CSV Files → Load → Parse → Normalize → Wavelet Encoding → PyG Data Object
```

**Output**:
- `x`: (203769, 197) tensor (165 raw features + 32 wavelet embeddings)
- `edge_index`: (2, ~5M) edge list
- `edge_attr`: (~5M, 2) edge attributes (time_delta, same_time)
- `y`: (203769,) binary labels (0=licit, 1=fraud)

---

### 2. **Model Layer** (`src/model.py`)

**Architecture**: Temporal Graph Transformer (TGT)

**Key Components**:
```python
ETGT_FRD(
  input_features=197,           # 165 raw + 32 wavelet
  embedding_dim=128,            # Head embedding dimension
  num_heads=8,                  # Multi-head attention
  num_stacks=2,                 # TGT blocks depth
  num_layers_per_stack=2,       # Layers per block
  dropout=0.5,                  # MC-Dropout for uncertainty
  temporal_encoding='wavelet'   # Temporal patterns
)
```

**Forward Pass**:
1. **Input**: (N, 197) features + temporal graph
2. **Embedding**: Project to (N, embedding_dim)
3. **TGT Layers**: Apply 2 stacks × 2 layers temporal attention
4. **Attention**: Multi-head attention over temporal neighbors
5. **Output**: (N, 2) logits → softmax → (N, 2) probabilities

**Key Methods**:
- `forward(x, edge_index, edge_attr)`: Main inference
- `predict_with_uncertainty(x, edge_index, edge_attr, num_forward_passes)`: MC-Dropout uncertainty

**Model Size**: ~50MB (trained weights)

---

### 3. **XAI Layer** (`src/explain.py`)

**6 Complementary Explanation Methods**:

#### a) **AttentionVisualizer**
- Extracts attention maps from TGT layers
- Identifies most important edges
- **Latency**: 10ms (zero-cost after inference)

#### b) **GraphSVXExplainer**
- Coalition Sampling Shapley Values
- Estimates feature importance
- **Parameters**: 5 coalitions (vs. standard 20)
- **Latency**: 300-400ms

#### c) **CaptumExplainer**
- Integrated Gradients attribution
- Gradient-based feature importance
- **Latency**: 200-300ms

#### d) **GNNExplainerWrapper**
- PyTorch Geometric GNNExplainer
- Learns subgraph and feature masks
- **Latency**: 500-1000ms

#### e) **FraudRingExplainer**
- Louvain community detection
- Identifies fraud rings/clusters
- LLM-based explanation generation
- **Latency**: 100-200ms

#### f) **XAIPipeline**
- Orchestrates all 6 methods
- Returns unified explanation dict
- **Total Latency**: 2-3 seconds
- **Output**: Comprehensive explanation object

---

### 4. **Dashboard Layer** (`app.py`)

**Framework**: Streamlit

**Modes**:

#### Historical Analysis
```
Select Node → Load Data → Run Inference → Compute XAI 
  → Visualize Predictions, Features, Rings, LLM explanation
```

#### Real-Time Prediction
```
Input Features (sliders) → Generate tensor → Inference
  → Display metrics & gauges
```

**Key Visualizations**:
- Risk Gauge (Plotly)
- Feature Importance Bar Chart
- Confidence/Uncertainty Bars
- Fraud Ring Statistics
- LLM Narrative Explanation

---

### 5. **API Layer** (`app/api.py`)

**Framework**: FastAPI

**Endpoints**:
- `POST /predict`: Single transaction
- `POST /batch-predict`: Multiple transactions
- `GET /model-info`: Model metadata
- `GET /health`: Health check

**Response Schema**:
```python
{
  "fraud_probability": float,
  "confidence": float,
  "uncertainty": float,
  "xai_explanation": {
    "graphsvx_importance": {...},
    "captum_importance": [...],
    "attention_maps": {...},
    "fraud_ring_analysis": {...},
    "llm_explanation": str
  }
}
```

---

## Data Flow: Complete Example

### Scenario: Predict fraud for transaction #1000 (real-time)

```
1. USER INPUT (Real-Time Prediction Mode)
   └─ Feature Intensity: 0.5
      Randomization: OFF
   
2. FEATURE GENERATION
   └─ Create (1, 197) tensor with scaled features
   └─ NO edges (isolated new node)
   
3. MODEL INFERENCE
   Input: x=(1,197), edge_index=(2,0), edge_attr=(0,2)
   └─ Embedding: (1,197) → (1,128)
   └─ TGT Stack 1: Multi-head attention 
   └─ TGT Stack 2: Temporal attention
   └─ Output: logits=(1,2), probs=(1,2)
   
4. MC-DROPOUT UNCERTAINTY
   Forward Pass 1-5 with dropout enabled
   └─ Compute mean probs, std probs
   └─ Confidence = 1 - uncertainty
   
5. XAI EXPLANATION (2-3 seconds)
   a) GraphSVX: Masked coalitions → Shapley values
   b) Captum: Integrated gradients
   c) Attention: Extract from TGT heads
   d) Fraud Ring (N/A for new node)
   e) LLM: Generate narrative explanation
   
6. DISPLAY
   ├─ Metrics: fraud_prob, confidence, uncertainty
   ├─ Gauge: Risk visualization
   ├─ Charts: Feature importance, confidence bars
   └─ Text: LLM narrative + risk assessment

7. OUTPUT
   "This transaction shows 72% fraud probability with high confidence
   due to unusual feature patterns (Feature_5, Feature_12). 
   Recommend manual review. Shows patterns consistent with..."
```

---

## Configuration Management (`config.yaml`)

**Key Parameters**:

```yaml
# Model Architecture
model:
  embedding_dim: 128
  num_heads: 8
  num_stacks: 2
  num_layers_per_stack: 2
  dropout: 0.5

# Training (reference only, model is pre-trained)
training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001

# Paths
paths:
  data_raw: data/raw
  checkpoints: outputs/checkpoints
  logs: logs

# Device
device: auto  # auto-detect GPU, fallback to CPU

# WaveNet Temporal Encoding
wavelet:
  embedding_dim: 32
  scales: [0.5, 1.0, 2.0]

# Feature Normalization
normalization:
  method: gaussian  # mean=0, std=1
  clip: 3.0  # clip outliers
```

---

## Error Handling & Resilience

**Layer 1: Input Validation**
- Feature dimension check
- Edge index bounds check
- NaN/Inf detection

**Layer 2: Inference Safety**
- CPU fallback if CUDA fails
- Batch size limits
- Memory checks

**Layer 3: Explanation Robustness**
- Graceful degradation (missing methods)
- Template fallback for LLM
- Timeout protection (30s per explanation)

**Layer 4: API Reliability**
- Request validation
- Response schema validation
- Error logging
- Health checks

---

## Performance Characteristics

### Inference Speed

| Component | Latency | Notes |
|-----------|---------|-------|
| Feature embedding | 5ms | TGT input projection |
| TGT forward pass | 50-100ms | All-to-all temporal attention |
| MC-Dropout (5 pass) | 500ms | 5× forward with dropout |
| GraphSVX (5 coal) | 300ms | Coalition sampling |
| Captum gradients | 200ms | Backward pass |
| Fraud Ring | 100ms | Louvain detection |
| LLM (Phi-3 mini) | 1000ms | Text generation |
| **Total** | **2000-2500ms** | Full XAI pipeline |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Model weights | ~50MB | ETGT_FRD parameters |
| Batch (100 nodes) | ~800MB | Forward pass |
| Peak inference | 2-4GB | All components active |
| Dashboard cache | <500MB | Session state |

---

## Scalability & Future Work

### Current Limitations
- Single GPU inference only
- Batch size limited by memory
- Full graph needed for training (transfer learning possible)

### Planned Improvements
- Multi-GPU support
- Distributed inference (Ray)
- Model quantization (INT8)
- KNN-based approximations
- Streaming inference for online learning

---

## Dependencies Diagram

```
app.py (Streamlit Dashboard)
  ├── src.model (ETGT_FRD)
  ├── src.explain (XAI Pipeline)
  └── src.data_loader (EllipticDataLoader)
      ├── torch
      ├── torch_geometric
      └── pandas

app/api.py (FastAPI)
  ├── src.model
  ├── src.explain
  └── fastapi

src/model.py
  ├── torch
  ├── torch_geometric
  └── src.utils

src/explain.py
  ├── torch
  ├── captum
  ├── torch_geometric
  ├── networkx
  ├── python-louvain
  ├── jinja2
  └── transformers (for LLM)

src/data_loader.py
  ├── torch
  ├── torch_geometric
  ├── pandas
  ├── numpy
  └── pywt (wavelet)
```

---

## Testing & Validation

**Automated Checks**:
- `scripts/validate_environment.py`: Dependencies
- `scripts/validate_data.py`: Data integrity
- `scripts/validate_model.py`: Model loading & inference
- `scripts/validate_xai.py`: XAI method correctness
- `scripts/benchmark_performance.py`: Speed & memory

---

For implementation details, refer to individual module docstrings.
