# ETGT-FRD v2.0 API Documentation

## Overview

This document provides comprehensive API documentation for the ETGT-FRD (Explainable Temporal Graph Transformer for Fraud Detection) system.

---

## FastAPI Backend (`app/api.py`)

### Base URL
```
http://localhost:8000
```

### Authentication
Currently no authentication required. Production deployments should implement API key validation.

---

## Prediction Endpoints

### 1. Single Transaction Prediction

**Endpoint:** `POST /predict`

**Description:** Predict fraud probability and explainability for a single transaction

**Request Body:**
```json
{
  "features": [0.5, 0.3, 0.8, ...],  // 165-dimensional feature vector
  "edge_targets": [10, 42, 100],      // (optional) Connected node indices
  "edge_attrs": [[1, 0], [0, 1]],     // (optional) Edge attributes
  "include_xai": true                 // Include XAI explanations
}
```

**Response (200 OK):**
```json
{
  "fraud_probability": 0.72,
  "confidence": 0.85,
  "uncertainty": 0.15,
  "predicted_class": 1,
  "xai_explanation": {
    "graphsvx_importance": {
      "features": ["Feat_5", "Feat_12", "Feat_3"],
      "importances": [0.25, 0.18, 0.12]
    },
    "captum_importance": [0.1, 0.05, ...],
    "attention_maps": {...},
    "fraud_ring_analysis": {
      "num_communities": 3,
      "largest_community_size": 7,
      "largest_community_fraud_fraction": 0.71
    },
    "llm_explanation": "This transaction shows fraud characteristics..."
  },
  "processing_time_ms": 2150
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input features
- `500 Internal Server Error`: Model inference failed

---

### 2. Batch Prediction

**Endpoint:** `POST /batch-predict`

**Description:** Predict fraud for multiple transactions at once

**Request Body:**
```json
{
  "transactions": [
    {"features": [...], "edge_targets": [...]},
    {"features": [...], "edge_targets": [...]}
  ],
  "include_xai": false  // Faster without XAI
}
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "transaction_id": 0,
      "fraud_probability": 0.72,
      "confidence": 0.85
    },
    {
      "transaction_id": 1,
      "fraud_probability": 0.23,
      "confidence": 0.92
    }
  ],
  "total_processing_time_ms": 3500
}
```

---

### 3. Model Information

**Endpoint:** `GET /model-info`

**Description:** Get model metadata and statistics

**Response (200 OK):**
```json
{
  "model_name": "ETGT-FRD v2.0",
  "architecture": "Temporal Graph Transformer",
  "input_features": 197,
  "num_stacks": 2,
  "num_heads": 8,
  "embedding_dim": 128,
  "dropout_rate": 0.5,
  "model_checkpoint": "outputs/checkpoints/best_model.pt",
  "last_updated": "2026-04-14",
  "training_metrics": {
    "f1_score": 0.89,
    "precision": 0.85,
    "recall": 0.83,
    "auc_roc": 0.94
  },
  "inference_latency_ms": {
    "min": 95,
    "max": 250,
    "mean": 145
  }
}
```

---

### 4. Health Check

**Endpoint:** `GET /health`

**Description:** Check if API is running and model is loaded

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "timestamp": "2026-04-14T10:30:45Z"
}
```

---

## Streamlit Dashboard (`app.py`)

### Modes

#### 1. Historical Analysis Mode
- **Purpose**: Analyze past transactions with full explainability
- **Input**: Select transaction node index from dataset
- **Output**: 
  - Prediction metrics (probability, confidence, uncertainty)
  - Risk gauge visualization
  - Feature importance (GraphSVX)
  - Fraud ring analysis
  - LLM-generated explanation

#### 2. Real-Time Prediction Mode
- **Purpose**: Score new transactions with sliders
- **Input**: 
  - Feature intensity (0-1 scale)
  - Randomization option
- **Options**:
  - 🚀 Predict & Explain: Score with current settings
  - 📊 Load Sample: Use real dataset transaction
  - 🔄 Random: Generate random transaction

---

## Request/Response Schemas

### Feature Vector Format

**65-dimensional vector** representing transaction characteristics:
```python
features = [
    # Time features (indices 0-5)
    time_normalized,        # normalized time
    day_of_week,            # 0-6
    hour_of_day,            # 0-23
    ...
    
    # Transaction amount features (indices 6-20)
    amount_normalized,
    amount_log,
    ...
    
    # Network features (indices 21-100)
    in_degree,
    out_degree,
    ...
    
    # Temporal patterns (indices 101-164)
    temporal_entropy,
    temporal_variance,
    ...
]
```

### Edge Attributes Format

For connected transactions:
```python
edge_attrs = [
    [time_delta, same_time_period],  # Each connected node
    ...
]
```

---

## Error Handling

### Standard Error Response
```json
{
  "detail": "Error message description",
  "status_code": 400,
  "timestamp": "2026-04-14T10:30:45Z"
}
```

### Common Errors

| Code | Message | Fix |
|------|---------|-----|
| 400 | Invalid feature dimension | Provide exactly 165 features |
| 400 | Edge index out of range | Ensure edge targets < num_nodes |
| 500 | Model inference failed | Check GPU memory available |
| 503 | Service unavailable | Model still loading, retry in 5s |

---

## Performance Characteristics

### Latency Targets

| Operation | Latency | Notes |
|-----------|---------|-------|
| Single prediction only | 100-150ms | No XAI |
| + GraphSVX (5 coalitions) | 300-400ms | Feature importance |
| + MC-Dropout (5 passes) | 500-600ms | Uncertainty |
| Full XAI (all methods) | 2000-3000ms | All explanations |

### Throughput

- **Single endpoint**: ~6-10 req/sec (with full XAI)
- **Batch endpoint**: 200-500 transactions/sec
- **Dashboard**: Single user, interactive

### Memory Usage

- **Model weights**: ~50MB
- **Peak inference**: 2-4GB  
- **Batch of 1000**: 6-8GB

---

## Example Usage

### Python Client

```python
import requests
import json

# Connect to API
BASE_URL = "http://localhost:8000"

# Single prediction
features = [0.5] * 165  # Your feature vector
payload = {
    "features": features,
    "include_xai": True
}

response = requests.post(f"{BASE_URL}/predict", json=payload)
result = response.json()

print(f"Fraud Probability: {result['fraud_probability']:.1%}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Top Features: {result['xai_explanation']['graphsvx_importance']['features']}")
```

### cURL

```bash
# Get model info
curl http://localhost:8000/model-info

# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, ...], "include_xai": true}'
```

---

## Rate Limiting & Quotas

### Current Policy
- No rate limiting (development mode)
- Production should implement:
  - 100 requests/min per API key
  - 10,000 requests/day per user
  - 1GB/month data transfer

---

## Changelog

### v2.0 (Current)
- ✅ 6 integrated XAI methods
- ✅ Real-time inference
- ✅ MC-Dropout uncertainty
- ✅ LLM explanations
- ✅ Fraud ring detection

### v1.0
- Basic fraud detection
- Single feature importance method

---

## Support & Questions

For issues or questions:
1. Check [SETUP_AND_VALIDATION.md](../SETUP_AND_VALIDATION.md)
2. Review error messages in terminal
3. Check [Architecture Guide](ARCHITECTURE.md)
4. See [Research Paper](../research_contribution.md)
