#!/usr/bin/env python
"""
DEBUG SCRIPT: Test Real-Time Prediction Components
This script tests all components individually to isolate the error
"""

import sys
import traceback
from pathlib import Path

print("="*60)
print("ETGT-FRD Real-Time Prediction - Debug Test")
print("="*60)

# Test 1: Basic imports
print("\n[1/8] Testing basic imports...")
try:
    import torch
    import yaml
    print("✓ Basic imports OK")
except Exception as e:
    print(f"✗ Basic imports FAILED: {e}")
    sys.exit(1)

# Test 2: Load config
print("\n[2/8] Loading config.yaml...")
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"✓ Config loaded")
    print(f"   - Paths: {cfg.get('paths', {})}")
except Exception as e:
    print(f"✗ Config loading FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Get device
print("\n[3/8] Getting device...")
try:
    from src.utils import get_device
    device = get_device(cfg)
    print(f"✓ Device: {device}")
except Exception as e:
    print(f"✗ Device getting FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load data
print("\n[4/8] Loading data...")
try:
    from src.data_loader import EllipticDataLoader
    loader = EllipticDataLoader(cfg)
    data, splits = loader.load()
    print(f"✓ Data loaded")
    print(f"   - Nodes (x): {data.x.shape}")
    print(f"   - Edges: {data.edge_index.shape}")
    print(f"   - Labels: {data.y.shape if hasattr(data, 'y') else 'No labels'}")
except Exception as e:
    print(f"✗ Data loading FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create model
print("\n[5/8] Creating model...")
try:
    import src.model
    model = src.model.ETGT_FRD.from_config(cfg).to(device)
    print(f"✓ Model created")
    print(f"   - Type: {type(model).__name__}")
    print(f"   - Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ Model creation FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Load checkpoint
print("\n[6/8] Loading model checkpoint...")
try:
    from src.utils import load_checkpoint
    ckpt_path =  Path(cfg["paths"]["checkpoints"]) / "best_model.pt"
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device=str(device))
        print(f"✓ Checkpoint loaded from {ckpt_path}")
    else:
        print(f"⚠ Checkpoint not found at {ckpt_path}")
except Exception as e:
    print(f"✗ Checkpoint loading FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Run inference
print("\n[7/8] Running inference...")
try:
    model.eval()
    
    # Test with real sample
    sample_idx = 50
    x_input = data.x[sample_idx:sample_idx+1].clone()
    edge_idx_input = data.edge_index
    edge_attr_input = data.edge_attr if data.edge_attr is not None else torch.zeros((data.edge_index.shape[1], 2))
    
    with torch.no_grad():
        logits, probs = model(
            x_input.to(device),
            edge_idx_input.to(device),
            edge_attr_input.to(device)
        )
    
    fraud_prob = probs[0, 1].item()
    pred_class = logits[0].argmax().item()
    
    print(f"✓ Inference successful")
    print(f"   - Sample: {sample_idx}")
    print(f"   - Fraud Prob: {fraud_prob:.4f}")
    print(f"   - Prediction: {'FRAUD' if pred_class == 1 else 'LICIT'}")
except Exception as e:
    print(f"✗ Inference FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: MC-Dropout uncertainty
print("\n[8/8] Testing MC-Dropout uncertainty...")
try:
    mean_probs, std_probs = model.predict_with_uncertainty(
        x_input.to(device),
        edge_idx_input.to(device),
        edge_attr_input.to(device),
        num_forward_passes=3
    )
    uncertainty = std_probs[0, 1].item()
    confidence = 1.0 - uncertainty
    
    print(f"✓ MC-Dropout uncertainty OK")
    print(f"   - Uncertainty: {uncertainty:.4f}")
    print(f"   - Confidence: {confidence:.4f}")
except Exception as e:
    print(f"✗ MC-Dropout FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nAll Real-Time Prediction components are working correctly.")
print("The error must be in the Streamlit UI code or interaction logic.")
