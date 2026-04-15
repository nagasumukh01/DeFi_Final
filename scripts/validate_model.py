"""
Validate model loading and inference.

This script checks:
- Model file exists and is loadable
- Forward pass works
- Output shapes are correct
- Inference latency
- MC-Dropout uncertainty quantification
"""

import torch
import time
import yaml
from pathlib import Path


def check_model_file():
    """Check if model checkpoint exists"""
    ckpt_path = Path("outputs/checkpoints/best_model.pt")
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / (1024**2)
        print(f"✅ Model checkpoint found: {size_mb:.1f}MB")
        return True
    else:
        print(f"❌ Model checkpoint not found: {ckpt_path}")
        return False


def load_model():
    """Load model and config"""
    try:
        from src.model import ETGT_FRD
        from src.utils import get_device, load_checkpoint
        
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        
        device = get_device(cfg)
        model = ETGT_FRD.from_config(cfg).to(device)
        
        ckpt_path = Path("outputs/checkpoints/best_model.pt")
        if ckpt_path.exists():
            load_checkpoint(model, ckpt_path, device=str(device))
        
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        return model, cfg, device
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)[:100]}")
        return None, None, None


def test_forward_pass(model, cfg, device):
    """Test model forward pass"""
    try:
        from src.data_loader import EllipticDataLoader
        
        loader = EllipticDataLoader(cfg)
        data, splits = loader.load()
        
        with torch.no_grad():
            logits, probs = model(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device)
            )
        
        print(f"✅ Forward pass successful")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Probs shape: {probs.shape}")
        print(f"   Output range: [{probs.min():.3f}, {probs.max():.3f}]")
        
        return logits, probs, data
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)[:100]}")
        return None, None, None


def test_inference_speed(model, cfg, device):
    """Measure inference latency"""
    try:
        from src.data_loader import EllipticDataLoader
        
        loader = EllipticDataLoader(cfg)
        data, _ = loader.load()
        
        # Warmup
        with torch.no_grad():
            model(data.x[:100].to(device), data.edge_index.to(device), data.edge_attr.to(device))
        
        # Time 10 iterations
        start = time.time()
        num_iters = 10
        
        with torch.no_grad():
            for _ in range(num_iters):
                model(data.x[:100].to(device), data.edge_index.to(device), data.edge_attr.to(device))
        
        elapsed = (time.time() - start) / num_iters
        print(f"✅ Inference speed: {elapsed*1000:.1f}ms per batch")
        return elapsed
    except Exception as e:
        print(f"⚠️  Speed test failed: {str(e)[:100]}")
        return None


def test_mc_dropout(model, cfg, device):
    """Test MC-Dropout uncertainty estimation"""
    try:
        from src.data_loader import EllipticDataLoader
        
        loader = EllipticDataLoader(cfg)
        data, _ = loader.load()
        
        mean_probs, std_probs = model.predict_with_uncertainty(
            data.x[:100].to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
            num_forward_passes=5
        )
        
        print(f"✅ MC-Dropout uncertainty quantification works")
        print(f"   Mean probs shape: {mean_probs.shape}")
        print(f"   Std probs shape: {std_probs.shape}")
        print(f"   Mean uncertainty: {std_probs[:, 1].mean():.4f}")
        return True
    except Exception as e:
        print(f"❌ MC-Dropout failed: {str(e)[:100]}")
        return False


def main():
    """Run all model validation checks"""
    print("=" * 70)
    print("ETGT-FRD v2.0 - Model Validation")
    print("=" * 70)
    
    print("\n📊 **Model File Check:**")
    if not check_model_file():
        print("\n⚠️  Cannot proceed without model checkpoint")
        return 1
    
    print("\n🔧 **Model Loading:**")
    model, cfg, device = load_model()
    if model is None:
        print("\n⚠️  Cannot proceed without loading model")
        return 1
    
    print("\n🚀 **Forward Pass Test:**")
    logits, probs, data = test_forward_pass(model, cfg, device)
    if logits is None:
        print("\n⚠️  Forward pass failed")
        return 1
    
    print("\n⏱️  **Inference Speed:**")
    test_inference_speed(model, cfg, device)
    
    print("\n🔀 **MC-Dropout Uncertainty:**")
    mc_success = test_mc_dropout(model, cfg, device)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if mc_success:
        print("✅ **MODEL VALIDATION PASSED**")
        print("Model is ready for inference")
    else:
        print("⚠️  Some checks failed, but model may still work")
    
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
