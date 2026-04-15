"""
Validate environment setup and dependencies.

This script checks:
- Python version
- Required packages installed
- GPU availability (if applicable)
- File structure integrity
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    required = (3, 10)
    passed = version >= required
    
    status = "✅" if passed else "❌"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro} (required >= 3.10)")
    return passed


def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {str(e)[:60]}")
        return False


def check_packages():
    """Check all critical packages"""
    print("\n📦 **Critical Packages:**")
    packages = [
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("streamlit", "streamlit"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
    ]
    
    results = []
    for pkg, import_name in packages:
        results.append(check_package(pkg, import_name))
    
    return all(results)


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Available: {device_name}")
            print(f"   CUDA Device: {torch.cuda.current_device()}")
            print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
        else:
            print("⚠️  GPU not available (will use CPU)")
        return True
    except Exception as e:
        print(f"⚠️  GPU check failed: {str(e)[:60]}")
        return False


def check_file_structure():
    """Check project file structure"""
    print("\n📁 **Project Structure:**")
    required_files = [
        "app.py",
        "config.yaml",
        "requirements.txt",
        "data/raw/elliptic_txs_features.csv",
        "data/raw/elliptic_txs_edgelist.csv",
        "data/raw/elliptic_txs_classes.csv",
        "src/model.py",
        "src/explain.py",
        "src/data_loader.py",
        "outputs/checkpoints/best_model.pt",
    ]
    
    results = []
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        results.append(exists)
    
    return all(results)


def check_imports():
    """Check if core modules import successfully"""
    print("\n🔧 **Core Module Imports:**")
    modules = [
        ("src.model", "ETGT_FRD Model"),
        ("src.explain", "XAI Pipeline"),
        ("src.data_loader", "Data Loader"),
        ("src.utils", "Utilities"),
    ]
    
    results = []
    for module_name, desc in modules:
        try:
            __import__(module_name)
            print(f"✅ {desc} ({module_name})")
            results.append(True)
        except Exception as e:
            print(f"❌ {desc} ({module_name}): {str(e)[:60]}")
            results.append(False)
    
    return all(results)


def main():
    """Run all validation checks"""
    print("=" * 70)
    print("ETGT-FRD v2.0 - Environment Validation")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Critical Packages", check_packages),
        ("GPU Availability", check_gpu),
        ("File Structure", check_file_structure),
        ("Core Imports", check_imports),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n⚠️  {check_name} check failed: {str(e)}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ **ALL VALIDATION CHECKS PASSED**")
        print("Ready to run: streamlit run app.py")
    else:
        print("❌ **SOME CHECKS FAILED**")
        print("Please fix the issues above before running the application")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
