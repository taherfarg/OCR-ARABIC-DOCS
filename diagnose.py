"""
Run this to diagnose GPU issues.
"""
import sys

print("=" * 60)
print("  GPU DIAGNOSTIC")
print("=" * 60)

# 1. Python
print(f"\nPython: {sys.version}")

# 2. PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA compiled: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        
        # Test
        x = torch.randn(100, 100, device="cuda")
        print(f"Test tensor device: {x.device}")
        print("\n✅ GPU IS WORKING!")
    else:
        print("\n❌ CUDA NOT AVAILABLE!")
        print("\nFix:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        
except ImportError:
    print("❌ PyTorch not installed!")

# 3. Transformers
try:
    import transformers
    print(f"\nTransformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed!")

print("\n" + "=" * 60)
