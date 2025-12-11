import sys
import torch
import numpy as np
try:
    import piq
    print(f"✅ piq version: {piq.__version__}")
    BRISQUE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import piq: {e}")
    print("Please run: pip install piq")
    BRISQUE_AVAILABLE = False
except Exception as e:
    print(f"❌ Unexpected error importing piq: {e}")
    BRISQUE_AVAILABLE = False

def test_brisque():
    if not BRISQUE_AVAILABLE:
        print("Skipping test because piq is missing.")
        return

    print("Testing BRISQUE calculation...")
    # Create a dummy image (N, C, H, W) range [0, 1]
    dummy_img = torch.rand(1, 3, 512, 512)
    
    try:
        score = piq.brisque(dummy_img, data_range=1.0, reduction='none')
        print(f"✅ BRISQUE calculation successful! Score: {score.item()}")
    except Exception as e:
        print(f"❌ Error during BRISQUE calculation: {e}")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    test_brisque()