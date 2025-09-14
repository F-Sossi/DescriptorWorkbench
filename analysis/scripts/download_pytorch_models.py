#!/usr/bin/env python3
"""
Download and trace PyTorch descriptor models for LibTorch C++ integration.

This script downloads pretrained CNN descriptor models (HardNet, SOSNet, L2-Net)
and saves them as traced PyTorch script modules (.pt files) that can be loaded
directly in C++ with LibTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import sys

# Add models directory to path if it doesn't exist
models_dir = Path(__file__).parent.parent.parent / "models"
models_dir.mkdir(exist_ok=True)

def download_hardnet():
    """Download and trace HardNet model"""
    try:
        from kornia.feature import HardNet
        print("📥 Downloading HardNet from Kornia...")

        # Load pretrained model
        model = HardNet(pretrained=True)
        model.eval()

        print(f"   Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

        # Create example input (batch_size=1, channels=1, height=32, width=32)
        example_input = torch.randn(1, 1, 32, 32)

        # Test model
        with torch.no_grad():
            output = model(example_input)
            print(f"   Test output shape: {output.shape}")
            print(f"   Output norm: {torch.norm(output).item():.4f}")

        # Trace the model
        print("🔧 Tracing HardNet model...")
        traced_model = torch.jit.trace(model, example_input)

        # Save traced model
        model_path = models_dir / "hardnet.pt"
        traced_model.save(str(model_path))
        print(f"✅ HardNet saved to: {model_path}")

        # Verify saved model
        loaded_model = torch.jit.load(str(model_path))
        with torch.no_grad():
            loaded_output = loaded_model(example_input)
            diff = torch.abs(output - loaded_output).max().item()
            print(f"   Verification diff: {diff:.8f} (should be ~0)")

        return True

    except ImportError:
        print("❌ Kornia not available. Install with: pip install kornia")
        return False
    except Exception as e:
        print(f"❌ Failed to download HardNet: {e}")
        return False

def download_sosnet():
    """Download and trace SOSNet model"""
    try:
        from kornia.feature import SOSNet
        print("📥 Downloading SOSNet from Kornia...")

        # Load pretrained model
        model = SOSNet(pretrained=True)
        model.eval()

        print(f"   Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

        # Create example input
        example_input = torch.randn(1, 1, 32, 32)

        # Test model
        with torch.no_grad():
            output = model(example_input)
            print(f"   Test output shape: {output.shape}")
            print(f"   Output norm: {torch.norm(output).item():.4f}")

        # Trace the model
        print("🔧 Tracing SOSNet model...")
        traced_model = torch.jit.trace(model, example_input)

        # Save traced model
        model_path = models_dir / "sosnet.pt"
        traced_model.save(str(model_path))
        print(f"✅ SOSNet saved to: {model_path}")

        # Verify saved model
        loaded_model = torch.jit.load(str(model_path))
        with torch.no_grad():
            loaded_output = loaded_model(example_input)
            diff = torch.abs(output - loaded_output).max().item()
            print(f"   Verification diff: {diff:.8f} (should be ~0)")

        return True

    except ImportError:
        print("❌ Kornia not available. Install with: pip install kornia")
        return False
    except Exception as e:
        print(f"❌ Failed to download SOSNet: {e}")
        return False

def create_simple_l2net():
    """Create a simple L2-Net inspired model (since L2-Net is not in Kornia)"""
    try:
        print("🏗️  Creating simple L2-Net inspired model...")

        class SimpleL2Net(nn.Module):
            def __init__(self):
                super(SimpleL2Net, self).__init__()
                self.features = nn.Sequential(
                    # Layer 1
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),

                    # Layer 2
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),

                    # Layer 3
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),

                    # Layer 4
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                )

                # Global average pooling + final layer
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(128, 128)

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)  # L2 normalize

        # Create and initialize model
        model = SimpleL2Net()
        model.eval()

        # Initialize weights (random, but consistent)
        torch.manual_seed(42)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        print(f"   Model created: {sum(p.numel() for p in model.parameters())} parameters")

        # Test model
        example_input = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            output = model(example_input)
            print(f"   Test output shape: {output.shape}")
            print(f"   Output norm: {torch.norm(output).item():.4f}")

        # Trace the model
        print("🔧 Tracing Simple L2-Net model...")
        traced_model = torch.jit.trace(model, example_input)

        # Save traced model
        model_path = models_dir / "simple_l2net.pt"
        traced_model.save(str(model_path))
        print(f"✅ Simple L2-Net saved to: {model_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to create Simple L2-Net: {e}")
        return False

def main():
    print("🚀 Downloading and tracing CNN descriptor models for LibTorch...")
    print(f"Models will be saved to: {models_dir.absolute()}")
    print()

    success_count = 0

    # Download HardNet
    if download_hardnet():
        success_count += 1
    print()

    # Download SOSNet
    if download_sosnet():
        success_count += 1
    print()

    # Create Simple L2-Net
    if create_simple_l2net():
        success_count += 1
    print()

    print(f"📊 Summary: {success_count}/3 models successfully created")

    if success_count > 0:
        print("\n🎯 Models ready for LibTorch C++ integration!")
        print("Usage in C++:")
        print('  LibTorchWrapper hardnet("../models/hardnet.pt");')
        print('  LibTorchWrapper sosnet("../models/sosnet.pt");')
        print('  LibTorchWrapper l2net("../models/simple_l2net.pt");')
    else:
        print("\n❌ No models were successfully created")
        sys.exit(1)

if __name__ == "__main__":
    main()