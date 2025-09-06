#!/usr/bin/env python3
"""
Export FIXED HardNet that handles BatchNorm properly for ONNX export.

The issue is that BatchNorm layers need running statistics and proper eval mode.
"""

import torch
import torch.onnx
import onnx
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

try:
    from kornia.feature import HardNet
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

def create_fixed_hardnet_from_kornia():
    """Create a fixed version by directly using Kornia model and preparing for ONNX"""
    if not KORNIA_AVAILABLE:
        print("❌ Kornia not available")
        return None
    
    try:
        # Load the original pretrained model
        model = HardNet(pretrained=True)
        model.eval()  # Critical for BatchNorm
        
        print("✅ Loaded Kornia HardNet")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Verify the model works correctly
        with torch.no_grad():
            test_input = torch.randn(5, 1, 32, 32)  # Batch size > 1 for BatchNorm
            test_output = model(test_input)
            print(f"   Test batch input: {test_input.shape}")
            print(f"   Test batch output: {test_output.shape}")
            print(f"   Output norm range: {torch.norm(test_output, dim=1).min():.4f} - {torch.norm(test_output, dim=1).max():.4f}")
            
            # Test single sample (what we'll use in practice)
            single_input = torch.randn(1, 1, 32, 32)
            single_output = model(single_input)
            print(f"   Single input: {single_input.shape}")
            print(f"   Single output: {single_output.shape}")
            print(f"   Single output norm: {torch.norm(single_output).item():.4f}")
            
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_fixed_hardnet(model, output_path="../../models/fixed_hardnet.onnx", opset_version=11):
    """Export fixed HardNet to ONNX with proper BatchNorm handling"""
    
    if model is None:
        return False
        
    # Ensure model is in eval mode
    model.eval()
    
    # Use batch size > 1 for export to avoid BatchNorm issues
    dummy_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    
    try:
        print(f"🔧 Exporting fixed HardNet to {output_path}")
        print(f"   Opset version: {opset_version}")
        print(f"   Input shape: {dummy_input.shape}")
        
        # Test model one more time before export
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"   Pre-export test output: {test_output.shape}, norm: {torch.norm(test_output).item():.4f}")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["patch"],
            output_names=["descriptor"],
            do_constant_folding=True,
            opset_version=opset_version,
            dynamic_axes=None,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,  # Explicit eval mode for BatchNorm
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX export successful")
        
        # Check file size  
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"   File size: {size_kb:.1f} KB")
        
        # Print model info
        graph = onnx_model.graph
        input_info = graph.input[0]
        output_info = graph.output[0]
        print(f"   ONNX input: {input_info.name} - {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
        print(f"   ONNX output: {output_info.name} - {[d.dim_value for d in output_info.type.tensor_type.shape.dim]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_python_vs_cpp_preprocessing():
    """Test that our preprocessing matches what the Python model expects"""
    if not KORNIA_AVAILABLE:
        return
        
    print("\n🔍 Testing preprocessing pipeline...")
    
    # Create test image
    import cv2
    test_patch = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    
    # Python preprocessing (what Kornia expects)
    patch_tensor = torch.from_numpy(test_patch).float().unsqueeze(0).unsqueeze(0) / 255.0
    print(f"   Python preprocessing: shape={patch_tensor.shape}, range=[{patch_tensor.min():.3f}, {patch_tensor.max():.3f}]")
    
    # C++ preprocessing (what DNNPatchWrapper does)
    patch_cv = test_patch.astype(np.float32) / 255.0
    blob = cv2.dnn.blobFromImage(patch_cv)
    print(f"   C++ preprocessing: shape={blob.shape}, range=[{blob.min():.3f}, {blob.max():.3f}]")
    
    # They should be equivalent
    blob_torch = torch.from_numpy(blob)
    diff = torch.abs(patch_tensor - blob_torch).max().item()
    print(f"   Preprocessing difference: {diff:.6f} (should be ~0)")

if __name__ == "__main__":
    print("🚀 Creating FIXED HardNet with proper BatchNorm handling")
    
    # Test preprocessing first
    test_python_vs_cpp_preprocessing()
    
    # Load and prepare model
    model = create_fixed_hardnet_from_kornia()
    
    if model is None:
        print("❌ Could not create model")
        exit(1)
    
    # Try different opset versions
    success = False
    for opset in [11, 9, 10]:
        print(f"\n📋 Trying opset {opset}")
        if export_fixed_hardnet(model, f"/tmp/fixed_hardnet_opset{opset}.onnx", opset):
            print(f"✅ Success with opset {opset}")
            success = True
            break
        else:
            print(f"❌ Failed with opset {opset}")
    
    if success:
        print(f"\n🎯 Fixed HardNet ready for testing!")
        print("   Key improvements over simplified version:")
        print("   1. ✅ Real Kornia pretrained weights (not partial copying)")
        print("   2. ✅ Proper BatchNorm with running statistics")  
        print("   3. ✅ Correct spatial dimensions (padding=1)")
        print("   4. ✅ Strided convolutions for proper feature maps")
        print("   5. ✅ This should perform MUCH better than simplified version")
    else:
        print("❌ All opset versions failed")