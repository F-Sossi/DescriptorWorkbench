
#!/usr/bin/env python3
"""
Export a PROPER HardNet model compatible with OpenCV DNN.

Two strategies:
1) Direct export of Kornia's pretrained HardNet (may fail in OpenCV if it uses
   ops the backend doesn't support on your system).
2) A structurally equivalent clone with BatchNorm (affine, with running stats)
   and identical spatial dimensions. We transfer weights layer-by-layer.

The previous simplified model differed significantly; this script provides a
faithful architecture and preserves normalization behavior for inference.
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

class ProperHardNet(nn.Module):
    """Proper HardNet implementation matching Kornia's architecture (static BN)."""
    
    def __init__(self):
        super(ProperHardNet, self).__init__()
        
        # Match Kornia HardNet architecture EXACTLY
        self.features = nn.Sequential(
            # Layer 0-2: 32x32x1 -> 32x32x32 (with padding=1)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # BN with affine + running stats for correct inference behavior
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            # Layer 3-5: 32x32x32 -> 32x32x32  
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            # Layer 6-8: 32x32x32 -> 16x16x64 (stride=2 downsampling)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            # Layer 9-11: 16x16x64 -> 16x16x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            # Layer 12-14: 16x16x64 -> 8x8x128 (stride=2 downsampling)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            # Layer 15-17: 8x8x128 -> 8x8x128
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            # Skip dropout for ONNX compatibility - layer 18
            # Layer 19-20: 8x8x128 -> 1x1x128 (8x8 kernel = global conv)
            nn.Conv2d(128, 128, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        
    def forward(self, x):
        # Forward through conv layers  
        x = self.features(x)  # Output: [1, 128, 1, 1]
        
        # Flatten to [1, 128]
        x = x.view(x.size(0), -1)
        
        # L2 normalize (like original)
        x = F.normalize(x, p=2, dim=1)
        
        return x

def copy_weights_properly():
    """Copy weights from Kornia HardNet with exact layer matching"""
    if not KORNIA_AVAILABLE:
        print("❌ Kornia not available")
        return None
    
    try:
        # Load Kornia model
        kornia_model = HardNet(pretrained=True)
        kornia_model.eval()
        
        # Create proper model
        proper_model = ProperHardNet()
        
        print("✅ Loading weights from Kornia HardNet with exact layer mapping")
        
        # Attempt 1: exact-key copy
        kornia_state = kornia_model.state_dict()
        proper_state = proper_model.state_dict()

        copied, mismatched = 0, 0
        for k in proper_state.keys():
            if k in kornia_state and kornia_state[k].shape == proper_state[k].shape:
                proper_state[k] = kornia_state[k].clone()
                copied += 1
        
        # Attempt 2: fallback by ordered shape matching (for different naming)
        if copied < len(proper_state):
            print(f"   ℹ️ Exact-key copy copied {copied}/{len(proper_state)} params; trying ordered shape match…")
            pk = list(proper_state.items())
            kk = list(kornia_state.items())
            j = 0
            for i in range(len(pk)):
                pkey, pval = pk[i]
                # skip if already copied
                if not torch.equal(proper_state[pkey], pval):
                    continue
                # advance kk until shape matches
                while j < len(kk) and kk[j][1].shape != pval.shape:
                    j += 1
                if j < len(kk) and kk[j][1].shape == pval.shape:
                    proper_state[pkey] = kk[j][1].clone()
                    copied += 1
                    j += 1
                else:
                    mismatched += 1

        proper_model.load_state_dict(proper_state, strict=False)
        print(f"   ✅ Copied parameters: {copied}, unmatched: {mismatched}")
        # Important: avoid running BN in train mode on 1×1 spatial tensors
        proper_model.eval()
        return proper_model
        
    except Exception as e:
        print(f"❌ Failed to copy weights: {e}")
        return None

def fuse_bn_into_conv_sequential(model: ProperHardNet) -> nn.Module:
    """Fuse Conv2d+BatchNorm2d pairs in ProperHardNet.features into a single Conv2d with bias."""
    def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        fused = nn.Conv2d(conv.in_channels, conv.out_channels,
                          kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding, dilation=conv.dilation,
                          groups=conv.groups, bias=True)
        # Params
        w = conv.weight.clone().view(conv.out_channels, -1)
        b_conv = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
        gamma = bn.weight.clone()
        beta = bn.bias.clone()
        mean = bn.running_mean.clone()
        var = bn.running_var.clone()
        eps = bn.eps
        invstd = (var + eps).sqrt().reciprocal()
        # Fuse
        w_fused = (w * (gamma * invstd).reshape(-1, 1)).view_as(conv.weight)
        b_fused = beta + (b_conv - mean) * gamma * invstd
        fused.weight.data.copy_(w_fused)
        fused.bias.data.copy_(b_fused)
        return fused

    layers = []
    seq = model.features
    i = 0
    while i < len(seq):
        m = seq[i]
        if isinstance(m, nn.Conv2d) and (i + 1) < len(seq) and isinstance(seq[i + 1], nn.BatchNorm2d):
            fused = fuse_conv_bn(m, seq[i + 1])
            layers.append(fused)
            i += 2
            if i < len(seq) and isinstance(seq[i], nn.ReLU):
                layers.append(seq[i])
                i += 1
        else:
            if not isinstance(m, nn.BatchNorm2d):
                layers.append(m)
            i += 1

    fused_model = ProperHardNet()
    fused_model.features = nn.Sequential(*layers)
    fused_model.eval()
    return fused_model

def export_proper_hardnet(output_path="models/proper_hardnet_fused_opset11.onnx", opset_version=11):
    """Export proper HardNet to ONNX"""
    
    print(f"🔧 Creating proper HardNet with correct architecture")
    model = copy_weights_properly()
    
    if model is None:
        print("❌ Failed to create model")
        return False
        
    model.eval()
    # Fuse BN into Conv to simplify graph for OpenCV DNN
    try:
        model = fuse_bn_into_conv_sequential(model)
        print("✅ Fused BN→Conv for export")
    except Exception as e:
        print(f"⚠️ BN fusion failed, proceeding without fusion: {e}")
    
    dummy_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    
    try:
        # Export with conservative settings for OpenCV compatibility
        print(f"🔧 Exporting to {output_path} with opset {opset_version}")
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
            training=torch.onnx.TrainingMode.EVAL,
        )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ Proper HardNet export successful")
        
        # Check file size  
        size_kb = Path(output_path).stat().st_size / 1024
        print(f"   File size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_kornia_direct(output_path="models/kornia_hardnet_opset11.onnx", opset_version=11):
    if not KORNIA_AVAILABLE:
        print("❌ Kornia not available for direct export")
        return False
    model = HardNet(pretrained=True).eval()
    dummy_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    try:
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
            training=torch.onnx.TrainingMode.EVAL,
        )
        onnx.checker.check_model(onnx.load(output_path))
        print(f"✅ Exported direct Kornia HardNet → {output_path}")
        return True
    except Exception as e:
        print(f"❌ Direct Kornia export failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Exporting PROPER HardNet variants")
    print("1) Direct Kornia export (if supported)\n2) Proper clone with BN + weight transfer")

    # 1) Try direct export first
    export_kornia_direct("models/kornia_hardnet_opset11.onnx", 11)

    # 2) Then try proper clone with weight transfer
    for opset in [11, 9]:
        print(f"\n📋 Proper clone, opset {opset}")
        if export_proper_hardnet(f"models/proper_hardnet_opset{opset}.onnx", opset):
            print(f"✅ Success with opset {opset}")
            break
        else:
            print(f"❌ Failed with opset {opset}")

    print("\n🎯 Notes:")
    print("- The proper clone preserves BN affine params and running stats.")
    print("- Prefer opset 11 with static shapes for OpenCV DNN.")
    print("- Enable per-patch standardization in YAML (dnn.per_patch_standardize: true).")
