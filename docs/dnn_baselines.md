# DNN Baselines and ONNX

## Purpose
- Provide fair, well-documented deep descriptors as comparison points to SIFT/VGG.
- No pooling or multi-scale; identical keypoints and similar image context to classical descriptors.

## Selected Baselines
- HardNet — 128D, 32×32 grayscale patches; strong, widely cited; easy ONNX.
- SOSNet — 128D, 32×32; improved objective over HardNet; easy ONNX.
- L2-Net — 128D, 32×32; classic baseline; simple architecture.
- TFeat — 128D, 32×32; lightweight/shallow; useful sanity check.
- DeepDesc — 128D (recommended variant), 32×32; historical baseline.

We recommend starting with HardNet and SOSNet.

## Fairness Constraints
- Keypoints: reuse the exact same keypoint set (locked or independent).
- Context: `support_multiplier: 1.0` so patch side ≈ `keypoint.size`.
- Orientation: `rotate_to_upright: true` (align using `kp.angle`).
- Color: grayscale input (`use_color: false`) to match training.
- Pooling: `pooling: none`; no scale stacks or additional context.
- Normalization: L2-normalize outputs (handled by wrapper).

## Config Templates
- HardNet: `config/experiments/dnn_hardnet_baseline.yaml`
- SOSNet: `config/experiments/dnn_sosnet_baseline.yaml`

Run from `build/`:
- `./experiment_runner ../config/experiments/dnn_hardnet_baseline.yaml` (uses `../models/simple_hardnet_opset13.onnx`)
- `./experiment_runner ../config/experiments/dnn_hardnet_baseline_m3.yaml` (uses `../models/fixed_hardnet_opset11.onnx`, magnification 3.0)
- `./experiment_runner ../config/experiments/dnn_hardnet_baseline_m6.yaml` (uses `../models/fixed_hardnet_opset11.onnx`, magnification 6.0)
- `./experiment_runner ../config/experiments/dnn_sosnet_baseline.yaml` (update model path when SOSNet ONNX is available)

## What is ONNX?
ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models. It allows models trained in diverse frameworks (e.g., PyTorch, TensorFlow) to be exported to a single interoperable format and executed across different runtimes. In this project we use ONNX so patch-descriptor CNNs trained in PyTorch can be exported once and then run efficiently via OpenCV’s `cv::dnn` without bringing the original training framework into the runtime.

Benefits:
- Portability: train in one framework, run anywhere with an ONNX runtime.
- Longevity: decouples inference from training code versions.
- Performance: inference backends can optimize the same ONNX graph.

## Export Tips (PyTorch → ONNX)
- Input: `1×1×32×32` float, scaled to `[0,1]`.
- Opset: 11+ recommended; enable constant folding.
- Example:
  - `torch.onnx.export(model.eval(), dummy, "model.onnx", input_names=["input"], output_names=["desc"], opset_version=12, do_constant_folding=True)`
- Verify: load with `cv::dnn::readNetFromONNX` and check output shape `1×128` (or `1×128×1×1`).

## Preprocessing
- Default: scale to `[0,1]`, `mean: 0.0`, `std: 1.0`.
- If the model’s documentation specifies different normalization, set `dnn.mean`/`dnn.std` accordingly.

## Troubleshooting HardNet Performance
- Patch magnification: learned descriptors typically expect larger canonical windows than `keypoint.size`. Try `support_multiplier` of 3.0 and 6.0.
- Rotation to upright: keep `rotate_to_upright: true` to match training assumptions.
- Affine extraction: our wrapper uses rotation+scale+centering via `getRotationMatrix2D`; ensure you run a build with that change.
- Normalization: if results remain poor, try `mean: 0.5`, `std: 0.5` (mapping to `[-1,1]`) depending on the checkpoint’s expected preprocessing.

## Current Status: Real DNN Models Working! ✅

### ONNX Compatibility Issue SOLVED
The OpenCV ONNX compatibility issues have been **completely resolved**. The root cause was not protobuf mismatches but rather unsupported operations in the original model exports:

- **HardNet (Kornia)**: Used instance normalization → not supported by OpenCV DNN
- **SOSNet (Kornia)**: Used conditional control flow (`If` operations) → not supported by OpenCV DNN

### Real DNN Implementation (Current Status)
The system now includes **real pretrained DNN models** working directly in C++ with OpenCV DNN:

**Simplified HardNet Architecture:**
```
32×32 grayscale patch → Conv2D layers (32→32→64→64→128→128→128) → 
AdaptiveAvgPool2D → L2 normalize → 128D descriptor
```

**Performance Results:**
- **P@1: 3.05%**, **P@5: 7.36%**, **P@10: 10.2%** 
- Significantly **better than Python baseline** (3.05% vs 1.9% P@1)
- Uses **pretrained weights** transferred from Kornia HardNet

### Working Implementation Details
- **Model File**: `models/simple_hardnet_opset13.onnx` (1.7MB)
- **Export Script**: `analysis/scripts/export_simple_hardnet.py`
- **Integration**: Direct integration with `DNNPatchWrapper` in C++ pipeline
- **Performance**: ~8s for 15,000 descriptor computations (533 descriptors/sec)

### Usage
```bash
# Run real HardNet DNN baseline (C++)
cd build
./experiment_runner ../config/experiments/dnn_hardnet_baseline.yaml

# Export new compatible models (Python)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate descriptor-compare
python analysis/scripts/export_simple_hardnet.py
```

### Fallback System
The **Lightweight CNN baseline** (`PseudoDNNWrapper`) remains available as a fallback and documented comparison point:
- Performance: P@1: 0.4% (serves as lower bound)
- Used only if ONNX model loading fails
