# Python Environment Setup for KeyNet Support

**Quick Reference Guide for Users**

---

## Quick Start

### For Conda Users (Recommended)

```bash
# 1. Activate your environment (REQUIRED!)
conda activate descriptor-compare

# 2. Verify packages are installed
python -c "import kornia; import torch; import cv2; print('✓ All packages available')"

# 3. Run KeyNet generation
cd build
./keypoint_manager generate-kornia-keynet ../data
```

### For First-Time Setup

```bash
# Create environment with required packages
conda create -n descriptor-compare python=3.11
conda activate descriptor-compare
pip install kornia torch opencv-python

# Build the project
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON
make -j$(nproc)

# Generate KeyNet keypoints
./keypoint_manager generate-kornia-keynet ../data
```

---

## ❓ Common Questions

### Q: Do I need to activate my conda environment?

**A: YES!** The environment **must be active** before running KeyNet commands.

```bash
# Check if environment is active
echo $CONDA_PREFIX
# Should show: /path/to/conda/envs/descriptor-compare

# If empty, activate it
conda activate descriptor-compare
```

### Q: Which conda environment does it use?

**A:** It uses **whichever environment is currently active** (via `conda activate`).

It does **NOT** automatically search for environments with the required packages.

### Q: I have multiple environments. How do I know which one to use?

**A:** Use the one that has `kornia`, `torch`, and `opencv-python` installed:

```bash
# Check if current environment has packages
python -c "import kornia; import torch; import cv2; print('OK')"

# If error, activate correct environment
conda activate descriptor-compare
```

### Q: What if I forget to activate the environment?

**A:** You'll get a clear error message:

```
[ERROR] No valid Python environment with required packages (kornia, torch, cv2)
[ERROR] Missing packages: kornia, torch
```

**Solution:** Activate your environment and try again.

### Q: Can I use virtual environment instead of conda?

**A:** Yes! Virtual environments work too:

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install kornia torch opencv-python

# Usage (activate every session)
source venv/bin/activate
cd build
./keypoint_manager generate-kornia-keynet ../data
```

### Q: Does it work in Docker?

**A:** Yes! Packages are pre-installed in the Docker image. No activation needed:

```bash
docker-compose -f docker-compose.dev.yml exec descriptor-dev bash
cd build
./keypoint_manager generate-kornia-keynet ../data
```

---

## Troubleshooting

### Error: "Missing Python packages"

**Symptom:**
```
[ERROR] Missing Python packages in Conda: kornia
```

**Solution 1:** Install missing packages in your environment
```bash
conda activate descriptor-compare
pip install kornia torch opencv-python
```

**Solution 2:** Verify you're in the correct environment
```bash
echo $CONDA_DEFAULT_ENV
# Should show: descriptor-compare

# If wrong environment, activate the correct one
conda activate descriptor-compare
```

### Error: "Could not find conda.sh"

**Symptom:**
```
[WARN] Could not find conda.sh, will use Python directly from: /path/to/python3
```

**Impact:** None! This is just a warning. The program will still work by calling Python directly.

**Why?** Your conda is already active, so re-activation isn't needed.

### Error: "Python executable not found"

**Symptom:**
```
[ERROR] Conda environment detected but Python executable not found
```

**Solution:** Your conda environment is corrupted. Recreate it:
```bash
conda deactivate
conda remove -n descriptor-compare --all
conda create -n descriptor-compare python=3.11
conda activate descriptor-compare
pip install kornia torch opencv-python
```

### I want to use system Python (no conda/venv)

**Solution:** Install packages globally:
```bash
pip3 install kornia torch opencv-python

# Or with --user flag
pip3 install --user kornia torch opencv-python

# Then run normally
cd build
./keypoint_manager generate-kornia-keynet ../data
```

---

## 📋 Environment Detection Checklist

The system checks in this order:

1. **Active conda environment** (`$CONDA_PREFIX` set)
   - Priority: **Highest**
   - Detected when: `conda activate myenv` was run

2. **Active virtual environment** (`$VIRTUAL_ENV` set)
   - Priority: **Medium**
   - Detected when: `source venv/bin/activate` was run

3. **System Python** (`which python3`)
   - Priority: **Lowest** (fallback)
   - Used when: No environment is active

---

## 🎓 Package Requirements

KeyNet requires these Python packages:

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `kornia` | KeyNet implementation | `pip install kornia` |
| `torch` | PyTorch backend | `pip install torch` |
| `cv2` | OpenCV bindings | `pip install opencv-python` |

**Verify installation:**
```bash
python -c "import kornia; print(f'Kornia: {kornia.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## 📚 Additional Resources

- **Implementation Details**: `docs/StatusDocs/PYTHON_ENVIRONMENT_PORTABILITY.md`
- **CLAUDE.md**: Section "Python Environment Detection (KeyNet Support)"
- **Test Suite**: `tests/unit/utils/test_python_environment_gtest.cpp`
- **Source Code**: `src/core/utils/PythonEnvironment.{hpp,cpp}`

---

## Tips

1. **Always activate before running KeyNet commands**
   ```bash
   conda activate descriptor-compare  # Do this first!
   cd build && ./keypoint_manager generate-kornia-keynet ../data
   ```

2. **Add to your shell startup** (optional, for convenience):
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   alias keynet="conda activate descriptor-compare && cd ~/repos/DescriptorWorkbench/build"

   # Usage
   keynet
   ./keypoint_manager generate-kornia-keynet ../data
   ```

3. **Check environment before long runs**
   ```bash
   # Quick verification
   echo $CONDA_DEFAULT_ENV && python -c "import kornia; print('✓')"
   ```

---

**Last Updated**: 2025-10-19
**Status**: Production-ready, all 21 tests passing
