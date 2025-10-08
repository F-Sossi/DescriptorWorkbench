# Keypoint Parameters Analysis

## 🔍 Issue: `keypoints_params` Section Not Parsed

You're correct - the `keypoints_params` section in your YAML files **does nothing** and is being **silently ignored**.

---

## Current YAML Structure (WRONG)

```yaml
keypoints:
  generator: "sift"
  source: "locked_in"
  keypoint_set_name: "sift_keynet_pairs"

keypoints_params:           # ❌ THIS SECTION IS IGNORED!
  max_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10.0
  sigma: 1.6
  num_octaves: 4
```

---

## Root Cause Analysis

### **YAML Parser Behavior** (`YAMLConfigLoader.cpp:42-44`)

```cpp
if (root["keypoints"]) {
    parseKeypoints(root["keypoints"], config.keypoints);  // Only parses "keypoints" section
}
// There is NO parsing for root["keypoints_params"]!
```

The parser **only looks for** `root["keypoints"]`, not `root["keypoints_params"]`.

### **What Actually Gets Parsed** (`YAMLConfigLoader.cpp:84-117`)

The `parseKeypoints()` function reads parameters **nested under** `keypoints:`, not as a separate section:

```cpp
void YAMLConfigLoader::parseKeypoints(const YAML::Node& node, ExperimentConfig::Keypoints& keypoints) {
    if (node["generator"]) { ... }

    // These keys must be INSIDE the "keypoints:" section, not separate!
    if (node["max_features"]) {
        keypoints.params.max_features = node["max_features"].as<int>();
    }
    if (node["contrast_threshold"]) { ... }
    if (node["edge_threshold"]) { ... }
    if (node["sigma"]) { ... }
    if (node["num_octaves"]) { ... }
}
```

---

## ✅ Correct YAML Structure

### **Option 1: All Parameters Under `keypoints:`** (RECOMMENDED)

```yaml
keypoints:
  generator: "sift"
  source: "locked_in"
  keypoint_set_name: "sift_keynet_pairs"
  # Keypoint detection parameters (nested)
  max_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10.0
  sigma: 1.6
  num_octaves: 4
```

This is the **intended structure** based on the parser implementation.

### **Option 2: Keep Separate Section (Requires Code Change)**

If you want to keep `keypoints_params:` as a separate section, you need to modify `YAMLConfigLoader.cpp`:

```cpp
// Add after line 44 in loadFromYAML():
if (root["keypoints_params"]) {
    parseKeypointsParams(root["keypoints_params"], config.keypoints.params);
}

// Add new function:
static void parseKeypointsParams(const YAML::Node& node, KeypointParams& params) {
    if (node["max_features"]) params.max_features = node["max_features"].as<int>();
    if (node["contrast_threshold"]) params.contrast_threshold = node["contrast_threshold"].as<float>();
    if (node["edge_threshold"]) params.edge_threshold = node["edge_threshold"].as<float>();
    if (node["sigma"]) params.sigma = node["sigma"].as<float>();
    if (node["num_octaves"]) params.num_octaves = node["num_octaves"].as<int>();
}
```

---

## Where Parameters Are Actually Used

### **1. Experiment Runner** (`experiment_runner.cpp:121-126`)

```cpp
static cv::Ptr<cv::Feature2D> makeDetector(const ExperimentConfig& cfg) {
    int maxf = cfg.keypoints.params.max_features;  // ✅ Uses max_features
    if (maxf > 0) return cv::SIFT::create(maxf);
    return cv::SIFT::create();
}
```

**Currently Used**:
- ✅ `max_features` → Passed to `cv::SIFT::create(maxf)`

**NOT Used** (ignored):
- ❌ `contrast_threshold`
- ❌ `edge_threshold`
- ❌ `sigma`
- ❌ `num_octaves`

**Why?** The simplified `makeDetector()` only passes `max_features`. OpenCV SIFT uses default values for other parameters.

### **2. Keypoint Manager** (`keypoint_manager.cpp`)

**Does NOT use YAML configurations at all!** It uses:
- **Command-line arguments** for detector type, min_distance, etc.
- **Hardcoded defaults** for SIFT parameters

Example:
```bash
./keypoint_manager generate-detector ../data sift
# Uses cv::SIFT::create() with ALL default parameters
```

There's **no way** to pass SIFT parameters to the keypoint manager currently.

---

## Impact Assessment

### **When Using Database Keypoints** (Most Common)
```yaml
keypoints:
  source: "locked_in"
  keypoint_set_name: "sift_keynet_pairs"
```

**Impact**: **ZERO** - All parameters are ignored because keypoints are loaded from database, not detected.

### **When Using Independent Detection**
```yaml
keypoints:
  source: "independent_detection"
  max_features: 2000        # ✅ This works (if nested correctly)
  contrast_threshold: 0.04  # ❌ This is ignored
```

**Impact**: **PARTIAL**
- `max_features` is used (limits keypoint count)
- All other parameters are ignored (SIFT uses OpenCV defaults)

---

## Full Parameter Support Implementation

To fully support SIFT parameters, modify `experiment_runner.cpp:121-126`:

```cpp
static cv::Ptr<cv::Feature2D> makeDetector(const ExperimentConfig& cfg) {
    const auto& kp = cfg.keypoints.params;

    // Full SIFT configuration
    return cv::SIFT::create(
        kp.max_features,          // nfeatures
        kp.num_octaves,           // nOctaveLayers
        kp.contrast_threshold,    // contrastThreshold
        kp.edge_threshold,        // edgeThreshold
        kp.sigma                  // sigma
    );
}
```

**OpenCV SIFT::create() signature**:
```cpp
cv::SIFT::create(
    int nfeatures = 0,              // Max features (0 = unlimited)
    int nOctaveLayers = 3,          // Pyramid layers per octave
    double contrastThreshold = 0.04, // Filter low-contrast keypoints
    double edgeThreshold = 10,      // Filter edge responses
    double sigma = 1.6              // Gaussian sigma for initial image
)
```

---

## Recommended Fixes

### **Fix 1: Update YAML Files** (IMMEDIATE)

Change from:
```yaml
keypoints:
  generator: "sift"
  source: "locked_in"
  keypoint_set_name: "sift_keynet_pairs"

keypoints_params:  # ❌ IGNORED
  max_features: 2000
```

To:
```yaml
keypoints:
  generator: "sift"
  source: "locked_in"
  keypoint_set_name: "sift_keynet_pairs"
  max_features: 2000          # ✅ Nested under keypoints
  contrast_threshold: 0.04
  edge_threshold: 10.0
  sigma: 1.6
  num_octaves: 4
```

### **Fix 2: Enable All Parameters** (CODE CHANGE)

Update `experiment_runner.cpp` to use all SIFT parameters:

```cpp
static cv::Ptr<cv::Feature2D> makeDetector(const ExperimentConfig& cfg) {
    const auto& kp = cfg.keypoints.params;

    if (cfg.keypoints.generator == KeypointGenerator::SIFT) {
        return cv::SIFT::create(
            kp.max_features,
            kp.num_octaves,
            kp.contrast_threshold,
            kp.edge_threshold,
            kp.sigma
        );
    } else if (cfg.keypoints.generator == KeypointGenerator::HARRIS) {
        // Add Harris parameters
    } else if (cfg.keypoints.generator == KeypointGenerator::ORB) {
        // Add ORB parameters
    }

    return cv::SIFT::create();  // Fallback
}
```

### **Fix 3: Add YAML Support to Keypoint Manager** (OPTIONAL)

Create `keypoint_config.yaml`:
```yaml
detector:
  type: "sift"
  max_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10.0
  sigma: 1.6
  num_octaves: 4

generation:
  mode: "independent"  # or "projected"
  set_name: "sift_custom"
  min_distance: 0.0    # For non-overlapping
```

Update `keypoint_manager.cpp` to accept YAML config:
```bash
./keypoint_manager generate-with-config ../data keypoint_config.yaml
```

---

## Default Values Reference

### **SIFT Defaults** (when parameters are missing)
```cpp
max_features:        0      // Unlimited
num_octaves:         3      // 3 layers per octave
contrast_threshold:  0.04   // Filters low-contrast keypoints
edge_threshold:      10     // Filters edge responses (higher = more edge kps)
sigma:               1.6    // Gaussian blur for initial image
```

### **Current Behavior** (Your YAML files)
```yaml
keypoints_params:    # Completely ignored
  max_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10.0
```

**Actual values used**:
- `max_features`: 2000 ✅ (if nested under `keypoints:`)
- `contrast_threshold`: 0.04 (OpenCV default, coincidentally matches)
- `edge_threshold`: 10.0 (OpenCV default, coincidentally matches)
- `sigma`: 1.6 (OpenCV default, coincidentally matches)
- `num_octaves`: 3 (OpenCV default, **different from your 4**)

---

## Summary

### ❌ **What's Broken**
1. `keypoints_params:` as a separate YAML section is **never parsed**
2. Most SIFT parameters are **ignored** even when correctly nested
3. Keypoint manager has **no YAML configuration** support

### ⚠️ **Current Workarounds**
1. Your experiments work because you use `source: "locked_in"` (pre-generated keypoints)
2. SIFT parameters coincidentally match defaults (except `num_octaves`)

### ✅ **Recommended Actions**
1. **Immediate**: Merge `keypoints_params` into `keypoints` section in all YAML files
2. **Short-term**: Update `makeDetector()` to use all SIFT parameters
3. **Long-term**: Add YAML config support to keypoint_manager

---

## Testing Your Fixes

### **Test 1: Verify Parameters Are Ignored**
```bash
# Current (parameters ignored)
./experiment_runner config.yaml

# Check database for max_features value
sqlite3 experiments.db "SELECT parameters FROM experiment_configs WHERE id = (SELECT MAX(id) FROM experiment_configs);"
# Should show max_features: 2000 in metadata
```

### **Test 2: After Fixing YAML Structure**
```yaml
keypoints:
  source: "independent_detection"
  max_features: 500  # Intentionally low to verify it works
```

```bash
./experiment_runner config.yaml
# Should see exactly 500 keypoints per image (or fewer if image is small)
```

### **Test 3: Verify Full Parameter Support**
After implementing Fix 2:
```yaml
keypoints:
  source: "independent_detection"
  max_features: 1000
  contrast_threshold: 0.08  # Higher = fewer keypoints (filters more)
  edge_threshold: 5         # Lower = fewer edge keypoints
```

```bash
./experiment_runner config.yaml
# Should see different keypoint distributions based on parameters
```

---

## Affected YAML Files

Run this to find all files with the broken `keypoints_params` section:

```bash
find config/experiments -name "*.yaml" -exec grep -l "keypoints_params" {} \;
```

**Files to fix**:
- All `*_systematic_analysis.yaml` files
- Any custom experiment configs using `keypoints_params`
