# Patch Benchmark Matching Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable matching methods (1-NN and Ratio Test) to the patch benchmark pipeline using OpenCV's cv::BFMatcher.

**Architecture:** Add `MatchingConfig` struct to `BenchmarkTypes.hpp`, parse from YAML in `patch_benchmark.cpp`, modify `MatchingTask` to use `cv::BFMatcher` with the configured method instead of manual L2 loop.

**Tech Stack:** C++17, OpenCV cv::BFMatcher, yaml-cpp

---

### Task 1: Add MatchingConfig to BenchmarkTypes.hpp

**Files:**
- Modify: `src/core/benchmark/BenchmarkTypes.hpp:52-100`

**Step 1: Add the enum and struct after existing structs**

Add after line 47 (after `RetrievalTaskItem` struct):

```cpp
/**
 * @brief Matching method for patch benchmark evaluation
 */
enum class PatchMatchingMethod {
    NEAREST_NEIGHBOR,  ///< 1-NN: find single closest match per query
    RATIO_TEST         ///< Lowe's ratio test: reject if d1/d2 >= threshold
};

/**
 * @brief Configuration for descriptor matching in patch benchmark
 */
struct MatchingConfig {
    PatchMatchingMethod method = PatchMatchingMethod::NEAREST_NEIGHBOR;
    float ratio_threshold = 0.8f;  ///< Ratio test threshold (only for RATIO_TEST)
    int norm_type = cv::NORM_L2;   ///< OpenCV norm type (NORM_L2 or NORM_L1)
};
```

**Step 2: Add matching field to Config struct**

Add inside `Config` struct around line 65, after `bool matching_enabled = true;`:

```cpp
    MatchingConfig matching;  ///< Matching method configuration
```

**Step 3: Build to verify compilation**

Run: `cd /home/frank/repos/DescriptorWorkbench/build && make -j$(nproc) 2>&1 | head -50`
Expected: Successful compilation (or unrelated errors)

**Step 4: Commit**

```bash
git add src/core/benchmark/BenchmarkTypes.hpp
git commit -m "feat(benchmark): add MatchingConfig struct for configurable matching"
```

---

### Task 2: Update MatchingTask signature

**Files:**
- Modify: `src/core/benchmark/tasks/MatchingTask.hpp:72-84`
- Modify: `src/core/benchmark/tasks/MatchingTask.cpp:235-299`

**Step 1: Update computeSceneMAP signature in header**

Change the private method signature at line 81:

```cpp
    static float computeSceneMAP(
        const cv::Mat& ref_desc,
        const cv::Mat& target_desc,
        const MatchingConfig& matching_config,
        float* accuracy_out = nullptr);
```

**Step 2: Update the call sites in MatchingTask.cpp**

Change line 109 (inside OpenMP parallel block):

```cpp
                    float mAP = computeSceneMAP(ref_desc, target_desc, config.matching, &accuracy);
```

Change line 184 (inside non-OpenMP fallback):

```cpp
                float mAP = computeSceneMAP(ref_desc, target_desc, config.matching, &accuracy);
```

**Step 3: Update computeSceneMAP function signature at definition**

Change line 235:

```cpp
float MatchingTask::computeSceneMAP(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const MatchingConfig& matching_config,
    float* accuracy_out) {
```

**Step 4: Build to verify compilation**

Run: `cd /home/frank/repos/DescriptorWorkbench/build && make -j$(nproc) 2>&1 | head -50`
Expected: Compiles (implementation still uses old logic, but signature matches)

**Step 5: Commit**

```bash
git add src/core/benchmark/tasks/MatchingTask.hpp src/core/benchmark/tasks/MatchingTask.cpp
git commit -m "refactor(benchmark): update computeSceneMAP signature to accept MatchingConfig"
```

---

### Task 3: Implement cv::BFMatcher-based matching

**Files:**
- Modify: `src/core/benchmark/tasks/MatchingTask.cpp:235-299`

**Step 1: Replace the manual L2 loop with cv::BFMatcher implementation**

Replace the entire `computeSceneMAP` function body (lines 240-298) with:

```cpp
float MatchingTask::computeSceneMAP(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const MatchingConfig& matching_config,
    float* accuracy_out) {

    if (ref_desc.empty() || target_desc.empty()) {
        if (accuracy_out) *accuracy_out = 0.0f;
        return 0.0f;
    }

    if (ref_desc.rows != target_desc.rows) {
        if (accuracy_out) *accuracy_out = 0.0f;
        return 0.0f;
    }

    const int N = ref_desc.rows;

    std::vector<float> nn_scores;
    std::vector<int> nn_labels;
    nn_scores.reserve(N);
    nn_labels.reserve(N);

    int correct = 0;

    cv::BFMatcher matcher(matching_config.norm_type, false);

    if (matching_config.method == PatchMatchingMethod::NEAREST_NEIGHBOR) {
        // 1-NN: find single best match per query
        std::vector<cv::DMatch> matches;
        matcher.match(ref_desc, target_desc, matches);

        for (const auto& match : matches) {
            bool is_correct = (match.trainIdx == match.queryIdx);
            if (is_correct) correct++;

            nn_scores.push_back(-match.distance);
            nn_labels.push_back(is_correct ? 1 : 0);
        }
    } else if (matching_config.method == PatchMatchingMethod::RATIO_TEST) {
        // kNN with k=2, apply Lowe's ratio test
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(ref_desc, target_desc, knn_matches, 2);

        for (int q = 0; q < N; ++q) {
            const auto& match_pair = knn_matches[q];

            if (match_pair.size() >= 2) {
                float ratio = match_pair[0].distance / match_pair[1].distance;

                if (ratio < matching_config.ratio_threshold) {
                    // Match accepted
                    bool is_correct = (match_pair[0].trainIdx == q);
                    if (is_correct) correct++;

                    nn_scores.push_back(-match_pair[0].distance);
                    nn_labels.push_back(is_correct ? 1 : 0);
                } else {
                    // Match rejected by ratio test - counts as incorrect
                    nn_scores.push_back(-match_pair[0].distance);
                    nn_labels.push_back(0);
                }
            } else if (match_pair.size() == 1) {
                // Only one match available, can't apply ratio test
                // Accept it (no second neighbor to compare)
                bool is_correct = (match_pair[0].trainIdx == q);
                if (is_correct) correct++;

                nn_scores.push_back(-match_pair[0].distance);
                nn_labels.push_back(is_correct ? 1 : 0);
            } else {
                // No match found
                nn_scores.push_back(-std::numeric_limits<float>::max());
                nn_labels.push_back(0);
            }
        }
    }

    if (accuracy_out) {
        *accuracy_out = static_cast<float>(correct) / static_cast<float>(N);
    }

    return metrics::computeAPTrapz(nn_scores, nn_labels, N);
}
```

**Step 2: Build to verify compilation**

Run: `cd /home/frank/repos/DescriptorWorkbench/build && make -j$(nproc) 2>&1 | head -50`
Expected: Successful compilation

**Step 3: Commit**

```bash
git add src/core/benchmark/tasks/MatchingTask.cpp
git commit -m "feat(benchmark): implement cv::BFMatcher-based matching with 1-NN and ratio test"
```

---

### Task 4: Parse matching config from YAML

**Files:**
- Modify: `src/cli/patch_benchmark.cpp:614-664`

**Step 1: Add helper function for parsing matching method**

Add this helper function near the top of the file, after `toLowerCopy` (around line 230):

```cpp
thesis_project::benchmark::PatchMatchingMethod parseMatchingMethod(const std::string& str) {
    std::string lower = toLowerCopy(str);
    if (lower == "ratio_test" || lower == "ratio" || lower == "snn") {
        return thesis_project::benchmark::PatchMatchingMethod::RATIO_TEST;
    }
    // Default to nearest neighbor
    return thesis_project::benchmark::PatchMatchingMethod::NEAREST_NEIGHBOR;
}
```

**Step 2: Parse matching config in loadConfig**

Inside the `if (tasks["matching"])` block (around line 614), add after `config.benchmark.matching_enabled = parseEnabled(tasks["matching"], ...)`:

```cpp
        if (tasks["matching"].IsMap()) {
            const auto& matching = tasks["matching"];
            if (matching["method"]) {
                config.benchmark.matching.method = parseMatchingMethod(matching["method"].as<std::string>());
            }
            if (matching["ratio_threshold"]) {
                config.benchmark.matching.ratio_threshold = matching["ratio_threshold"].as<float>();
            }
            if (matching["norm"]) {
                std::string norm_str = toLowerCopy(matching["norm"].as<std::string>());
                if (norm_str == "l1") {
                    config.benchmark.matching.norm_type = cv::NORM_L1;
                } else {
                    config.benchmark.matching.norm_type = cv::NORM_L2;
                }
            }
        }
```

**Step 3: Build to verify compilation**

Run: `cd /home/frank/repos/DescriptorWorkbench/build && make -j$(nproc) patch_benchmark 2>&1 | head -50`
Expected: Successful compilation

**Step 4: Commit**

```bash
git add src/cli/patch_benchmark.cpp
git commit -m "feat(benchmark): parse matching config from YAML"
```

---

### Task 5: Test with existing config (1-NN default)

**Files:**
- Test: existing patch benchmark configs

**Step 1: Run patch benchmark with default matching (should behave as before)**

Run: `cd /home/frank/repos/DescriptorWorkbench/build && ./patch_benchmark --patches ../hpatches-release --descriptor sift --no-tough 2>&1 | tail -20`
Expected: Results should show mAP values (verify it runs without errors)

**Step 2: Create a test config with ratio_test**

Create test config file:

```bash
cat > /tmp/test_matching.yaml << 'EOF'
patches:
  path: "../hpatches-release"
  difficulty:
    easy: true
    hard: false
    tough: false

tasks:
  matching:
    enabled: true
    method: "ratio_test"
    ratio_threshold: 0.8
  verification:
    enabled: false
  retrieval:
    enabled: false

descriptors:
  - name: "sift_ratio_test"
    type: "sift"

performance:
  verbose: true
  num_threads: 4

output:
  print_results: true
  save_to_database: false
EOF
```

**Step 3: Run with ratio_test config**

Run: `cd /home/frank/repos/DescriptorWorkbench/build && ./patch_benchmark /tmp/test_matching.yaml 2>&1 | tail -30`
Expected: Should run and show mAP values (may differ from 1-NN results)

**Step 4: Commit test config as example**

```bash
cp /tmp/test_matching.yaml /home/frank/repos/DescriptorWorkbench/config/patch_benchmarks/patch_matching_method_test.yaml
git add config/patch_benchmarks/patch_matching_method_test.yaml
git commit -m "test(benchmark): add example config for ratio_test matching method"
```

---

### Task 6: Update skill documentation

**Files:**
- Modify: `skills/patch-benchmark/README.md` (if exists) or note in design doc

**Step 1: Check if patch-benchmark skill exists**

Run: `ls -la /home/frank/repos/DescriptorWorkbench/skills/`

**Step 2: Update design doc with usage examples**

Append to the design doc:

```markdown
## Usage Examples

### YAML Configuration

**1-NN (default, backward compatible):**
```yaml
tasks:
  matching:
    enabled: true
```

**Ratio Test:**
```yaml
tasks:
  matching:
    enabled: true
    method: "ratio_test"
    ratio_threshold: 0.8
```

**L1 Norm (for binary descriptors):**
```yaml
tasks:
  matching:
    enabled: true
    method: "nearest_neighbor"
    norm: "l1"
```
```

**Step 3: Commit**

```bash
git add docs/plans/2025-03-01-patch-benchmark-matching-config-design.md
git commit -m "docs: add usage examples to matching config design doc"
```

---

## Summary

After completing all tasks:
- `BenchmarkTypes.hpp` has `PatchMatchingMethod` enum and `MatchingConfig` struct
- `MatchingTask` uses `cv::BFMatcher` with configurable method
- YAML configs can specify `tasks.matching.method: "nearest_neighbor"` or `"ratio_test"`
- Default behavior unchanged (1-NN with L2 norm)
- Example config provided for testing
