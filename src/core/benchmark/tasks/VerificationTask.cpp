#include "VerificationTask.hpp"
#include "../metrics/APMetrics.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <vector>
#include <random>

namespace thesis_project::benchmark::tasks {

namespace {

// Helper to sample pairs from a larger set
std::vector<VerificationTaskPair> samplePairs(
    const std::vector<VerificationTaskPair>& pairs,
    int count,
    std::mt19937& rng) {

    if (count <= 0 || count >= static_cast<int>(pairs.size())) {
        return pairs;  // Return all if count is invalid or >= size
    }

    auto indices = metrics::sampleIndices(static_cast<int>(pairs.size()), count, rng);
    std::vector<VerificationTaskPair> sampled;
    sampled.reserve(count);
    for (int idx : indices) {
        sampled.push_back(pairs[idx]);
    }
    return sampled;
}

} // anonymous namespace

VerificationResults VerificationTask::run(
    const DescriptorBank& bank,
    const Config& config) {

    VerificationResults results;

    // Initialize RNG with config seed for reproducibility
    std::mt19937 rng(config.random_seed);

    // Sample pairs according to HPatches protocol (200K pos, 1M neg by default)
    // This maintains the correct positive:negative ratio for AP computation
    std::vector<VerificationTaskPair> sampled_pos = samplePairs(
        config.verification_pos_pairs,
        config.verification_num_positives,
        rng);

    std::vector<VerificationTaskPair> sampled_neg_intra = samplePairs(
        config.verification_neg_intra_pairs,
        config.verification_num_negatives,
        rng);

    std::vector<VerificationTaskPair> sampled_neg_inter = samplePairs(
        config.verification_neg_inter_pairs,
        config.verification_num_negatives,
        rng);

    // Helper to compute mean of three values
    auto mean3 = [](float a, float b, float c) {
        return (a + b + c) / 3.0f;
    };

    // Same-sequence verification (intra-sequence negatives)
    if (config.verification_same_seq && !sampled_pos.empty()) {
        results.same_seq_easy = config.include_easy
            ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "easy", "full")
            : 0.0f;

        results.same_seq_hard = config.include_hard
            ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "hard", "full")
            : 0.0f;

        results.same_seq_tough = config.include_tough
            ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "tough", "full")
            : 0.0f;

        results.same_seq_overall = mean3(
            results.same_seq_easy,
            results.same_seq_hard,
            results.same_seq_tough);

        // By scene type
        results.same_seq_illumination = mean3(
            config.include_easy ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "easy", "illum") : 0.0f,
            config.include_hard ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "hard", "illum") : 0.0f,
            config.include_tough ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "tough", "illum") : 0.0f);

        results.same_seq_viewpoint = mean3(
            config.include_easy ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "easy", "view") : 0.0f,
            config.include_hard ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "hard", "view") : 0.0f,
            config.include_tough ? runFromTasks(bank, sampled_pos, sampled_neg_intra, "tough", "view") : 0.0f);
    }

    // Different-sequence verification (inter-sequence negatives)
    if (config.verification_diff_seq && !sampled_pos.empty()) {
        results.diff_seq_easy = config.include_easy
            ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "easy", "full")
            : 0.0f;

        results.diff_seq_hard = config.include_hard
            ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "hard", "full")
            : 0.0f;

        results.diff_seq_tough = config.include_tough
            ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "tough", "full")
            : 0.0f;

        results.diff_seq_overall = mean3(
            results.diff_seq_easy,
            results.diff_seq_hard,
            results.diff_seq_tough);

        // By scene type
        results.diff_seq_illumination = mean3(
            config.include_easy ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "easy", "illum") : 0.0f,
            config.include_hard ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "hard", "illum") : 0.0f,
            config.include_tough ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "tough", "illum") : 0.0f);

        results.diff_seq_viewpoint = mean3(
            config.include_easy ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "easy", "view") : 0.0f,
            config.include_hard ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "hard", "view") : 0.0f,
            config.include_tough ? runFromTasks(bank, sampled_pos, sampled_neg_inter, "tough", "view") : 0.0f);
    }

    return results;
}

float VerificationTask::runFromTasks(
    const DescriptorBank& bank,
    const std::vector<VerificationTaskPair>& pos_pairs,
    const std::vector<VerificationTaskPair>& neg_pairs,
    const std::string& difficulty,
    const std::string& split) {

    if (pos_pairs.empty()) {
        return 0.0f;
    }

    std::vector<float> scores;
    std::vector<int> labels;

    // Reserve space for efficiency
    scores.reserve(pos_pairs.size() + neg_pairs.size());
    labels.reserve(pos_pairs.size() + neg_pairs.size());

    int pos_count = 0;

    // Process positive pairs
    for (const auto& pair : pos_pairs) {
        // Check split filter
        if (!matchesSplit(pair.s1, split)) continue;

        // Get target names for this difficulty
        std::string target1 = PatchKey::targetFromIndex(pair.t1, difficulty);
        std::string target2 = PatchKey::targetFromIndex(pair.t2, difficulty);

        // Get descriptors
        const cv::Mat& desc1 = bank.get(pair.s1, difficulty, target1);
        const cv::Mat& desc2 = bank.get(pair.s2, difficulty, target2);

        if (desc1.empty() || desc2.empty()) continue;
        if (pair.idx1 >= desc1.rows || pair.idx2 >= desc2.rows) continue;

        // Compute distance (lower = more similar)
        float dist = metrics::l2Distance(desc1.row(pair.idx1), desc2.row(pair.idx2));

        // Score is negative distance (higher = more likely positive)
        scores.push_back(-dist);
        labels.push_back(1);
        pos_count++;
    }

    // Process negative pairs
    for (const auto& pair : neg_pairs) {
        // Check split filter
        if (!matchesSplit(pair.s1, split)) continue;

        // Get target names for this difficulty
        std::string target1 = PatchKey::targetFromIndex(pair.t1, difficulty);
        std::string target2 = PatchKey::targetFromIndex(pair.t2, difficulty);

        // Get descriptors
        const cv::Mat& desc1 = bank.get(pair.s1, difficulty, target1);
        const cv::Mat& desc2 = bank.get(pair.s2, difficulty, target2);

        if (desc1.empty() || desc2.empty()) continue;
        if (pair.idx1 >= desc1.rows || pair.idx2 >= desc2.rows) continue;

        // Compute distance
        float dist = metrics::l2Distance(desc1.row(pair.idx1), desc2.row(pair.idx2));

        scores.push_back(-dist);
        labels.push_back(0);
    }

    if (pos_count == 0 || scores.empty()) {
        return 0.0f;
    }

    // Use trapezoidal AP computation (matches HPatches protocol)
    return metrics::computeAPTrapz(scores, labels, pos_count);
}

bool VerificationTask::matchesSplit(const std::string& scene, const std::string& split) {
    if (split == "full" || split == "all") {
        return true;
    }
    if (split == "illum") {
        return scene.size() > 2 && scene[0] == 'i' && scene[1] == '_';
    }
    if (split == "view") {
        return scene.size() > 2 && scene[0] == 'v' && scene[1] == '_';
    }
    return true;  // Unknown split, include all
}

} // namespace thesis_project::benchmark::tasks
