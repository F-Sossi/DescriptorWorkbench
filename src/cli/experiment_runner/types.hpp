#pragma once

struct ProfilingSummary {
    double detect_ms = 0.0;
    double compute_ms = 0.0;
    double match_ms = 0.0;
    long total_images = 0;
    long total_kps = 0;
    int descriptor_dimension = 0;
};
