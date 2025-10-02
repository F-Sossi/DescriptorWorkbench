#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/config/experiment_config.hpp"

using thesis_project::pooling::PoolingFactory;

namespace {
cv::Mat makeGray(int w=220, int h=160) {
    cv::Mat img(h, w, CV_8UC1, cv::Scalar(0));
    cv::circle(img, {w/2, h/2}, std::min(w,h)/4, cv::Scalar(200), -1);
    return img;
}
std::vector<cv::KeyPoint> gridKps(int w, int h, int step=24, int margin=24) {
    std::vector<cv::KeyPoint> kps;
    for (int y=margin;y<h-margin;y+=step) for (int x=margin;x<w-margin;x+=step) kps.emplace_back((float)x,(float)y,12.0f);
    return kps;
}
void expectNear(const cv::Mat& a, const cv::Mat& b, double atol=1e-4, double rtol=1e-4) {
    ASSERT_EQ(a.type(), b.type()); ASSERT_EQ(a.rows,b.rows); ASSERT_EQ(a.cols,b.cols);
    for (int r=0;r<a.rows;++r) { const float* pa=a.ptr<float>(r),*pb=b.ptr<float>(r); for (int c=0;c<a.cols;++c) {
        double va=pa[c], vb=pb[c]; double diff=std::abs(va-vb); double tol=atol+rtol*std::max(std::abs(va),std::abs(vb)); ASSERT_LE(diff,tol);
    }}
}

}

TEST(DSPWeightedPoolingTest, ManualWeightedAverageMatches) {
    experiment_config cfg;
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_SIFT;
    cfg.descriptorOptions.descriptorColorSpace = D_BW;
    cfg.descriptorOptions.poolingStrategy = DOMAIN_SIZE_POOLING;
    cfg.descriptorOptions.scales = {0.75f, 1.0f, 1.25f};
    cfg.descriptorOptions.scale_weights = {1.0f, 2.0f, 1.0f};
    cfg.refreshDetectors();

    cv::Mat img = makeGray();
    auto kps = gridKps(img.cols, img.rows, 26, 30);

    auto dsp = PoolingFactory::createStrategy(DOMAIN_SIZE_POOLING);
    cv::Mat pooled = dsp->computeDescriptors(img, kps, cfg.detector, cfg);
    ASSERT_FALSE(pooled.empty());

    auto descriptorForScale = [&](float scale) {
        experiment_config tmp = cfg;
        tmp.descriptorOptions.scales = {scale};
        tmp.descriptorOptions.scale_weights.clear();
        tmp.descriptorOptions.scale_weighting_mode = 0;
        auto single = PoolingFactory::createStrategy(DOMAIN_SIZE_POOLING);
        return single->computeDescriptors(img, kps, cfg.detector, tmp);
    };

    cv::Mat d1 = descriptorForScale(0.75f);
    cv::Mat d2 = descriptorForScale(1.0f);
    cv::Mat d3 = descriptorForScale(1.25f);
    cv::Mat expected = cv::Mat::zeros(d1.rows, d1.cols, d1.type());
    cv::Mat weight_sum = cv::Mat::zeros(d1.rows, d1.cols, CV_32F);
    std::array<std::pair<const cv::Mat*, float>, 3> terms{{{&d1,1.0f}, {&d2,2.0f}, {&d3,1.0f}}};
    for (const auto& term : terms) {
        const cv::Mat& desc = *term.first;
        float weight = term.second;
        for (int r = 0; r < desc.rows; ++r) {
            const float* src = desc.ptr<float>(r);
            float* dst = expected.ptr<float>(r);
            float* ws  = weight_sum.ptr<float>(r);
            for (int c = 0; c < desc.cols; ++c) {
                float v = src[c];
                if (v != -1.f) {
                    dst[c] += weight * v;
                    ws[c]  += weight;
                }
            }
        }
    }
    for (int r = 0; r < expected.rows; ++r) {
        float* dst = expected.ptr<float>(r);
        float* ws  = weight_sum.ptr<float>(r);
        for (int c = 0; c < expected.cols; ++c) {
            if (ws[c] > 0.0f) {
                dst[c] /= ws[c];
            } else {
                dst[c] = 0.0f;
            }
        }
    }

    expectNear(pooled, expected, 1e-4, 1e-4);
}
