/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * sobel_cv_stage.cpp - Sobel/Bilateral filter with frame timestamp and label logging,
 * and noise estimation before and after filtering.
 *
 * This module captures the image, logs the current timestamp and frame label,
 * estimates noise before filtering, applies bilateral filtering, and finally
 * estimates noise after filtering. The text overlay with the Pi timestamp is now
 * commented-out so that no overlay is added to the output image.
 */

#include <libcamera/stream.h>
#include "core/rpicam_app.hpp"
#include "post_processing_stages/post_processing_stage.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <opencv2/imgcodecs.hpp>

#include <ctime>
#include <sstream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>  // For std::sqrt

using namespace cv;
using Stream = libcamera::Stream;

class FastCVDenoise : public PostProcessingStage
{
public:
    FastCVDenoise(RPiCamApp *app) : PostProcessingStage(app) {}

    char const *Name() const override;
    void Read(boost::property_tree::ptree const &params) override;
    void Configure() override;
    bool Process(CompletedRequestPtr &completed_request) override;

private:
    Stream *stream_;
    float diameter_ = 9;       // For bilateral filtering
    int sigmaColor_ = 50;
    int sigmaSpace_ = 50;
};

#define NAME "fast_cv_denoise"

// Return the module name.
char const *FastCVDenoise::Name() const {
    return NAME;
}

void FastCVDenoise::Read(boost::property_tree::ptree const &params)
{
    diameter_ = params.get<float>("diameter", 9);
    sigmaColor_ = params.get<int>("sigmaColor", 50);
    sigmaSpace_ = params.get<int>("search_window_size", 50);
}

void FastCVDenoise::Configure()
{
    stream_ = app_->GetMainStream();
    if (!stream_ || stream_->configuration().pixelFormat != libcamera::formats::YUV420)
        throw std::runtime_error("FastCVDenoise: only YUV420 format supported");
}

bool FastCVDenoise::Process(CompletedRequestPtr &completed_request)
{
    // === Stage 1: Acquire the Raw Image ===
    StreamInfo info = app_->GetStreamInfo(stream_);
    BufferWriteSync w(app_, completed_request->buffers[stream_]);
    libcamera::Span<uint8_t> buffer = w.Get()[0];
    uint8_t *ptr = (uint8_t *)buffer.data();

    // Assume the incoming image is in YUV420 format; treat it as a grayscale image.
    Mat src(info.height, info.width, CV_8UC1, ptr, info.stride);
    
    // === Stage 2: Noise Estimation BEFORE Filtering ===
    Mat kernel = (Mat_<float>(3, 3) <<
                   1, -2, 1,
                  -2,  4, -2,
                   1, -2, 1);
    Mat convResult;
    filter2D(src, convResult, CV_32F, kernel);
    Mat absConvResult;
    convertScaleAbs(convResult, absConvResult);
    double sigma_now = sum(absConvResult)[0];
    int width = src.cols, height = src.rows;
    sigma_now = sigma_now * std::sqrt(0.5 * M_PI) / (6.0 * (width - 2) * (height - 2));

    // === Stage 3: Log Noise Estimation ===

    std::ofstream pi_log("/home/comtek450/latencytest/pi_noise_log.txt", std::ios::app);
    if (pi_log.is_open())
    {
        // Log format: label time_string ts_ms
        pi_log << sigma_now << "\n";
        pi_log.close();
    }
    else
    {
        std::cerr << "Unable to open pi_timestamp_log.txt for writing.\n";
    }
    return false;  // Continue processing indefinitely.
}

static PostProcessingStage *Create(RPiCamApp *app)
{
    return new FastCVDenoise(app);
}

static RegisterStage reg(NAME, &Create);
