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

// Use a static counter to label frames sequentially.
static int frame_counter = 0;

bool FastCVDenoise::Process(CompletedRequestPtr &completed_request)
{
    // === Stage 1: Capture Timestamp and Logging ===
    

    // === Stage 2: Acquire the Raw Image ===
    StreamInfo info = app_->GetStreamInfo(stream_);
    BufferWriteSync w(app_, completed_request->buffers[stream_]);
    libcamera::Span<uint8_t> buffer = w.Get()[0];
    uint8_t *ptr = (uint8_t *)buffer.data();

    // Assume the incoming image is in YUV420 format; treat it as a grayscale image.
    Mat src(info.height, info.width, CV_8UC1, ptr, info.stride);
    Mat dst; // To hold the filtered image.
    /*
    // === Stage 3: Noise Estimation BEFORE Filtering ===
    Mat kernel = (Mat_<float>(3, 3) <<
                   1, -2, 1,
                  -2,  4, -2,
                   1, -2, 1);
    Mat convResult;
    filter2D(src, convResult, CV_32F, kernel);
    Mat absConvResult;
    convertScaleAbs(convResult, absConvResult);
    double sigma_before = sum(absConvResult)[0];
    int width = src.cols, height = src.rows;
    sigma_before = sigma_before * std::sqrt(0.5 * M_PI) / (6.0 * (width - 2) * (height - 2));
    std::cout << "Sigma before filtering: " << sigma_before << std::endl;
    */
    // === Stage 4: Apply Bilateral Filtering ===
    bilateralFilter(src, dst, diameter_, sigmaColor_, sigmaSpace_);

    // === Stage 5: (Commented Out) Overlay the Pi Timestamp on the Filtered Image ===
    

    int label = frame_counter++;

    auto now = std::chrono::system_clock::now();
    uint64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now.time_since_epoch()).count();
    std::time_t time_tt = std::chrono::system_clock::to_time_t(now);
    std::tm *tm_ptr = std::localtime(&time_tt);
    char time_str[50];
    std::strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_ptr);

    std::ofstream pi_log("/home/comtek450/latencytest/pi_timestamp_log.txt", std::ios::app);
    if (pi_log.is_open())
    {
        // Log format: label time_string ts_ms
        pi_log << label << " " << ts_ms << "\n";
        pi_log.close();
    }
    else
    {
        std::cerr << "Unable to open pi_timestamp_log.txt for writing.\n";
    }
    std::stringstream overlay_text;
    overlay_text << "Frame " << label << " " << time_str << " " << ts_ms;
    putText(dst, overlay_text.str(), Point(10, dst.rows - 10),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255), 2);
    
    /*
    // === Stage 6: Noise Estimation AFTER Filtering (no overlay) ===
    Mat convResult_filtered;
    filter2D(dst, convResult_filtered, CV_32F, kernel);
    Mat absConvResult_filtered;
    convertScaleAbs(convResult_filtered, absConvResult_filtered);
    double sigma_after = sum(absConvResult_filtered)[0];
    sigma_after = sigma_after * std::sqrt(0.5 * M_PI) / (6.0 * (width - 2) * (height - 2));
    std::cout << "Sigma after filtering (no overlay): " << sigma_after << std::endl;
    */
    // === Stage 7: Copy the Processed Frame Back to the Buffer ===
    memcpy(ptr, dst.data, dst.total() * dst.elemSize());

    return false;  // Continue processing indefinitely.
}

static PostProcessingStage *Create(RPiCamApp *app)
{
    return new FastCVDenoise(app);
}

static RegisterStage reg(NAME, &Create);
