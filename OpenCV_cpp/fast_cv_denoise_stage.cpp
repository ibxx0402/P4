/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2021, Raspberry Pi (Trading) Limited
 *
 * sobel_cv_stage.cpp - Sobel filter implementation, using OpenCV
 */

#include <libcamera/stream.h>

#include "core/rpicam_app.hpp"

#include "post_processing_stages/post_processing_stage.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"

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
	float diameter_ = 9;	// Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
	int sigmaColor_ = 50;	// Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
	int sigmaSpace_ = 50;   // Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
};

#define NAME "fast_cv_denoise"

char const *FastCVDenoise::Name() const
{
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
	StreamInfo info = app_->GetStreamInfo(stream_);
	BufferWriteSync w(app_, completed_request->buffers[stream_]);
	libcamera::Span<uint8_t> buffer = w.Get()[0];
	uint8_t *ptr = (uint8_t *)buffer.data();

	//Everything beyond this point is image processing...
	Mat src = Mat(info.height, info.width, CV_8UC1, ptr, info.stride);
	Mat dst;

	// Laplacian kernel
	cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        1, -2, 1,
        -2, 4, -2,
        1, -2, 1
    );

	// The source is already grayscale (Y channel from YUV420)
	cv::Mat greyMat = src.clone();

	// Convolution result matrix
	cv::Mat convResult;
    cv::filter2D(greyMat, convResult, CV_32F, kernel);

	// Calculate absolute values and sum
    cv::Mat absConvResult;
    cv::convertScaleAbs(convResult, absConvResult);

    // Calculate sigma 
    double sigma = cv::sum(absConvResult)[0];
    int width = greyMat.cols, height = greyMat.rows;
    
    // Normalize with mathematical adjustment similar to Python version
    sigma = sigma * std::sqrt(0.5 * M_PI) / (6.0 * (width - 2) * (height - 2));
	
	std::cout << "Sigma: " << sigma << std::endl;

	// Apply the bilateral filter
	bilateralFilter(src, dst, diameter_, sigmaColor_, sigmaSpace_);

	// Copy the filtered image back to the original buffer
	memcpy(ptr, dst.data, dst.total() * dst.elemSize());

	return false;
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new FastCVDenoise(app);
}

static RegisterStage reg(NAME, &Create);
