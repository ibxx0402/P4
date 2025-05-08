#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/photo.hpp>
#include <chrono>

#include <cmath>
#include <iostream>
#include <string>
#include <filesystem>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace std::chrono;

Mat src; Mat dst;

double estimateNoise(const cv::Mat& image) {
    // Laplacian kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        1, -2, 1,
        -2, 4, -2,
        1, -2, 1
    );

    // Convolution result matrix
    cv::Mat convResult;
    cv::filter2D(image, convResult, CV_32F, kernel);

    // Calculate absolute values and sum
    cv::Mat absConvResult;
    cv::convertScaleAbs(convResult, absConvResult);

    // Calculate sigma 
    double sigma = cv::sum(absConvResult)[0];
    int width = image.cols, height = image.rows;
    
    // Normalize with mathematical adjustment similar to Python version
    sigma = sigma * std::sqrt(0.5 * M_PI) / (6.0 * (width - 2) * (height - 2));

    return sigma;
}

int main() {
    string imagePath = "/home/comtek450/P4/tests/time_test/image.jpg";
    //src = imread(imagePath, IMREAD_GRAYSCALE);

    src = imread(imagePath);
    int iterations = 20000;
    cout << "medianBlur(src, dst, 3); @" << iterations << endl;
    for (int x = 0; x < 5; x++) {
        auto start = high_resolution_clock::now();
        
        for (int i =0; i<iterations; i++) {
            //GaussianBlur( src, dst, Size( 7, 7 ), 0.7, 0.7 );
            //GaussianBlur( src, dst, Size( 3, 3 ), 0.5, 0.5 );

            medianBlur(src, dst, 3);
            //medianBlur(src, dst, 1);

            //fastNlMeansDenoisingColored(src, dst, 2, 3, 3, 7);
            //fastNlMeansDenoisingColored(src, dst, 5, 7, 3, 7);

            //bilateralFilter(src, dst, 6, 10, 2);
            //bilateralFilter(src, dst, 8, 20, 2);

            //double noise = estimateNoise(src);
        }
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);

        // Calculate average time
        int64_t avg_time = duration.count() / iterations;

        // Collect all timings for percentile analysis
        std::vector<int64_t> timings;
        timings.reserve(iterations);

        for (int i = 0; i < iterations; i++) {
            auto iter_start = high_resolution_clock::now();
            
            medianBlur(src, dst, 3);
            
            auto iter_stop = high_resolution_clock::now();
            auto iter_duration = duration_cast<microseconds>(iter_stop - iter_start);
            timings.push_back(iter_duration.count());
        }

        auto slowest_time = *std::max_element(timings.begin(), timings.end());

        //sort the timing vector to be able to find 99 percentile
        std::sort(timings.begin(), timings.end());

        // Calculate the 99th percentile, as iterations * 0.99 is the index of the 99th percentile
        int64_t p99_time = timings[static_cast<size_t>(iterations * 0.99)];
        
        // Print the results
        cout << "AVG Time: " << avg_time << " microseconds" << endl;
        cout << "MAX Time: " << slowest_time << " microseconds percent slower: " << ((slowest_time - avg_time) / (double)avg_time) * 100 << endl;
        cout << "P99 Time: " << p99_time << " microseconds percent slower: " << ((p99_time - avg_time) / (double)avg_time) * 100  << "\n" << endl;
        }
    
    return 0;
}