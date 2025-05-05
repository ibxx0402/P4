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
    string imagePath = "c++/image.jpg";
    src = cv::imread(imagePath);
     auto start = high_resolution_clock::now();

    for (int i =0; i<10000; i++) {
        //GaussianBlur( src, dst, Size( 7, 7 ), 0.7, 0.7 );
        //GaussianBlur( src, dst, Size( 3, 3 ), 0.5, 0.5 );

        //medianBlur(src, dst, 3);
        //medianBlur(src, dst, 1);

        fastNlMeansDenoisingColored(src, dst, 2, 3, 3, 7);
        fastNlMeansDenoisingColored(src, dst, 5, 7, 3, 7);

        //bilateralFilter(src, dst, 6, 10, 2);
        //bilateralFilter(src, dst, 8, 20, 2);

    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    cout << "Time taken by function: "
         << duration.count() / 100000 << " microseconds" << endl;
 /* 
    auto slowest = microseconds::zero();
    for (int i =0; i<10000; i++){
        auto start = high_resolution_clock::now();
        bilateralFilter(src, dst, 6, 10, 2);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        
        if (duration > slowest) {
            slowest = duration;
        }
    }
    cout << "Slowest time taken by function: "
         << slowest.count() << " microseconds" << endl; */



    return 0;
} 