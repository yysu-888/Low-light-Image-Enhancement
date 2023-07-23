#pragma once
#include "opencv2/opencv.hpp"
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
#include "guidedfilter.h"
#include "wls_filter.hpp"
#include <cmath>
#include "timer.hpp"

using namespace cv;
void gamma_correct(Mat_<float>& t_our, Mat& src, Mat& dst, float gamma)
{
    uchar lut1d[256];
    for (int i = 0;i < 256;i++) {
        lut1d[i] = uchar(fmin(std::powf(i / 255.f, gamma) * 255, 255));
    }
    uchar lut2d[256][256];
    for (int i = 0;i < 256;i++) {
        for (int j = 0;j < 256;j++) {
            lut2d[i][j] = uchar(fmin(i / (lut1d[j] + 1e-5) * 255, 255));
        }
    }

    for (int i = 0;i < dst.rows;i++) {
        uchar* p0 = src.ptr(i);
        uchar* p1 = dst.ptr(i);
        float* p2 = (float*)t_our.ptr(i);
        for (int j = 0;j < dst.cols;j++) {
            uchar idx = uchar(p2[j] * 255);
            p1[3 * j] = lut2d[p0[3 * j]][idx];
            p1[3 * j + 1] = lut2d[p0[3 * j + 1]][idx];
            p1[3 * j + 2] = lut2d[p0[3 * j + 2]][idx];

        }
    }
}

void LIME(cv::Mat& src, cv::Mat& dst, float gamma = 0.5)
{
    if (dst.empty()) {
        dst.create(src.size(), CV_8UC3);
    }
    // init illumination map
    Mat_<Vec3f> imgDouble;
    src.convertTo(imgDouble, CV_32F, 1 / 255.0);

    Mat_<float> t_b(imgDouble.size());
    t_b.forEach(
        [&](float& pixel, const int* position) -> void
        {
            pixel = std::max(std::max(imgDouble(position[0], position[1])[0],
            imgDouble(position[0], position[1])[1]),
            imgDouble(position[0], position[1])[2]);
        }
    );

    Mat_<float> t_b_resize;
    resize(t_b, t_b_resize, Size(), 0.5, 0.5);


#ifdef FAST_SMOOTH
    int h = t_b_resize.rows, w = t_b_resize.cols;
    int min_hw = max(h, w);
    if (min_hw > 500) {
        float s = 500.f / min_hw;
        resize(t_b_resize, t_b_resize, Size(0, 0), s, s);
    }
    int r = 21; 
    double eps = 0.1 * 0.1;
    eps *= 1 * 1;
    Mat_<float> t_our = guidedFilter(t_b_resize, t_b_resize, r, eps);
#else
    const float lambda = 0.5;
    const float sigma = 5;
    Mat_<float> t_our = tsmooth(t_b_resize, lambda, sigma);
#endif

    resize(t_our, t_our, t_b.size());
    TICK(gamma);
    gamma_correct(t_our, src, dst, gamma);
    TOCK(gamma);
}