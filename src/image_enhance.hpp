#include "opencv2/opencv.hpp"
#include "brimef.hpp"
#include "lime.hpp"
#include "timer.hpp"

#ifdef OPEN_PYBIND11
#include <pybind11/pybind11.h>     
#include <pybind11/numpy.h>
#include "warp_cv_mat.hpp"
#endif

using namespace cv::intensity_transform;

void image_enhance_bimef(cv::Mat& input,cv::Mat& output){
    TICK(BIMEF)
    BIMEF(input,output);
    TOCK(BIMEF)
}

void image_enhance_lime(cv::Mat& input,cv::Mat& output,float gamma=0.8){
    TICK(LIME)
    LIME(input,output,gamma);
    TOCK(LIME)
}

#ifdef OPEN_PYBIND11
PYBIND11_MODULE(test, m)
{
    m.doc() = "image enhancement"; 
    m.def("image_enhance_bimef", &image_enhance_bimef);
    m.def("image_enhance_lime", &image_enhance_lime);
}
#endif





