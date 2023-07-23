#include "opencv2/opencv.hpp"
#include "brimef.hpp"
#include "timer.hpp"
#include "Eigen/Core"
#include "lime.hpp"
#include "image_enhance.hpp"

using namespace cv;
using namespace std;
using namespace cv::intensity_transform;
using namespace Eigen;

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "input the image path:argv[1]" << std::endl;
        return 0;
    }
    string path = argv[1];

    Mat im = imread(path);

    TICK(BIMEF);
    Mat dst(im.size(), im.type());
    image_enhance_bimef(im, dst);
    TOCK(BIMEF);

    Mat mat_viz;
    cv::hconcat(im, dst, mat_viz);

    imshow("0", mat_viz);
    waitKey(0);
    imwrite("../output/bimef.jpg", mat_viz);

    Mat dst_, mat_viz_;
    image_enhance_lime(im, dst_, 0.7);

    cv::hconcat(im, dst_, mat_viz_);
    imshow("1", mat_viz_);
    waitKey(0);
    imwrite("../output/lime_1.jpg", mat_viz_);



}