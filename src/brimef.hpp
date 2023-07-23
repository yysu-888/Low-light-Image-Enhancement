#pragma once
#ifdef HAVE_EIGEN
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#endif
#include "timer.hpp"
#include "wls_filter.hpp"

#ifdef FAST_SMOOTH
#include "guidedfilter.h"
#endif


namespace cv {
    namespace intensity_transform {
#ifdef HAVE_EIGEN
        static Mat_<float> rgb2gm(const Mat_<Vec3f>& I)
        {
            Mat_<float> gm(I.rows, I.cols);
            gm.forEach(
                [&](float& pixel, const int* position) -> void
                {
                    pixel = std::pow(I(position[0], position[1])[0] * I(position[0], position[1])[1] * I(position[0], position[1])[2], 1 / 3.0f);
                }
            );

            return gm;
        }

        static Mat_<float> applyK(const Mat_<float>& I, float k, float a = -0.3293f, float b = 1.1258f) {
            float beta = std::exp((1 - std::pow(k, a)) * b);
            float gamma = std::pow(k, a);

            Mat_<float> J(I.size());
            pow(I, gamma, J);
            J = J * beta;

            return J;
        }

        static Mat_<Vec3f> applyK(const Mat_<Vec3f>& I, float k, float a = -0.3293f, float b = 1.1258f, float offset = 0) {
            float beta = std::exp((1 - std::pow(k, a)) * b);
            float gamma = std::pow(k, a);

            Mat_<Vec3f> J(I.size());
            pow(I, gamma, J);

            return J * beta + Scalar::all(offset);
        }

        static float entropy(const Mat_<float>& I)
        {
            Mat_<uchar> I_uchar;
            I.convertTo(I_uchar, CV_8U, 255);

            std::vector<Mat> planes;
            planes.push_back(I_uchar);
            Mat_<float> hist;
            const int histSize = 256;
            float range[] = { 0, 256 };
            const float* histRange = { range };
            calcHist(&I_uchar, 1, NULL, Mat(), hist, 1, &histSize, &histRange);

            Mat_<float> hist_norm = hist / cv::sum(hist)[0];

            float E = 0;
            for (int i = 0; i < hist_norm.rows; i++)
            {
                if (hist_norm(i, 0) > 0)
                {
                    E += hist_norm(i, 0) * std::log2(hist_norm(i, 0));
                }
            }

            return -E;
        }

        template <typename T> static int sgn(T val)
        {
            return (T(0) < val) - (val < T(0));
        }

        static double minimize_scalar_bounded(const Mat_<float>& I, double begin, double end,
            double xatol = 1e-4, int maxiter = 500)
        {
            // From scipy: https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/optimize.py#L1753-L1894
            //    """
            //    Options
            //    -------
            //    maxiter : int
            //        Maximum number of iterations to perform.
            //    disp: int, optional
            //        If non-zero, print messages.
            //            0 : no message printing.
            //            1 : non-convergence notification messages only.
            //            2 : print a message on convergence too.
            //            3 : print iteration results.
            //    xatol : float
            //        Absolute error in solution `xopt` acceptable for convergence.
            //    """
            double x1 = begin, x2 = end;

            if (x1 > x2) {
                std::runtime_error("The lower bound exceeds the upper bound.");
            }

            double sqrt_eps = std::sqrt(2.2e-16);
            double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
            double a = x1, b = x2;
            double fulc = a + golden_mean * (b - a);
            double nfc = fulc, xf = fulc;
            double rat = 0.0, e = 0.0;
            double x = xf;
            double fx = -entropy(applyK(I, static_cast<float>(x)));
            int num = 1;
            double fu = std::numeric_limits<double>::infinity();

            double ffulc = fx, fnfc = fx;
            double xm = 0.5 * (a + b);
            double tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
            double tol2 = 2.0 * tol1;

            for (int iter = 0; iter < maxiter && std::abs(xf - xm) >(tol2 - 0.5 * (b - a)); iter++)
            {
                int golden = 1;
                // Check for parabolic fit
                if (std::abs(e) > tol1) {
                    golden = 0;
                    double r = (xf - nfc) * (-entropy(applyK(I, static_cast<float>(x))) - ffulc);
                    double q = (xf - fulc) * (-entropy(applyK(I, static_cast<float>(x))) - fnfc);
                    double p = (xf - fulc) * q - (xf - nfc) * r;
                    q = 2.0 * (q - r);

                    if (q > 0.0) {
                        p = -p;
                    }
                    q = std::abs(q);
                    r = e;
                    e = rat;

                    // Check for acceptability of parabola
                    if (((std::abs(p) < std::abs(0.5 * q * r)) && (p > q * (a - xf)) &
                        (p < q * (b - xf)))) {
                        rat = (p + 0.0) / q;
                        x = xf + rat;

                        if (((x - a) < tol2) || ((b - x) < tol2)) {
                            double si = sgn(xm - xf) + ((xm - xf) == 0);
                            rat = tol1 * si;
                        }
                    }
                    else {      // do a golden-section step
                        golden = 1;
                    }
                }

                if (golden) {  // do a golden-section step
                    if (xf >= xm) {
                        e = a - xf;
                    }
                    else {
                        e = b - xf;
                    }
                    rat = golden_mean * e;
                }

                double si = sgn(rat) + (rat == 0);
                x = xf + si * std::max(std::abs(rat), tol1);
                fu = -entropy(applyK(I, static_cast<float>(x)));
                num += 1;

                if (fu <= fx) {
                    if (x >= xf) {
                        a = xf;
                    }
                    else {
                        b = xf;
                    }

                    fulc = nfc;
                    ffulc = fnfc;
                    nfc = xf;
                    fnfc = fx;
                    xf = x;
                    fx = fu;
                }
                else {
                    if (x < xf) {
                        a = x;
                    }
                    else {
                        b = x;
                    }

                    if ((fu <= fnfc) || (nfc == xf)) {
                        fulc = nfc;
                        ffulc = fnfc;
                        nfc = x;
                        fnfc = fu;
                    }
                    else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                        fulc = x;
                        ffulc = fu;
                    }
                }

                xm = 0.5 * (a + b);
                tol1 = sqrt_eps * std::abs(xf) + xatol / 3.0;
                tol2 = 2.0 * tol1;
            }

            return xf;
        }

        static Mat_<Vec3f> maxEntropyEnhance(const Mat_<Vec3f>& I, const Mat_<uchar>& isBad, float a, float b)
        {
            Mat_<Vec3f> input;
            resize(I, input, Size(50, 50));

            Mat_<float> Y = rgb2gm(input);

            Mat_<uchar> isBad_resize;
            resize(isBad, isBad_resize, Size(50, 50));

            std::vector<float> Y_vec;
            for (int i = 0; i < isBad_resize.rows; i++)
            {
                for (int j = 0; j < isBad_resize.cols; j++)
                {
                    if (isBad_resize(i, j) >= 0.5)
                    {
                        Y_vec.push_back(Y(i, j));
                    }
                }
            }

            if (Y_vec.empty())
            {
                return I;
            }

            Mat_<float> Y_mat(static_cast<int>(Y_vec.size()), 1, Y_vec.data());
            float opt_k = static_cast<float>(minimize_scalar_bounded(Y_mat, 1, 7));

            return applyK(I, opt_k, a, b, -0.01f);
        }

        static void BIMEF_impl(InputArray input_, OutputArray output_, float mu, float* k, float a, float b)
        {
            // CV_INSTRUMENT_REGION();

            Mat input = input_.getMat();
            if (input.empty())
            {
                return;
            }
            CV_CheckTypeEQ(input.type(), CV_8UC3, "Input image must be 8-bits color image (CV_8UC3).");

            Mat_<Vec3f> imgDouble;
            input.convertTo(imgDouble, CV_32F, 1 / 255.0);

            // t: scene illumination map
            Mat_<float> t_b(imgDouble.size());
            t_b.forEach(
                [&](float& pixel, const int* position) -> void
                {
                    pixel = std::max(std::max(imgDouble(position[0], position[1])[0],
                    imgDouble(position[0], position[1])[1]),
                    imgDouble(position[0], position[1])[2]);
                }
            );

            const float lambda = 0.5;
            const float sigma = 5;

            Mat_<float> t_b_resize;
            resize(t_b, t_b_resize, Size(), 0.5, 0.5);

#ifdef FAST_SMOOTH
            TICK(smooth_fast);
            int r = 21; // try r=2, 4, or 8
            double eps = 0.2 * 0.2; // try eps=0.1^2, 0.2^2, 0.4^2
            eps *= 1.0 * 1.0;   // Because the intensity range of our images is [0, 255]
            Mat_<float> t_our = guidedFilter(t_b_resize, t_b_resize, r, eps);
            TOCK(smooth_fast);
#else
            TICK(smooth);
            Mat_<float> t_our = tsmooth(t_b_resize, lambda, sigma);
            TOCK(smooth);
#endif

            resize(t_our, t_our, t_b.size());
            // k: exposure ratio
            Mat_<Vec3f> J;
            if (k == NULL)
            {
                Mat_<uchar> isBad(t_our.size());
                isBad.forEach(
                    [&](uchar& pixel, const int* position) -> void
                    {
                        pixel = t_our(position[0], position[1]) < 0.5 ? 1 : 0;
                    }
                );

                J = maxEntropyEnhance(imgDouble, isBad, a, b);
            }
            else
            {
                J = applyK(imgDouble, *k, a, b);

                // fix overflow
                J.forEach(
                    [](Vec3f& pixel, const int* /*position*/) -> void
                    {
                        pixel(0) = std::min(1.0f, pixel(0));
                        pixel(1) = std::min(1.0f, pixel(1));
                        pixel(2) = std::min(1.0f, pixel(2));
                    }
                );
            }
            // W: Weight Matrix
            Mat_<float> W(t_our.size());
            pow(t_our, mu, W);

            output_.create(input.size(), CV_8UC3);
            Mat output = output_.getMat();
            output.forEach<Vec3b>(
                [&](Vec3b& pixel, const int* position) -> void
                {
                    float w = W(position[0], position[1]);
                    pixel(0) = saturate_cast<uchar>((imgDouble(position[0], position[1])[0] * w + J(position[0], position[1])[0] * (1 - w)) * 255);
                    pixel(1) = saturate_cast<uchar>((imgDouble(position[0], position[1])[1] * w + J(position[0], position[1])[1] * (1 - w)) * 255);
                    pixel(2) = saturate_cast<uchar>((imgDouble(position[0], position[1])[2] * w + J(position[0], position[1])[2] * (1 - w)) * 255);
                }
            );
        }
#else
        static void BIMEF_impl(InputArray, OutputArray, float, float*, float, float)
        {
            CV_Error(Error::StsNotImplemented, "This algorithm requires OpenCV built with the Eigen library.");
        }
#endif

        void BIMEF(InputArray input, OutputArray output, float mu = 0.5, float a = -0.3293f, float b = 1.1258f)
        {
            BIMEF_impl(input, output, mu, NULL, a, b);
        }

        void BIMEF(InputArray input, OutputArray output, float k, float mu, float a, float b)
        {
            BIMEF_impl(input, output, mu, &k, a, b);
        }

    }
} // cv::intensity_transform::



