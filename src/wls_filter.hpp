#pragma once
#include "opencv2/opencv.hpp"
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>

using namespace cv;

static void diff(const Mat_<float>& src, Mat_<float>& srcVDiff, Mat_<float>& srcHDiff)
{
    srcVDiff = Mat_<float>(src.size());
    for (int i = 0; i < src.rows; i++)
    {
        if (i < src.rows-1)
        {
            for (int j = 0; j < src.cols; j++)
            {
                srcVDiff(i,j) = src(i+1,j) - src(i,j);
            }
        }
        else
        {
            for (int j = 0; j < src.cols; j++)
            {
                srcVDiff(i,j) = src(0,j) - src(i,j);
            }
        }
    }

    srcHDiff = Mat_<float>(src.size());
    for (int j = 0; j < src.cols-1; j++)
    {
        for (int i = 0; i < src.rows; i++)
        {
            srcHDiff(i,j) = src(i,j+1) - src(i,j);
        }
    }
    for (int i = 0; i < src.rows; i++)
    {
        srcHDiff(i,src.cols-1) = src(i,0) - src(i,src.cols-1);
    }
}


template <class numeric_t>
static Eigen::SparseMatrix<numeric_t> spdiags(const Eigen::Matrix<numeric_t,-1,-1> &B,
                                              const Eigen::VectorXi &d, int m, int n) {
    typedef Eigen::Triplet<numeric_t> triplet_t;
    std::vector<triplet_t> triplets;
    triplets.reserve(static_cast<size_t>(std::min(m,n)*d.size()));

    for (int k = 0; k < d.size(); ++k) {
        int diag = d(k);  // get diagonal
        int i_start = std::max(-diag, 0); // get row of 1st element
        int i_end = std::min(m, m-diag-(m-n)); // get row of last element
        int j = -std::min(0, -diag); // get col of 1st element
        int B_i; // start index i in matrix B
        if (m < n) {
            B_i = std::max(-diag,0); // m < n
        } else {
            B_i = std::max(0,diag); // m >= n
        }
        for (int i = i_start; i < i_end; ++i, ++j, ++B_i) {
            triplets.push_back( {i, j,  B(B_i,k)} );
        }
    }
    Eigen::SparseMatrix<numeric_t> A(m,n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

static void computeTextureWeights(const Mat_<float>& x, float sigma, float sharpness, Mat_<float>& W_h, Mat_<float>& W_v)
{
    Mat_<float> dt0_v, dt0_h;
    diff(x, dt0_v, dt0_h);

    Mat_<float> gauker_h;
    Mat_<float> kernel_h = Mat_<float>::ones(1, static_cast<int>(sigma));
    filter2D(dt0_h, gauker_h, -1, kernel_h, Point(-1,-1), 0, BORDER_CONSTANT);

    Mat_<float> gauker_v;
    Mat_<float> kernel_v = Mat_<float>::ones(static_cast<int>(sigma), 1);
    filter2D(dt0_v, gauker_v, -1, kernel_v, Point(-1,-1), 0, BORDER_CONSTANT);

    W_h = Mat_<float>(gauker_h.size());
    W_v = Mat_<float>(gauker_v.size());

    for (int i = 0; i < gauker_h.rows; i++)
    {
        for (int j = 0; j < gauker_h.cols; j++)
        {
            W_h(i,j) = 1 / (std::abs(gauker_h(i,j)) * std::abs(dt0_h(i,j)) + sharpness);
            W_v(i,j) = 1 / (std::abs(gauker_v(i,j)) * std::abs(dt0_v(i,j)) + sharpness);
        }
    }
}

static Mat solveLinearEquation(const Mat_<float>& img, Mat_<float>& W_h_, Mat_<float>& W_v_, float lambda)
{
    Eigen::MatrixXf W_h;
    cv2eigen(W_h_, W_h);
    Eigen::MatrixXf tempx(W_h.rows(), W_h.cols());
    tempx.block(0, 1, tempx.rows(), tempx.cols()-1) = W_h.block(0, 0, W_h.rows(), W_h.cols()-1);
    for (Eigen::Index i = 0; i < tempx.rows(); i++)
    {
        tempx(i,0) = W_h(i, W_h.cols()-1);
    }

    Eigen::MatrixXf W_v;
    cv2eigen(W_v_, W_v);
    Eigen::MatrixXf tempy(W_v.rows(), W_v.cols());
    tempy.block(1, 0, tempx.rows()-1, tempx.cols()) = W_v.block(0, 0, W_v.rows()-1, W_v.cols());
    for (Eigen::Index j = 0; j < tempy.cols(); j++)
    {
        tempy(0,j) = W_v(W_v.rows()-1, j);
    }


    Eigen::VectorXf dx(W_h.rows()*W_h.cols());
    Eigen::VectorXf dy(W_v.rows()*W_v.cols());

    Eigen::VectorXf dxa(tempx.rows()*tempx.cols());
    Eigen::VectorXf dya(tempy.rows()*tempy.cols());

    //Flatten in a col-major order
    for (Eigen::Index j = 0; j < W_h.cols(); j++)
    {
        for (Eigen::Index i = 0; i < W_h.rows(); i++)
        {
            dx(j*W_h.rows() + i) = -lambda*W_h(i,j);
            dy(j*W_h.rows() + i) = -lambda*W_v(i,j);

            dxa(j*W_h.rows() + i) = -lambda*tempx(i,j);
            dya(j*W_h.rows() + i) = -lambda*tempy(i,j);
        }
    }

    tempx.setZero();
    tempx.col(0) = W_h.col(W_h.cols()-1);

    tempy.setZero();
    tempy.row(0) = W_v.row(W_v.rows()-1);

    W_h.col(W_h.cols()-1).setZero();
    W_v.row(W_v.rows()-1).setZero();

    Eigen::VectorXf dxd1(tempx.rows()*tempx.cols());
    Eigen::VectorXf dyd1(tempy.rows()*tempy.cols());
    Eigen::VectorXf dxd2(W_h.rows()*W_h.cols());
    Eigen::VectorXf dyd2(W_v.rows()*W_v.cols());

    //Flatten in a col-major order
    for (Eigen::Index j = 0; j < tempx.cols(); j++)
    {
        for (Eigen::Index i = 0; i < tempx.rows(); i++)
        {
            dxd1(j*tempx.rows() + i) = -lambda*tempx(i,j);
            dyd1(j*tempx.rows() + i) = -lambda*tempy(i,j);

            dxd2(j*tempx.rows() + i) = -lambda*W_h(i,j);
            dyd2(j*tempx.rows() + i) = -lambda*W_v(i,j);
        }
    }

    Eigen::MatrixXf dxd(dxd1.rows(), dxd1.cols()+dxd2.cols());
    dxd << dxd1, dxd2;

    Eigen::MatrixXf dyd(dyd1.rows(), dyd1.cols()+dyd2.cols());
    dyd << dyd1, dyd2;

    const int k = img.rows*img.cols;
    const int r = img.rows;
    Eigen::Matrix<int, 2, 1> diagx_idx;
    diagx_idx << -k+r, -r;
    Eigen::SparseMatrix<float> Ax = spdiags(dxd, diagx_idx, k, k);

    Eigen::Matrix<int, 2, 1> diagy_idx;
    diagy_idx << -r+1, -1;
    Eigen::SparseMatrix<float> Ay = spdiags(dyd, diagy_idx, k, k);

    Eigen::MatrixXf D = (dx + dy + dxa + dya);
    D = Eigen::MatrixXf::Ones(D.rows(), D.cols()) - D;

    Eigen::Matrix<int, 1, 1> diag_idx_zero;
    diag_idx_zero << 0;
    Eigen::SparseMatrix<float> A = (Ax + Ay) + Eigen::SparseMatrix<float>((Ax + Ay).transpose()) + spdiags(D, diag_idx_zero, k, k);

    //CG solver of Eigen
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<float> > cg;
    cg.setTolerance(0.1f);
    cg.setMaxIterations(50);
    cg.compute(A);
    Mat_<float> img_t = img.t();
    Eigen::Map<const Eigen::VectorXf> tin(img_t.ptr<float>(), img_t.rows*img_t.cols);
    Eigen::VectorXf x = cg.solve(tin);

    Mat_<float> tout(img.rows, img.cols);
    tout.forEach(
        [&](float &pixel, const int * position) -> void
        {
            pixel = x(position[1]*img.rows + position[0]);
        }
    );

    return tout;
}

static Mat_<float> tsmooth(const Mat_<float>& src, float lambda=0.01f, float sigma=3.0f, float sharpness=0.001f)
{
    Mat_<float> W_h, W_v;
    computeTextureWeights(src, sigma, sharpness, W_h, W_v);

    Mat_<float> S = solveLinearEquation(src, W_h, W_v, lambda);

    return S;
}
