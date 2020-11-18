//
// Created by ubuntu on 2020/5/24.
//

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <array>
#include <chrono>
#include <unordered_map>

#include <gflags/gflags.h>
#include <glog/logging.h>

namespace vloam
{

template<typename Scalar = double>
class DenseAccumulator
{
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

    template<int ROWS, int COLS, typename Derived>
    inline void addH(int i, int j, const Eigen::MatrixBase<Derived> &data)
    {
        H.template block<ROWS, COLS>(i, j) += data;
    }

    template<int ROWS, typename Derived>
    inline void addB(int i, const Eigen::MatrixBase<Derived> &data)
    {
        b.template segment<ROWS>(i) += data;
    }

    inline void reset(int opt_size)
    {
        H.setZero(opt_size, opt_size);
        b.setZero(opt_size);
    }
    inline void join(const DenseAccumulator<Scalar> &other)
    {
        H.template noalias() += other.H;
        b.template noalias() += other.b;
    }
    inline void print()
    {
        Eigen::IOFormat CleanFmt(2);
        LOG(INFO) << "H\n" << H.format(CleanFmt) << std::endl;
        LOG(INFO) << "b\n" << b.transpose().format(CleanFmt) << std::endl;
    }
    inline MatrixX &getH()
    { return H; }
    inline VectorX &getB()
    { return b; }
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    MatrixX H;
    VectorX b;
}; // DenseAccumulator
} // namespace vloam