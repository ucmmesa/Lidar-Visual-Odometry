//
// Created by ubuntu on 2020/4/29.
//

#pragma once

#include <cstdint>
#include <memory> /// for forward

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>

#include <opencv2/core.hpp>

namespace vloam
{
using Timestamp = std::int64_t;
// Definitions relavant to frame types
using FrameId = std::uint64_t; // Frame id is used as the index of gtsam symbol
// (not as a gtsam key).

// Typedfs of commonly used Eigen matrices and vectors
using Point2 = gtsam::Point2;
using Point3 = gtsam::Point3;
using Vector3d = gtsam::Vector3;
using Vector6d = gtsam::Vector6;
using Matrix3x3d = gtsam::Matrix33;
using Matrix6x6d = gtsam::Matrix66;

using Matrices3
= std::vector<gtsam::Matrix3, Eigen::aligned_allocator<gtsam::Matrix3>>;
using Vectors3
= std::vector<Vector3d, Eigen::aligned_allocator<Vector3d>>;

// float version
using Vector3f = Eigen::Matrix<float, 3, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Matrix3x3f = Eigen::Matrix<float, 3,3>;
using Matrix6x6f = Eigen::Matrix<float, 6,6>;
using Matrixf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using Matrices3f = std::vector<Matrix3x3f, Eigen::aligned_allocator<Matrix3x3f>>;
using Vectors3f = std::vector<Vector3f, Eigen::aligned_allocator<Vector3f>>;

}
