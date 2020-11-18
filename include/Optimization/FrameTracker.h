//
// Created by ubuntu on 2020/5/10.
//

#ifndef FRAMETRACKE_H
#define FRAMETRACKE_H


#include <ceres/ceres.h>

#include "Optimization/FrameParameterization.h"
#include "vloam/Frame.h"
#include "vloam/Keyframe.h"
#include "sophus/se3.hpp"

namespace vloam
{
class TwoFramePhotometricFunction: public ceres::CostFunction
{
public:
    static const int patch_halfsize_ = 2;
    static const int patch_size_ = 2 * patch_halfsize_;
    static const int patch_area_ = patch_size_ * patch_size_;
    static const int pattern_length_ = 8;
    int pattern_[8][2] = {{0, 0}, {2, 0}, {1, 1}, {0, -2}, {-1, -1}, {-2, 0}, {-1, 1}, {0, 2}};

    static constexpr double eps = 1e-8;
public:
    explicit TwoFramePhotometricFunction(Keyframe::Ptr reference, Frame::Ptr current);

    bool Evaluate(double const *const *parameters, double *residual, double **jacobians) const;

private:
    inline double build_LinearSystem(const Sophus::SE3f &model) const;
    inline double compute_residuals(const Sophus::SE3f &transformation) const;
    inline void precompute_patches(cv::Mat &img, PointCloud &pointcloud, cv::Mat &patch_buf, bool is_derivative) const;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Keyframe::Ptr reference_;
    Frame::Ptr current_;

    CameraModel::Ptr camera_;
    PinholeModel::Ptr pinhole_model_;

    mutable bool is_precomputed_ = false;
    mutable int current_level_ = 2;

    mutable Eigen::Matrix<float, 6, 6> H_;
    mutable Eigen::Matrix<float, 6, 1> Jres_;
    mutable Eigen::Matrix<float, 6, 1> x_;

    mutable cv::Mat ref_patch_buf_, cur_patch_buf_;
    mutable Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::ColMajor> dI_buf_;
    mutable Eigen::Matrix<float, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_buf_;

    mutable std::vector<float> errors_;
    mutable std::vector<Vector6, Eigen::aligned_allocator<Vector6>> J_;
    mutable std::vector<float> weight_;

    mutable float affine_a_ = 1.0f;
    mutable float affine_b_ = 0.0f;

    // weight function
    bool use_weight_scale_;
    mutable float scale_;
    ScaleEstimator::Ptr scale_estimator_;
    WeightFunction::Ptr weight_function_;
    void set_weightfunction();

};




class FrameTracker
{

public:
    ceres::Problem problem;
};
} // namespace vloam

#endif //FRAMETRACKE_H
