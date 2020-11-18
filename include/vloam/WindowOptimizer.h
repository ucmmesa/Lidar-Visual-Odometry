//
// Created by ubuntu on 2020/5/19.
//

#ifndef WINDOWOPTIMIZER_H
#define WINDOWOPTIMIZER_H

#include <Eigen/Core>
#include <vloam/Config.h>
#include <vloam/KeyframeWindow.h>
#include <vloam/LSQNonlinear.hpp>

namespace vloam {
class KeyframeWindow;
} // namespace vloam

namespace vloam {
class WindowOptimizer {
  static const int patch_halfsize_ = 2;
  static const int patch_size_ = 2 * patch_halfsize_;
  static const int patch_area_ = patch_halfsize_ * patch_halfsize_;

#if 0
    static const int pattern_length_ = 8;
    int pattern_[8][2] = {{0, 0}, {2, 0}, {1, 1}, {0, -2}, {-1, -1}, {-2, 0}, {-1, 1}, {0, 2}};
#else
  static const int pattern_length_ = 4;
  // clang-format off
  int pattern_[8][2] = {{1, -1},  {1, 1},  {-1, -1} ,{-1, 1}};
#endif
public:

    typedef std::shared_ptr<WindowOptimizer> Ptr;

    WindowOptimizer(KeyframeWindow::Ptr kf_window);
    ~WindowOptimizer();

    bool refine();
    bool solve();
    void update();

    void precompute_patches(
        cv::Mat &img,
        PointCloud &pointcloud,
        cv::Mat &patch_buf,
        Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::ColMajor> &jacobian_buf,
        bool is_derivative);

    void compute_residuals(
        Keyframe::Ptr keyframe_t,
        Keyframe::Ptr keyframe_h,
        std::vector<float> &residuals,
        Eigen::vector<Matrix1x6> &frame_jacobian_t,
        Eigen::vector<Matrix1x6> &frame_jacobian_h);

protected:
    double build_LinearSystem();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    CameraModel::Ptr camera_;
    PinholeModel::Ptr pinhole_model_;

    int iter_;
    int max_iteration_;
    bool stop_;
    double eps_;

    int current_level_=1;

    int num_keyframe_ = 5;
    KeyframeWindow::Ptr kf_window_;

    // weight function
    bool use_weight_scale_;
    float scale_;
    ScaleEstimator::Ptr scale_estimator_ = nullptr;
    WeightFunction::Ptr weight_function_ = nullptr;
    void set_weightfunction();

    int n_measurement_;
    double chi2_;
    double residual_;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> H_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Jres_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> x_;
}; // class WindowOptimizer
} // namespace vloam

#endif //WINDOWOPTIMIZER_H
