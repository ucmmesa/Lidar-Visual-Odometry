#pragma once

#include <algorithm>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include <vloam/CameraModel.h>
#include <vloam/Config.h>
#include <vloam/Datatypes.h>
#include <vloam/Frame.h>
#include <vloam/Keyframe.h>
#include <vloam/LSQNonlinear.hpp>
#include <vloam/PinholeModel.h>
#include <vloam/Twist.h>
#include <vloam/WeightFunction.h>

namespace vloam {

class WeightFunction;

class Tracker2
    : public LSQNonlinearGaussNewton<
          6, Transformf> // LSQNonlinearGaussNewton <6, Sophus::SE3f>
                         // LSQNonlinearLevenbergMarquardt <6, Sophus::SE3f>
{
  static const int patch_halfsize_ = 2;
  static const int patch_size_ = 2 * patch_halfsize_;
  static const int patch_area_ = patch_size_ * patch_size_;

#if 0
  static const int pattern_length_ = 8;
  // clang-format off
      int pattern_[8][2] = {{0, 0}, {2, 0}, {1, 1}, {0, -2}, {-1, -1}, {-2,
      0}, {-1, 1}, {0, 2}};
  // clang-format on
#else
  static const int pattern_length_ = 4;
  // clang-format off
  int pattern_[8][2] = {{1, -1},  {1, 1},  {-1, -1} ,{-1, 1}};
  // clang-format on
#endif

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<Tracker2> Ptr;

  Tracker2();
  ~Tracker2();

  bool tracking(Keyframe::Ptr reference, Frame::Ptr current,
                Transformf &transformation);

private:
  int current_level_;

  int min_level_;
  int max_level_;

  CameraModel::Ptr camera_;
  PinholeModel::Ptr pinhole_model_;

  Sophus::SE3f Tji_;

  Keyframe::Ptr reference_;
  Frame::Ptr current_;

  bool is_precomputed_;
  cv::Mat ref_patch_buf_, cur_patch_buf_;
  Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::ColMajor> dI_buf_;
  Eigen::Matrix<float, 6, Eigen::Dynamic, Eigen::ColMajor> jacobian_buf_;

  std::vector<float> errors_;
  std::vector<Vector6, Eigen::aligned_allocator<Vector6>> J_;
  std::vector<float> weight_;

  float affine_a_;
  float affine_b_;

  void precompute_patches(cv::Mat &img, PointCloud &pointcloud,
                          cv::Mat &patch_buf, bool is_derivative);
  double compute_residuals(const Transformf &transformation);

  // implementation for LSQNonlinear class
  virtual void update(const Transformf &old_model, Transformf &new_model);

public:
  // weight function
  bool use_weight_scale_;
  float scale_;
  ScaleEstimator::Ptr scale_estimator_;
  WeightFunction::Ptr weight_function_;
  void set_weightfunction();
  void max_level(int level) { max_level_ = level; }

protected:
  virtual double build_LinearSystem(Transformf &model);
};
} // namespace vloam
