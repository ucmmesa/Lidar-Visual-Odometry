//
// Created by ubuntu on 2020/5/19.
//

#include "aloam_velodyne/utility.h"
#include "vloam/WindowOptimizer.h"
#include "vloam/accumulator.h"

#include "vloam/Twist.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

namespace vloam
{
WindowOptimizer::WindowOptimizer(KeyframeWindow::Ptr kf_window)
{
    this->iter_ = 0;
    this->max_iteration_ = 100;
    this->stop_ = false;
    this->eps_ = 1e-10;
    this->n_measurement_ = 0;
    this->chi2_ = 0.0f;

    this->kf_window_ = kf_window;

    this->camera_ = Config::cfg()->camera();
    pinhole_model_ = static_pointer_cast<vloam::PinholeModel>(camera_);

    current_level_ = 0;

    auto tracker_info = Config::cfg()->tracker();

    use_weight_scale_ = tracker_info.use_weight_scale;

    set_weightfunction();

}

WindowOptimizer::~WindowOptimizer()
{

}

void WindowOptimizer::set_weightfunction()
{
    auto tracker_info = Config::cfg()->tracker();
    if (use_weight_scale_) {
        switch (tracker_info.scale_estimator_type) {
        case ScaleEstimatorType::TDistributionScale:scale_estimator_.reset(new TDistributionScaleEstimator());
            break;
        default:std::cerr << "Do not use scale estimator." << std::endl;
        }
    }

    switch (tracker_info.weight_function_type) {
    case WeightFunctionType::TDistributionWeight:weight_function_.reset(new TDistributionWeightFunction());
        break;

    default:std::cerr << "Do not use weight function." << std::endl;
    }
}

bool WindowOptimizer::refine()
{
    LOG(INFO) << "[WindowOptimizer2]\t called refine()";

    // step1: 获取滑窗中关键帧的数目
    num_keyframe_ = kf_window_->size();

    // step2: 申请一个新的容器存储关键帧的位姿
    Eigen::vector<Transformf> old_T;
    old_T.reserve(num_keyframe_);

    auto kf_window = kf_window_->frames();
    for (int i = 0; i < num_keyframe_; i++) {
        old_T[i] = kf_window[i]->frame()->Twc();
    }

    // step3: 重置hessian 矩阵的size
    H_.resize(6 * num_keyframe_, 6 * num_keyframe_);
    Jres_.resize(6 * num_keyframe_, 1);
    x_.resize(6 * num_keyframe_, 1);

    this->stop_ = false;

    for (iter_ = 0; iter_ < max_iteration_; ++iter_) {
        // hessian 矩阵重置为0
        H_.setZero();
        Jres_.setZero();

        n_measurement_ = 0;
        chi2_ = 0.0f;

        build_LinearSystem();

        // 计算平均误差
        double new_residual = chi2_ / n_measurement_;

        // 求解正规方程
        if (!solve()) {
            // 如果求解没有成功,
            stop_ = true;
            LOG(INFO) << "x_ is NaN: " << x_.array().isNaN();
        }

        if ((iter_ > 0 and new_residual > residual_) || stop_) {
            // 如果本次迭代得到的误差比上一次的误差比较大或者判断为迭代停止,
            // 那么相机的位姿使用上一时刻的位姿
            for (int i = 0; i < num_keyframe_; i++) {
                kf_window[i]->frame()->Twc(old_T[i]);
            }
            break;
        }

        update();

        // 保存上一次迭代的关键帧的位姿和和误差
        for (int i = 0; i < num_keyframe_; i++) {
            old_T[i] = kf_window[i]->frame()->Twc();
        }

        residual_ = new_residual;

        // 如果出现NaN的情况,跌打停止
        if (((x_ - x_).array() != (x_ - x_).array()).all()) {
            break;
        }
        else {
            // 如果增量的绝对值的最大值小于设定的阈值，那么迭代停止
            double max = -1;
            for (int i = 0; i < x_.size(); i++) {
                double abs = fabs(x_(i, 0));
                if (abs > max)
                    max = abs;
            }

            if (max < eps_)
                break;
        }
    }

    LOG(INFO) << "[WindowOptimzer2]\t " << iter_;
}

void WindowOptimizer::update()
{
    // 左扰动更新
    try {
        auto kf_window = kf_window_->frames();

        for (int i = 0; i < num_keyframe_; i++) {
            Transformf tmp = Transformf::se3exp(x_.block<6, 1>(6 * i, 0));
            auto new_Ti = tmp * kf_window[i]->frame()->Twc();
            kf_window[i]->frame()->Twc(new_Ti);
        }
    }
    catch (int exception) {
        LOG(ERROR) << x_;
    }
} // update end

bool WindowOptimizer::solve()
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
    Eigen::Matrix<double, Eigen::Dynamic, 1> b;
    Eigen::Matrix<double, Eigen::Dynamic, 1> xx;

    A.resize(6 * num_keyframe_, 6 * num_keyframe_);
    b.resize(6 * num_keyframe_, 1);
    xx.resize(6 * num_keyframe_, 1);

    A = H_.cast<double>();
    b = Jres_.cast<double>();

    A.block<6, 6>(0, 0) += Eigen::Matrix<double, 6, 6>::Identity(6, 6)
        * (static_cast<double>(n_measurement_) * static_cast<double>(n_measurement_) * 10000000.0);

    xx = A.ldlt().solve(b);

    x_ = xx.cast<float>();

    return ((x_ - x_).array() == (x_ - x_).array()).all();
} //solve

void WindowOptimizer::precompute_patches(
    cv::Mat &img,
    PointCloud &pointcloud,
    cv::Mat &patch_buf,
    Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::ColMajor> &jacobian_buf,
    bool is_derivative)
{
    const int border = patch_halfsize_ + 2 + 2;
    const int stride = img.cols;
    const float scale = 1.0f / (1 << current_level_);

    Eigen::vector<Eigen::Vector2f> uv_set = camera_->pointcloud_to_uv(pointcloud, scale);

    patch_buf = cv::Mat(pointcloud.size(), pattern_length_, CV_32F);

    if (is_derivative) {
        jacobian_buf.resize(Eigen::NoChange, patch_buf.rows * pattern_length_);
        jacobian_buf.setZero();
    }
    auto pc_iter = pointcloud.begin();
    size_t point_counter = 0;

#if 1
    /// 多线程
    auto compute_func = [&](const tbb::blocked_range<Eigen::vector<Eigen::Vector2f>::iterator> &range)
    {
        for (auto uv_iter = range.begin(); uv_iter != range.end(); ++uv_iter, ++pc_iter, ++point_counter) {
            Eigen::Vector2f &uv = *uv_iter;

            float &u_f = uv[0];
            float &v_f = uv[1];
            const int u_i = static_cast<int>(u_f);
            const int v_i = static_cast<int>(v_f);

            if (u_i - border < 0 || u_i + border > img.cols || v_i - border < 0 || v_i + border > img.rows
                || pc_iter->z <= 0.0) {
                float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

                for (int i = 0; i < pattern_length_; ++i, ++patch_buf_ptr)
                    *patch_buf_ptr = std::numeric_limits<float>::quiet_NaN();

                continue;
            }

            const float subpix_u = u_f - u_i;
            const float subpix_v = v_f - v_i;
            const float w_tl = (1.0f - subpix_u) * (1.0f - subpix_v);
            const float w_tr = subpix_u * (1.0f - subpix_v);
            const float w_bl = (1.0f - subpix_u) * subpix_v;
            const float w_br = subpix_u * subpix_v;

            size_t pixel_counter = 0;
            float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

            for (int i = 0; i < pattern_length_; ++i, ++pixel_counter, ++patch_buf_ptr) {
                int x = pattern_[i][0];
                int y = pattern_[i][1];

                float *img_ptr = (float *) img.data + (v_i + y) * stride + (u_i + x);
                *patch_buf_ptr =
                    w_tl * img_ptr[0] + w_tr * img_ptr[1] + w_bl * img_ptr[stride] + w_br * img_ptr[stride + 1];

                if (is_derivative) {
                    // precompute image gradient
                    float dx = 0.5f
                        * ((w_tl * img_ptr[1] + w_tr * img_ptr[2] + w_bl * img_ptr[stride + 1]
                            + w_br * img_ptr[stride + 2])
                            - (w_tl * img_ptr[-1] + w_tr * img_ptr[0] + w_bl * img_ptr[stride - 1]
                                + w_br * img_ptr[stride]));
                    float dy = 0.5f * ((w_tl * img_ptr[stride] + w_tr * img_ptr[1 + stride] + w_bl * img_ptr[stride * 2]
                        + w_br * img_ptr[stride * 2 + 1])
                        - (w_tl * img_ptr[-stride] + w_tr * img_ptr[1 - stride] + w_bl * img_ptr[0]
                            + w_br * img_ptr[1]));

                    Matrix2x6 frame_jac;
                    Eigen::Vector3f xyz(pc_iter->x, pc_iter->y, pc_iter->z);
                    Matrix1x3 frame_jacobian;

                    auto fx = pinhole_model_->fx();
                    auto fy = pinhole_model_->fy();

                    frame_jacobian << dx * fx / pc_iter->z,
                        dy * fy / pc_iter->z,
                        (-1 * dx * fx * pc_iter->x - 1 * dy * fy * pc_iter->y) / (pc_iter->z * pc_iter->z);

                    jacobian_buf.col(point_counter * pattern_length_ + i) = frame_jacobian;
                }
            } // 遍历每一个投影点周围的pattern
        } // 遍历每一个投影点
    };

    tbb::blocked_range<Eigen::vector<Eigen::Vector2f>::iterator> range(uv_set.begin(), uv_set.end());
    tbb::parallel_for(range, compute_func);

#else
    /// 单线程
    for (auto uv_iter = uv_set.begin(); uv_iter != uv_set.end(); ++uv_iter, ++pc_iter, ++point_counter) {
        Eigen::Vector2f &uv = *uv_iter;

        float &u_f = uv[0];
        float &v_f = uv[1];
        const int u_i = static_cast<int>(u_f);
        const int v_i = static_cast<int>(v_f);

        if (u_i - border < 0 || u_i + border > img.cols || v_i - border < 0 || v_i + border > img.rows
            || pc_iter->z <= 0.0) {
            float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

            for (int i = 0; i < pattern_length_; ++i, ++patch_buf_ptr)
                *patch_buf_ptr = std::numeric_limits<float>::quiet_NaN();

            continue;
        }

        const float subpix_u = u_f - u_i;
        const float subpix_v = v_f - v_i;
        const float w_tl = (1.0f - subpix_u) * (1.0f - subpix_v);
        const float w_tr = subpix_u * (1.0f - subpix_v);
        const float w_bl = (1.0f - subpix_u) * subpix_v;
        const float w_br = subpix_u * subpix_v;

        size_t pixel_counter = 0;
        float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

        for (int i = 0; i < pattern_length_; ++i, ++pixel_counter, ++patch_buf_ptr) {
            int x = pattern_[i][0];
            int y = pattern_[i][1];

            float *img_ptr = (float *) img.data + (v_i + y) * stride + (u_i + x);
            *patch_buf_ptr =
                w_tl * img_ptr[0] + w_tr * img_ptr[1] + w_bl * img_ptr[stride] + w_br * img_ptr[stride + 1];

            if (is_derivative) {
                // precompute image gradient
                float dx = 0.5f
                    * ((w_tl * img_ptr[1] + w_tr * img_ptr[2] + w_bl * img_ptr[stride + 1] + w_br * img_ptr[stride + 2])
                        - (w_tl * img_ptr[-1] + w_tr * img_ptr[0] + w_bl * img_ptr[stride - 1]
                            + w_br * img_ptr[stride]));
                float dy = 0.5f * ((w_tl * img_ptr[stride] + w_tr * img_ptr[1 + stride] + w_bl * img_ptr[stride * 2]
                    + w_br * img_ptr[stride * 2 + 1])
                    - (w_tl * img_ptr[-stride] + w_tr * img_ptr[1 - stride] + w_bl * img_ptr[0] + w_br * img_ptr[1]));

                Matrix2x6 frame_jac;
                Eigen::Vector3f xyz(pc_iter->x, pc_iter->y, pc_iter->z);
                Matrix1x3 frame_jacobian;

                auto fx = pinhole_model_->fx();
                auto fy = pinhole_model_->fy();

                frame_jacobian << dx * fx / pc_iter->z,
                    dy * fy / pc_iter->z,
                    (-1 * dx * fx * pc_iter->x - 1 * dy * fy * pc_iter->y) / (pc_iter->z * pc_iter->z);

                jacobian_buf.col(point_counter * pattern_length_ + i) = frame_jacobian;
            }
        } // 遍历每一个投影点周围的pattern
    } // 遍历每一个投影点
#endif

} // precompute_patches


void WindowOptimizer::compute_residuals(
    Keyframe::Ptr keyframe_t,
    Keyframe::Ptr keyframe_h,
    std::vector<float> &residuals,
    Eigen::vector<Matrix1x6> &frame_jacobian_t,
    Eigen::vector<Matrix1x6> &frame_jacobian_h)
{
    cv::Mat img_h = keyframe_h->frame()->level(current_level_);
    PointCloud &pointcloud_h = keyframe_h->pointcloud();

    cv::Mat patch_buf_h;

    Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::ColMajor> jacobian_buf_h;

    precompute_patches(img_h, pointcloud_h, patch_buf_h, jacobian_buf_h, false);

    Transformf T_t = keyframe_t->frame()->Twc();
    Transformf T_h = keyframe_h->frame()->Twc();
    Transformf T_t_h = T_t.inverse() * T_h;

    cv::Mat img_t = keyframe_t->frame()->level(current_level_);
    PointCloud pointCloud_t;

    pcl::transformPointCloud(pointcloud_h, pointCloud_t, T_t_h.matrix());

    cv::Mat patch_buf_t;

    Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::ColMajor> jacobian_buf_t;

    precompute_patches(img_t, pointCloud_t, patch_buf_t, jacobian_buf_t, true);

    cv::Mat errors = patch_buf_t - patch_buf_h;

    // 其实这个做个一个假设,整幅图像的内存空间是连续的
    float *errors_ptr = errors.ptr<float>();



    // TODO:　使用TBB 将下面的代码改为并行程序
#if 1
    int k = 0;
    tbb::concurrent_vector<float> concurrent_residual;
    tbb::concurrent_vector<Matrix1x6> concurrent_frame_jacobian_t;
    tbb::concurrent_vector<Matrix1x6> concurrent_frame_jacobian_h;

    concurrent_residual.reserve(pointcloud_h.size());
    concurrent_frame_jacobian_t.reserve(pointcloud_h.size());
    concurrent_frame_jacobian_h.reserve(pointcloud_h.size());
    // 定义匿名函数
    auto compute_func = [&](const tbb::blocked_range<PointCloud::iterator> &range)
    {
        for (auto iter = range.begin(); iter != range.end(); ++iter, ++k) {
            for (int i = 0; i < pattern_length_; ++i) {
                int buf_idx = k * pattern_length_ + i;

                float &res = *(errors_ptr + buf_idx);

                if (std::isfinite(res)) {
                    concurrent_residual.push_back(res);

                    Eigen::Vector3f P_h(iter->x, iter->y, iter->z);
                    Eigen::Vector3f P = T_h * P_h;

                    Matrix3x6 J_P_J_h;
                    J_P_J_h.setZero();
                    J_P_J_h.leftCols(3) = Eigen::Matrix3f::Identity();
                    J_P_J_h(0, 4) = P(2);
                    J_P_J_h(0, 5) = -P(1);
                    J_P_J_h(1, 3) = -P(2);
                    J_P_J_h(1, 5) = P(0);
                    J_P_J_h(2, 3) = P(1);
                    J_P_J_h(2, 4) = -P(0);

                    Matrix3x6 J_P_t_J_h = T_t.inverse().rotationMatrix() * J_P_J_h;
                    Matrix1x6 J_error_J_h = jacobian_buf_t.col(buf_idx).transpose() * J_P_t_J_h;

                    concurrent_frame_jacobian_t.push_back(-J_error_J_h);
                    concurrent_frame_jacobian_h.push_back(J_error_J_h);
                }
            }
        }
    };

    // 开启并行
    tbb::blocked_range<PointCloud::iterator> range(pointcloud_h.begin(), pointcloud_h.end());
    tbb::parallel_for(range, compute_func);

    // 预留空间，避免内存的重新分配
    residuals.reserve(concurrent_residual.size());
    frame_jacobian_t.reserve(concurrent_residual.size());
    frame_jacobian_h.reserve(concurrent_residual.size());

    // 数据类型转换
    residuals.insert(residuals.end(), concurrent_residual.begin(), concurrent_residual.end());
    frame_jacobian_t.insert(frame_jacobian_t.end(), concurrent_frame_jacobian_t.begin(), concurrent_frame_jacobian_t.end());
    frame_jacobian_h.insert(frame_jacobian_h.end(), concurrent_frame_jacobian_h.begin(), concurrent_frame_jacobian_h.end());

#else

    residuals.reserve(pointcloud_h.size());
    frame_jacobian_t.reserve(pointcloud_h.size());
    frame_jacobian_h.reserve(pointcloud_h.size());
    int k = 0;
    for (auto iter = pointcloud_h.begin(); iter != pointcloud_h.end(); ++iter, ++k) {
        for (int i = 0; i < pattern_length_; ++i) {
            int buf_idx = k * pattern_length_ + i;

            float &res = *(errors_ptr + buf_idx);

            if (std::isfinite(res)) {
                residuals.push_back(res);

                Eigen::Vector3f P_h(iter->x, iter->y, iter->z);
                Eigen::Vector3f P = T_h * P_h;

                Matrix3x6 J_P_J_h;
                J_P_J_h.setZero();
                J_P_J_h.leftCols(3) = Eigen::Matrix3f::Identity();
                J_P_J_h(0, 4) = P(2);
                J_P_J_h(0, 5) = -P(1);
                J_P_J_h(1, 3) = -P(2);
                J_P_J_h(1, 5) = P(0);
                J_P_J_h(2, 3) = P(1);
                J_P_J_h(2, 4) = -P(0);

                Matrix3x6 J_P_t_J_h = T_t.inverse().rotationMatrix() * J_P_J_h;

                Matrix1x6 J_error_J_h = jacobian_buf_t.col(buf_idx).transpose() * J_P_t_J_h;
                frame_jacobian_t.push_back(-J_error_J_h);
                frame_jacobian_h.push_back(J_error_J_h);
            }
        }
    }
#endif
} // compute_residuals end

double WindowOptimizer::build_LinearSystem()
{
    auto frames = kf_window_->frames();

    Eigen::vector<Matrix1x6> full_jacobian_h, full_jacobian_t;
    std::vector<float> full_residuals;
    std::vector<std::pair<size_t, size_t>> frame_idx_map;

    for (size_t h = 0; h < frames.size(); ++h) {
        Keyframe::Ptr frame_h = frames[h];

        for (size_t t = 0; t < frames.size(); ++t) {
            if (h != t) {
                std::pair<size_t, size_t> frame_idx_pair = std::make_pair(t, h);

                Keyframe::Ptr frame_t = frames[t];

                std::vector<float> residuals;
                Eigen::vector<Matrix1x6> frame_jacobian_h, frame_jacobian_t;

                compute_residuals(frame_t, frame_h, residuals, frame_jacobian_t, frame_jacobian_h);

                full_residuals.insert(full_residuals.end(), residuals.begin(), residuals.end());
                full_jacobian_t.insert(full_jacobian_t.end(), frame_jacobian_t.begin(), frame_jacobian_t.end());
                full_jacobian_h.insert(full_jacobian_h.end(), frame_jacobian_h.begin(), frame_jacobian_h.end());

                // 存储特征点约束的 host frame 和　target frame　begin
                for (size_t k = 0; k < residuals.size(); ++k) {
                    frame_idx_map.push_back(frame_idx_pair);
                } // 存储特征点约束的 host frame 和 tatget frame end
            } // host frame 和　target frame id号不同的情况
        } // 遍历target frame　end
    } // 遍历host frame end

    // 对误差值进行排序
    std::vector<float> sorted_errors;
    sorted_errors.resize(full_residuals.size());
    std::copy(full_residuals.begin(), full_residuals.end(), sorted_errors.begin());
#if 0
    std::sort(sorted_errors.begin(), sorted_errors.end());
#else
    std::partial_sort(sorted_errors.begin(), sorted_errors.begin() + sorted_errors.size() / 2 + 1, sorted_errors.end());
#endif
    // 获得误差的中值
    float median_mu = sorted_errors[sorted_errors.size() / 2];

    // 计算误差减去误差的中位数中值的绝对值
    std::vector<float> absolute_res_error;
    for (auto error: full_residuals)
        absolute_res_error.push_back(std::abs(error - median_mu));
#if 0
    // 对误差值进行排序
    std::sort(absolute_res_error.begin(), absolute_res_error.end());
#else
    std::partial_sort(absolute_res_error.begin(),
                      absolute_res_error.begin() + absolute_res_error.size() / 2 + 1,
                      absolute_res_error.end());
#endif
    // 计算误差的方差
    float median_abs_deviation = 1.4826f * absolute_res_error[absolute_res_error.size() / 2];

    std::vector<float> weights;

    n_measurement_ = full_residuals.size();

    // 计算每个光度误差的的权重
    for (auto error: full_residuals) {

        float weight = weight_function_->weight((error - median_mu) / median_abs_deviation);
        weights.push_back(weight);

        chi2_ += error * error * weight;
    }

    // TODO:　将所有的误差对应的 jacobian 和 error 转换为 hessain矩阵
    for (size_t k = 0; k < full_residuals.size(); ++k) {
        Matrix12x12 H;
        Vector12 Jres;
        H.setZero();
        Jres.setZero();

        auto frame_idx_pair = frame_idx_map[k];
        size_t t_id = frame_idx_pair.first;
        size_t h_id = frame_idx_pair.second;

        float &res = full_residuals[k];
        float &weight = weights[k];
        Matrix1x6 &J_t = full_jacobian_t[k];
        Matrix1x6 &J_h = full_jacobian_h[k];

        Matrix1x12 J;
        J.block<1, 6>(0, 0) = J_t;
        J.block<1, 6>(0, 6) = J_h;

        H = J.transpose() * J * weight;
        Jres = J.transpose() * res * weight;

        Matrix6x6 H_tt = H.block<6, 6>(0, 0);
        Matrix6x6 H_th = H.block<6, 6>(0, 6);
        Matrix6x6 H_ht = H.block<6, 6>(6, 0);
        Matrix6x6 H_hh = H.block<6, 6>(6, 6);

        H_.block<6, 6>(6 * t_id, 6 * t_id).noalias() += H_tt;
        H_.block<6, 6>(6 * t_id, 6 * h_id).noalias() += H_th;
        H_.block<6, 6>(6 * h_id, 6 * t_id).noalias() += H_ht;
        H_.block<6, 6>(6 * h_id, 6 * h_id).noalias() += H_hh;

        Vector6 Jres_t = Jres.block<6, 1>(0, 0);
        Vector6 Jres_h = Jres.block<6, 1>(6, 0);

        Jres_.block<6, 1>(6 * t_id, 0).noalias() -= Jres_t;
        Jres_.block<6, 1>(6 * h_id, 0).noalias() -= Jres_h;
    }

    return chi2_ / n_measurement_;
} // build_linearSystem end

} // namespace vloam