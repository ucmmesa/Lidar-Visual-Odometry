#include <aloam_velodyne/utility.h>
#include <sophus/types.hpp>
#include <vloam/Tracker2.h>
namespace vloam
{

Tracker2::Tracker2()
{
    std::cout << "Tracker2 constructor" << std::endl;
    std::cout << "Step1" << std::endl;
    camera_ = Config::cfg()->camera();
    std::cout << "Step2" << std::endl;
    pinhole_model_ = static_pointer_cast<vloam::PinholeModel>(camera_);
    std::cout << "Step3" << std::endl;
    auto tracker_info = Config::cfg()->tracker();
    std::cout << "Step4" << std::endl;
    min_level_ = tracker_info.min_level;
    max_level_ = tracker_info.max_level;

    use_weight_scale_ = tracker_info.use_weight_scale;

    std::cout << "Step5" << std::endl;
    set_weightfunction();
    std::cout << "Step6" << std::endl;
}

Tracker2::~Tracker2()
{
}

void Tracker2::set_weightfunction()
{
    auto tracker_info = Config::cfg()->tracker();
    if (use_weight_scale_)
    {
        switch (tracker_info.scale_estimator_type)
        {
        case ScaleEstimatorType::TDistributionScale:
            scale_estimator_.reset(new TDistributionScaleEstimator());
            break;
        default:
            cerr << "Do not use scale estimator." << endl;
        }
    }

    switch (tracker_info.weight_function_type)
    {
    case WeightFunctionType::TDistributionWeight:
        weight_function_.reset(new TDistributionWeightFunction());
        break;
    default:
        cerr << "Do not use weight function." << endl;
    }
}

/// reference: 参考帧
/// current: 当前帧
/// transformation: 参考帧到到当前帧的位姿变换 T_cur_ref

bool Tracker2::tracking(Keyframe::Ptr reference, Frame::Ptr current, Transformf &transformation)
{
    bool status = true;

    reference_ = reference;
    current_ = current;

    affine_a_ = 1.0f;
    affine_b_ = 0.0f;

    // 金字塔式分层优化
    for (current_level_ = max_level_; current_level_ >= min_level_; current_level_--)
    {
        is_precomputed_ = false;
        stop_ = false;
        //        mu_ = 0.1f;

        optimize(transformation);
    }

    return status;
}

void Tracker2::update(const Transformf &old_model, Transformf &new_model)
{

    Sophus::Vector6f delta = Sophus::Vector6f::Zero();
    //    new_model = old_model * Sophus::SE3f::exp(-x_);
    // new_model = old_model * Sophus::SE3f::exp(-x_);

    new_model = Transformf::se3exp(x_) * old_model;

    //    Eigen::Matrix3f R =new_model.rotationMatrix();
    //    Eigen::Vector3d ypr = Utility::R2ypr(R.cast<double>());
    //    ypr[0] = -ypr[0];
    //    ypr[1] = 0.0;
    //    ypr[2] = -ypr[2];
    //    Eigen::Matrix3d newR = Utility::ypr2R(ypr);
    //    Eigen::Vector3d pos = newR * new_model.translation().cast<double>();
    //    newR = newR*R.cast<double>();

    //    Sophus::SE3f T(newR.cast<float>(), pos.cast<float>());
    //    new_model = T;

    //    cerr << "[Tracker2]\t The model was updated." << endl;
    //    cerr << new_model.matrix() << endl << endl;
}

void Tracker2::precompute_patches(cv::Mat &img, PointCloud &pointcloud, cv::Mat &patch_buf, bool is_derivative)
{
    const int border = patch_halfsize_ + 2 + 2;
    const int stride = img.cols;
    const float scale = 1.0f / (1 << current_level_);

    cv::Mat zbuf;
    zbuf = cv::Mat::zeros(img.size(), CV_8U);

    vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>
        uv_set = camera_->pointcloud_to_uv(pointcloud, scale);

    patch_buf = cv::Mat(pointcloud.size(), pattern_length_, CV_32F);

    if (is_derivative)
    {
        dI_buf_.resize(Eigen::NoChange, patch_buf.rows * pattern_length_);
        jacobian_buf_.resize(Eigen::NoChange, patch_buf.rows * pattern_length_);

        jacobian_buf_.setZero();
    }

    auto pc_iter = pointcloud.begin();
    size_t point_counter = 0;

    for (auto uv_iter = uv_set.begin(); uv_iter != uv_set.end(); ++uv_iter, ++pc_iter, ++point_counter)
    {
        Eigen::Vector2f &uv = *uv_iter;
        float &u_f = uv(0);
        float &v_f = uv(1);
        const int u_i = static_cast<int>(u_f);
        const int v_i = static_cast<int>(v_f);

        if (u_i - border < 0 || u_i + border > img.cols || v_i - border < 0 || v_i + border > img.rows || pc_iter->z <= 0.0)
        {
            float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;
            for (int i = 0; i < pattern_length_; ++i, ++patch_buf_ptr)
                *patch_buf_ptr = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        const float subpix_u = u_f - u_i;
        const float subpix_v = v_f - v_i;
        const float w_tl = (1.0 - subpix_u) * (1.0 - subpix_v);
        const float w_tr = subpix_u * (1.0 - subpix_v);
        const float w_bl = (1.0 - subpix_u) * subpix_v;
        const float w_br = subpix_u * subpix_v;

        size_t pixel_counter = 0;

        float *patch_buf_ptr = reinterpret_cast<float *>(patch_buf.data) + pattern_length_ * point_counter;

        // 预先计算Juv_JP
        Matrix2x6 frame_jac;
        if (is_derivative)
        {
            Eigen::Vector3f xyz(pc_iter->x, pc_iter->y, pc_iter->z);
            Frame::jacobian_xyz2uv(xyz, frame_jac);
        }

        for (int i = 0; i < pattern_length_; ++i, ++pixel_counter, ++patch_buf_ptr)
        {
            int x = pattern_[i][0];
            int y = pattern_[i][1];

            float *img_ptr = (float *)img.data + (v_i + y) * stride + (u_i + x);
            *patch_buf_ptr =
                w_tl * img_ptr[0] + w_tr * img_ptr[1] + w_bl * img_ptr[stride] + w_br * img_ptr[stride + 1];

            if (is_derivative)
            {
                // precompute image gradient
                float dx = 0.5f * ((w_tl * img_ptr[1] + w_tr * img_ptr[2] + w_bl * img_ptr[stride + 1] + w_br * img_ptr[stride + 2]) - (w_tl * img_ptr[-1] + w_tr * img_ptr[0] + w_bl * img_ptr[stride - 1] + w_br * img_ptr[stride]));
                float dy = 0.5f * ((w_tl * img_ptr[stride] + w_tr * img_ptr[1 + stride] + w_bl * img_ptr[stride * 2] + w_br * img_ptr[stride * 2 + 1]) - (w_tl * img_ptr[-stride] + w_tr * img_ptr[1 - stride] + w_bl * img_ptr[0] + w_br * img_ptr[1]));

                // Matrix2x6 frame_jac;
                // Eigen::Vector3f xyz(pc_iter->x, pc_iter->y, pc_iter->z);
                // Frame::jacobian_xyz2uv(xyz, frame_jac);

                Eigen::Vector2f dI_xy(dx, dy);
                dI_buf_.col(point_counter * pattern_length_ + i) = dI_xy;
                // TODO: 这个操作没有看懂,好像将雅克比矩阵进行了尺度缩放,为了啥
                jacobian_buf_.col(point_counter * pattern_length_ + pixel_counter) =
                    (dx * pinhole_model_->fx() * frame_jac.row(0) + dy * pinhole_model_->fy() * frame_jac.row(1)) / (1 << current_level_);
            }
        }
    }
}

double Tracker2::compute_residuals(const Transformf &transformation)
{
    errors_.clear();
    J_.clear();
    weight_.clear();

    if (!is_precomputed_)
    {
        cv::Mat &reference_img = reference_->frame()->level(current_level_);
        PointCloud &pointcloud_ref = reference_->pointcloud();

        precompute_patches(reference_img, pointcloud_ref, ref_patch_buf_, true);

        is_precomputed_ = true;
    }

    cv::Mat &current_img = current_->level(current_level_);
    PointCloud pointcloud_cur;
    pcl::transformPointCloud(reference_->pointcloud(), pointcloud_cur, transformation.matrix());

    precompute_patches(current_img, pointcloud_cur, cur_patch_buf_, false);

    cv::Mat errors = cv::Mat(pointcloud_cur.size(), pattern_length_, CV_32F);
#if 0
    errors = cur_patch_buf_ - (affine_a_ * ref_patch_buf_ + affine_b_);
#else
    errors = cur_patch_buf_ - ref_patch_buf_;
#endif

    scale_ = scale_estimator_->compute(errors);

    float chi2 = 0.0f;
    size_t n_measurement = 0;

    float *errors_ptr = errors.ptr<float>();
    float *ref_patch_buf_ptr = ref_patch_buf_.ptr<float>();
    float *cur_patch_buf_ptr = cur_patch_buf_.ptr<float>();

    float IiIj = 0.0f;
    float IiIi = 0.0f;
    float sum_Ii = 0.0f;
    float sum_Ij = 0.0f;

    for (int i = 0; i < errors.size().area(); ++i, ++errors_ptr, ++ref_patch_buf_ptr, ++cur_patch_buf_ptr)
    {

        float &res = *errors_ptr;

        float &Ii = *ref_patch_buf_ptr;
        float &Ij = *cur_patch_buf_ptr;

        //        if(std::isfinite(error)  && fabs(error) < 1.0f ) {
        // 这一块应该可以加速, 有公因子
        if (std::isfinite(res))
        {
            //            errors_.push_back(res);

            n_measurement++;

            Vector6 J(jacobian_buf_.col(i));

            errors_.push_back(res);
            J_.push_back(J);

            IiIj += Ii * Ij;
            IiIi += Ii * Ii;
            sum_Ii += Ii;
            sum_Ij += Ij;
            //            float weight = 1.0;
            //            weight = weight_function_->weight(res/scale_);
            //            weight_.push_back(weight);
            //            chi2 += res*res*weight;
        }
    }

    affine_a_ = IiIj / IiIi;
    affine_b_ = (sum_Ij - affine_a_ * sum_Ii) / n_measurement;

//    cerr << affine_a_ << ", " << affine_b_ << endl;

    vector<float> sorted_errors;
    sorted_errors.resize(errors_.size());
    copy(errors_.begin(), errors_.end(), sorted_errors.begin());
    sort(sorted_errors.begin(), sorted_errors.end());

    float median_mu = sorted_errors[sorted_errors.size() / 2];

    vector<float> absolute_res_error;
    for (auto error : errors_)
    {
        absolute_res_error.push_back(fabs(error - median_mu));
    }
    sort(absolute_res_error.begin(), absolute_res_error.end());
    // 绝对中位数和标准差之间的转化关系
    float median_abs_deviation = 1.4826 * absolute_res_error[absolute_res_error.size() / 2];

    //    scale_ = scale_estimator_->compute(absolute_res_error);
    for (auto error : errors_)
    {
        float weight = 1.0;
        weight = weight_function_->weight((error - median_mu) / median_abs_deviation);
        //        weight = weight_function_->weight(error/scale_);
        weight_.push_back(weight);

        chi2 += error * error * weight;
    }

    //    cerr << scale_ << ", " << n_measurement << "/" << errors.size().area() << endl;
    return chi2 / n_measurement;
}

double Tracker2::build_LinearSystem(Transformf &model)
{
    double res = compute_residuals(model);

    H_.setZero();
    Jres_.setZero();

    Matrix6x6 AdjT = model.inverse().SE3Adj();
    for (int i = 0; i < errors_.size(); ++i)
    {
        float &res = errors_[i];
        Vector6 &J = J_[i];
        float &weight = weight_[i];

        H_.noalias() += J * J.transpose() * weight;
        Jres_.noalias() -= J * res * weight;
    }

    H_ = AdjT.transpose() * H_ * AdjT;
    Jres_ = -AdjT.transpose() * Jres_;


    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> SVD((H_+ H_.transpose())*0.5);
    Eigen::VectorXf S = Eigen::VectorXf((SVD.eigenvalues().array() > 1e-8).select(SVD.eigenvalues().array(), 0));
    Eigen::VectorXf S_inv = Eigen::VectorXf((SVD.eigenvalues().array() > 1e-8).select(SVD.eigenvalues().array().inverse(), 0));

    Eigen::VectorXf S_sqrt = S.cwiseSqrt();
    Eigen::VectorXf S_inv_sqrt = S_inv.cwiseSqrt();
    //其实可以使用LLT分解求解
    // U*Singma^{1/2}Sigma^{1/2}*U^T = A
    // J^T = U*Singma^{1/2}
    // J^T * error = b
    // J = Sigma^{1/2}*U^T
    // U*Sigma^{1/2}*error = b
    // Sigma^{1/2} * error = U^T * b
    // error = Sigma^{-1/2}* U^T *b
    Eigen::MatrixXf linearized_jacobians = S_sqrt.asDiagonal() * SVD.eigenvectors().transpose();
    Eigen::VectorXf linearized_residuals = S_inv_sqrt.asDiagonal() * SVD.eigenvectors().transpose() * Jres_.cast<float>();

    H_ = linearized_jacobians.transpose()*linearized_jacobians;
    Jres_ = linearized_jacobians.transpose()*linearized_residuals;
    // std::cout << "H: \n"
    //           << H_ << std::endl;
    // y加约束
    //    Sophus::Vector6f se3 = model.log();
    //    H_(1,1) = 100.;
    //    Jres_(1,0) = -10.0*se3[1];
    //    H_(3,3) = 100.;
    //    Jres_(3,0) = -10.0*se3[3];
    //    H_(5,5) = 100.;
    //    Jres_(5,0) = -10.0*se3[5];
    return res;
}

} // namespace vloam
