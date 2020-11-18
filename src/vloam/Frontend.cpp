//
// Created by ubuntu on 2020/3/9.
//
#include "vloam/Frontend.h"
#include "Optimization/FrameTracker.h"
#include "vloam/WindowOptimizer.h"
#include <aloam_velodyne/tic_toc.h>
#include <aloam_velodyne/utility.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <string>

bool has_suffix(const std::string &str, const std::string &suffix)
{
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

namespace vloam
{
    Frontend::Frontend()
        : binitialize_(false)
    {
        std::cerr << "[Frontend] \t Called constructor of frontend" << std::endl;
    }

    Frontend::Frontend(const std::string &path, const std::string &fname)
        : binitialize_(false)
    {
        std::cerr << "[Frontend] \t Called constructor of frontend" << std::endl;

        Config::cfg(path, fname);

        auto camera_info = Config::cfg()->camera_info();

        std::cout << "begining load cam-lidar extrinsic" << std::endl;
        extrinsics_ = Config::cfg()->camlidar().extrinsic;
        std::cout << "load cam-lidar extrinsic successfully" << std::endl;

        std::cout << "begining load camera model" << std::endl;
        pcamera_ = Config::cfg()->camera();
        std::cout << "load camera model successfully" << std::endl;

        std::cout << "System constructe featuretracking " << std::endl;
        pfeaturetracker_.reset(new featureTracking());
        kdTree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        std::cout << "System constructe featuretracking successfully" << std::endl;

        std::cout << "System constructe Tracker2 " << std::endl;
        ptracker2_.reset(new Tracker2());
        std::cout << "System constructe Tracker2 successfully" << std::endl;

        // TODO: num_keyframe_ 最好设置为读取参数配置
        num_keyframe_ = 5;
        kf_window_.reset(new KeyframeWindow(num_keyframe_));
        window_optimizer_.reset(new WindowOptimizer(kf_window_));
    }

    Frontend::~Frontend()
    {
        std::cerr << "[System] \t Called destructor of system" << std::endl;
    }

    bool Frontend::track_camlidar(Frame::Ptr current)
    {
        if (!binitialize_)
        {
            Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
            extrinsic.block<3, 4>(0, 0) = extrinsics_;
            Transformf sophus_twc(extrinsic.inverse());
            std::cout << sophus_twc.matrix() << std::endl;

            // create first keyframe
            Keyframe::Ptr current_keyframe(new Keyframe(current, id_manager.id(), pcamera_));
            // set keyframe pose
            current_keyframe->frame()->Twc(sophus_twc);

            // save the new keyframe
            platestKf_ = current_keyframe;
            plastFrame_ = current;
            binitialize_ = true;

            return true;
        }

        // estimate the T_cur_k using constant velocity motion model
        Transformf T_cur_k = dT_cur_last_ * T_last_k_;

        auto tic = TicToc();
#if 1
        btracking_status_ = ptracker2_->tracking(platestKf_, current, T_cur_k);
#else
        Transformd T_cur_kd = T_cur_k.cast<double>();
        std::array<double, 7> parameters;
        std::copy(T_cur_kd.data(), T_cur_kd.data() + 7, parameters.data());

        //step1: create ceres problem
        ceres::Problem problem;
        ceres::LossFunction *loss_function;
        //step2: add paramteres
        ceres::LocalParameterization *localParameterization = new FrameParameterization();
        problem.AddParameterBlock(parameters.data(), 7, localParameterization);
        //step3 create costfuntion factor
        TwoFramePhotometricFunction *DirectFactor = new TwoFramePhotometricFunction(platestKf_, current);
        problem.AddResidualBlock(DirectFactor, nullptr, parameters.data());
        for (int lvl = 3; lvl >= 0; lvl--)
        {

            //        LOG(INFO) << "optimizatio lvl: " << lvl;
            DirectFactor->current_level_ = lvl;
            DirectFactor->is_precomputed_ = false;

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
            //options.num_threads = 2;
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.max_num_iterations = 15;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
        }

        T_cur_kd.rot = Eigen::Map<Eigen::Quaterniond>(parameters.data());
        T_cur_kd.pos = Eigen::Map<Eigen::Vector3d>(parameters.data() + 4);

        T_cur_k = T_cur_kd.cast<float>();
#endif
        // Eigen::Vector3d ypr = Utility::R2ypr(T_cur_k.rotationMatrix().cast<double>());
        // ypr[0] = -ypr[0];
        // ypr[1] = 0;
        // ypr[2] = -ypr[2];
        // Eigen::Matrix3d newR = Utility::ypr2R(ypr);
        // Eigen::Matrix3d newR1 = newR*T_cur_k.rotationMatrix().cast<double>();
        // Eigen::Quaterniond q(newR1);
        // q.normalize();
        // Eigen::Vector3d pose = newR*T_cur_k.translation().cast<double>();
        // Eigen::Quaterniond q1 = Eigen::Quaterniond::FromTwoVectors(pose, Eigen::Vector3d{pose[0],0,pose[2]});
        // pose = q1*pose;
        // Sophus::SE3d T(q, pose);
        // T_cur_k = T.cast<float>();
        std::cerr << "[Frontend] \t Computation time and rate : " << tic.toc() << "," << 1000 / (tic.toc()) << std::endl;

        dT_cur_last_ = T_cur_k * T_last_k_.inverse();
        T_last_cur_ = dT_cur_last_.inverse();
        current->Twc(platestKf_->frame()->Twc() * T_cur_k.inverse());

        // update T_last_k_ , T_k_last_
        T_last_k_ = T_cur_k;
        T_k_last_ = T_cur_k.inverse();

        // keyframe decision
        // auto v_rot_ji = dT_cur_last_.log().block<3, 1>(3, 0).norm();
        // auto v_t_ji = dT_cur_last_.log().block<3, 1>(0, 0).norm();

        float ratio_threshold = 1.0;
        Keyframe::Ptr current_keyframe(new Keyframe(current, pcamera_));

        current_keyframe->show_image_with_points(current_keyframe->frame()->original_img(), 0);

        LOG(INFO) << "keyframe show_image_with_points";

        float visible_ratio1 = platestKf_->get_visible_ratio(current_keyframe);
        float visible_ratio2 = current_keyframe->get_visible_ratio(platestKf_);

        bool is_keyframe =
            (visible_ratio1 < ratio_threshold ? true : false) || ((visible_ratio2 < ratio_threshold ? true : false));
        is_keyframe = true;
        if (is_keyframe)
        {
            // create new keyframe
            current_keyframe->id(id_manager.id());
            platestKf_ = current_keyframe;

            kf_window_->add(current_keyframe);
            // if(kf_window_->size() > 3)
            // {
            //     window_optimizer_->refine();
            // }
            // reset T_last_k_

            Transformf tmp;
            T_last_k_ = T_k_last_ = tmp;
        }

        return true;
    }

    void
    Frontend::trackfeature(double timestamp, const cv::Mat image, pcl::PointCloud<pcl::PointXYZI> &pc)
    {
        // step1 swap depth cloud
        depthCloundLast_.swap(depthCloundCur_);
        depthCloundCur_ = pc;
        // step2 swap feature point
        featurePointLast_.swap(featurePointCur_);
        featurePointCur_ = *(pfeaturetracker_->trackImage(timestamp, image));

        featureTrackResultLast_.swap(featureTrackResultCur_);
        featureTrackResultCur_ = pfeaturetracker_->featureTrackingResultCur();

        // step3 swap feature number
        featurePointLastNum_ = featurePointCurNum_;
        featurePointCurNum_ = featurePointCur_.size();
        // step4 swap feature first observations position and camera pose
        startPointLast_.swap(startPointCur_);
        startTransLast_.swap(startTransCur_);
        // step5 swap feature point depth
        featureDepthLast_.swap(featureDepthCur_);
        LOG(INFO) << "featurePointLast_.size: " << featurePointLast_.size();
        if (!featurePointLast_.empty())
        {
            LOG(INFO) << "second frame";

            pcl::PointXYZI ips;
            pcl::PointXYZHSV ipr;
            ipRelations_.clear();
            ipRelations_.reserve(featurePointLastNum_);
            ipInd_.clear();
            ipInd_.reserve(featurePointLastNum_);
            int depth1_count = 0;
            int depth2_count = 0;
            int depth3_count = 0;
            for (auto lastfeature : featureTrackResultLast_)
            {
                int key = lastfeature.first;
                if (featureTrackResultCur_.find(key) != featureTrackResultCur_.end())
                {
                    ipr.x = lastfeature.second.u;
                    ipr.y = lastfeature.second.v;
                    ipr.z = featureTrackResultCur_.at(key).u;
                    ipr.h = featureTrackResultCur_.at(key).v;

                    ips.x = 10 * ipr.x;
                    ips.y = 10 * ipr.y;
                    ips.z = 10;

                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqrDis;
                    pointSearchInd.clear();
                    pointSearchSqrDis.clear();
                    kdTree_->nearestKSearch(ips, 3, pointSearchInd, pointSearchSqrDis);

                    double minDepth, maxDepth;
                    // use lidar depth map to assign depth to feature
                    if (pointSearchSqrDis[0] < 0.5 && pointSearchInd.size() == 3)
                    {
                        pcl::PointXYZI depthPoint = depthCloundLast_.points[pointSearchInd[0]];
                        double x1 = depthPoint.x * depthPoint.intensity / 10;
                        double y1 = depthPoint.y * depthPoint.intensity / 10;
                        double z1 = depthPoint.intensity;
                        minDepth = z1;
                        maxDepth = z1;

                        depthPoint = depthCloundLast_.points[pointSearchInd[1]];
                        double x2 = depthPoint.x * depthPoint.intensity / 10;
                        double y2 = depthPoint.y * depthPoint.intensity / 10;
                        double z2 = depthPoint.intensity;
                        minDepth = (z2 < minDepth) ? z2 : minDepth;
                        maxDepth = (z2 > maxDepth) ? z2 : maxDepth;

                        depthPoint = depthCloundLast_.points[pointSearchInd[2]];
                        double x3 = depthPoint.x * depthPoint.intensity / 10;
                        double y3 = depthPoint.y * depthPoint.intensity / 10;
                        double z3 = depthPoint.intensity;
                        minDepth = (z3 < minDepth) ? z3 : minDepth;
                        maxDepth = (z3 > maxDepth) ? z3 : maxDepth;

                        double u = ipr.x;
                        double v = ipr.y;
                        ipr.s = (x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1) / (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2 + u * y1 * z2 - u * y2 * z1 - v * x1 * z2 + v * x2 * z1 - u * y1 * z3 + u * y3 * z1 + v * x1 * z3 - v * x3 * z1 + u * y2 * z3 - u * y3 * z2 - v * x2 * z3 + v * x3 * z2);
                        ipr.v = 1;

                        depth1_count++;
                        if (!std::isfinite(ipr.s))
                        {
                            ipr.s = z1;
                            ipr.v = 1;
                            depth1_count--;
                        }
                        if (maxDepth - minDepth > 2)
                        {
                            ipr.s = 0;
                            ipr.v = 0;
                            depth1_count--;
                        }
                        else if (ipr.s - maxDepth > 0.2)
                        {
                            ipr.s = maxDepth;
                        }
                        else if (ipr.s - minDepth < -0.2)
                        {
                            ipr.s = minDepth;
                        }

                        featureDepthLast_[key] = ipr.s;
                    } // use lidar depth map to successfully assign depth feature end
                    else
                    {
                        ipr.s = 0;
                        ipr.v = 0;
                    } // cannot find 3 lidar points to assign depth feature end

                    // try to use visual triangulation method recover feature depth
                    if (std::fabs(ipr.v) < 0.5)
                    {
                        depth2_count++;
                        Transformf T_first;
                        {
                            Eigen::Quaternionf q;
                            Eigen::Vector3f t;
                            q.x() = startTransLast_.at(key).x;
                            q.y() = startTransLast_.at(key).y;
                            q.z() = startTransLast_.at(key).z;
                            q.w() = startTransLast_.at(key).data[3];
                            t.x() = startTransLast_.at(key).h;
                            t.y() = startTransLast_.at(key).s;
                            t.z() = startTransLast_.at(key).v;
                            T_first.rot = q;
                            T_first.pos = t;
                        }
                        Transformf T_prev_first = Tw_odom_.inverse() * T_first;

                        if (T_prev_first.pos.norm() > 1.0)
                        {
                            float u0 = ipr.x;
                            float v0 = ipr.y;
                            float u1 = startPointLast_.at(key).u;
                            float v1 = startPointLast_.at(key).v;

                            Eigen::Vector3f p0_3d{u0, v0, 1.0};
                            Eigen::Vector3f p1_3d{u1, v1, 1.0};

                            // 开始三角化
                            // d1*P1 = d2*R*P2 + t
                            // P3 = RP2
                            // d1*P1.transpose()*P1 = d2*P1.tanspose()*P3 + P1.transpose()*t
                            // d1*P3.transpose()*P1 = d2*P3.transpose() + P3.transpose()*t
                            // 然后整理成关于d1 d2的二元一次方程
                            Eigen::Vector3f p1_3d_unrotated = T_prev_first.rot * p1_3d;
                            Eigen::Vector2f b;
                            b[0] = T_prev_first.pos.dot(p0_3d);
                            b[1] = T_prev_first.pos.dot(p1_3d_unrotated);
                            Eigen::Matrix2f A;
                            A(0, 0) = p0_3d.dot(p0_3d);
                            A(1, 0) = p0_3d.dot(p1_3d_unrotated);
                            A(0, 1) = -A(1, 0);
                            A(1, 1) = -p1_3d_unrotated.dot(p1_3d_unrotated);
                            Eigen::Vector2f lambda = A.inverse() * b;
                            Eigen::Vector3f xm = lambda[0] * p0_3d;
                            Eigen::Vector3f xn = T_prev_first.pos + lambda[1] * p1_3d_unrotated;
                            double depth = (xm[2] + xn[2]) / 2;

                            if (depth > 0.5 && depth < 100)
                            {
                                ipr.s = depth;
                                ipr.v = 2;
                            }
                        }

                        if (ipr.v == 2)
                        {
                            if (featureDepthLast_[key] > 0)
                            {

                                // ipr.s = 3 * ipr.s * featureDepthLast_[key] / (ipr.s + 2 * featureDepthLast_[key]);
                                ipr.s = featureDepthLast_[key] * 0.4 + ipr.s * 0.6;
                            }
                            featureDepthLast_[key] = ipr.s;
                            if (!std::isfinite(ipr.s))
                            {
                                featureDepthLast_[key] = 0;
                                ipr.s = 0;
                                ipr.v = 0;
                            }
                        }
                        else if (featureDepthLast_[key] > 0)
                        {
                            ipr.s = featureDepthLast_[key];
                            ipr.v = 2;
                        }
                    } // viusal triangulation end

                    if (!std::isfinite(ipr.s))
                    {
                        LOG(INFO) << "die";
                        featureDepthLast_[key] = 0;
                        ipr.s = 0;
                        ipr.v = 0;
                    }
                    if (ipr.v == 0)
                        depth3_count++;
                    ipRelations_.push_back(ipr); // save featue position in normalized image plane and depth
                    ipInd_.push_back(key);       // save feaure  index
                }
            } // end deal with all feature
            LOG(INFO) << "directly receive depth feature: " << depth1_count;
            LOG(INFO) << "canont directly receive depth feature: " << depth2_count;
            LOG(INFO) << "epipo_line constraint point: " << depth3_count;
            LOG(INFO) << "ipRelations: " << ipRelations_.size();

            int iterNum = 150;
            int ptNumNoDepthRec = 0;
            int ptNumWithDepthRec = 0;
            double meanValueWithDepthRec = 100000;
            LOG(INFO) << "before opt transform: " << T_cur_pre_;
            LOG(INFO) << "-------------------------------------";

            for (int iterCount = 0; iterCount < iterNum; iterCount++)
            {
                Matrix6x6 H;
                Vector6 b;
                double res = build_InDerectLinearSystem(
                    H, b,
                    T_cur_pre_,
                    ipRelations_,
                    iterCount,
                    ptNumNoDepthRec,
                    ptNumWithDepthRec,
                    meanValueWithDepthRec);
                Vector6 matX = H.ldlt().solve(b);

                if (iterCount == 0)
                    LOG(INFO) << "H:\n"
                              << H
                              << "\nb:\n"
                              << b;

                float x = matX(0, 0);
                float y = matX(1, 0);
                float z = matX(2, 0);
                float rho = matX(3, 0);
                float pitch = matX(4, 0);
                float yaw = matX(5, 0);
                if (!std::isfinite(x))
                    throw std::runtime_error("NAN");
                Eigen::Vector3f dt{x, y, z};
                Eigen::Quaternionf dtheta = Utility::deltaQ(Eigen::Vector3f{rho, pitch, yaw});
                Transformf deltaTransform = Transformf::so3Transexp(matX);
                T_cur_pre_.pos = T_cur_pre_.pos + dt;
                T_cur_pre_.rot = dtheta * T_cur_pre_.rot;
                T_cur_pre_.rot.normalize();

                float deltaR = Utility::R2ypr(dtheta.toRotationMatrix()).norm();
                float deltaT = 10.f * dt.norm();
                if ((deltaR < 0.00001 && deltaT < 0.00001))
                {
                    break;
                }
            }
            LOG(INFO) << "-------------------------------------";
            LOG(INFO) << "--------end optimization pose--------";
        } // end condition where kdTree_ isn't  null
        else
        {
            Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
            extrinsic.block<3, 4>(0, 0) = extrinsics_;
            Transformf sophus_twc(extrinsic.inverse());
            Tw_odom_ = sophus_twc;
        }

        Transformf T_pre_cur = T_cur_pre_.inverse();
        Tw_odom_ = Tw_odom_ * T_pre_cur;
        LOG(INFO) << "odometry: \n"
                  << Tw_odom_.matrix3x4();
        // TODO: KDTree reset used current pointcloud
        kdTree_->setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(pc)));

        pcl::PointXYZHSV spc;
        spc.x = Tw_odom_.rot.x();
        spc.y = Tw_odom_.rot.y();
        spc.z = Tw_odom_.rot.z();
        spc.data[3] = Tw_odom_.rot.w();
        spc.h = Tw_odom_.pos.x();
        spc.s = Tw_odom_.pos.y();
        spc.v = Tw_odom_.pos.z();

        startPointCur_.clear();
        startTransCur_.clear();
        featureDepthCur_.clear();
        startPointCur_.reserve(featureTrackResultCur_.size());
        startTransCur_.reserve(featureTrackResultCur_.size());
        featureDepthCur_.reserve(featureTrackResultCur_.size());

        LOG(INFO) << "update feature depth and fist point, trans";
        for (auto feature : featureTrackResultCur_)
        {
            int key = feature.first;
            if (featureTrackResultLast_.find(key) != featureTrackResultLast_.end())
            {
                startPointCur_[key] = startPointLast_[key];
                startTransCur_[key] = startTransLast_[key];

                if (featureDepthLast_[key] > 0)
                {
                    float ipz = featureDepthLast_[key];
                    float ipx = featureTrackResultLast_[key].u * ipz;
                    float ipy = featureTrackResultLast_[key].v * ipz;
                    Eigen::Vector3f P0{ipx, ipy, ipz};
                    Eigen::Vector3f P1 = T_cur_pre_ * P0;
                    featureDepthCur_[key] = P1[2];
                }
                else
                {
                    featureDepthCur_[key] = -1;
                }
            }
            else
            {
                startPointCur_[key] = feature.second;
                startTransCur_[key] = spc;
                featureDepthCur_[key] = -1;
            }
        }
        LOG(INFO) << "update feature depth and fist point, trans end";
    }

    float Frontend::build_InDerectLinearSystem(
        Eigen::Matrix<float, 6, 6> &H,
        Eigen::Matrix<float, 6, 1> &b,
        Transformf &T_cur_ref,
        pcl::PointCloud<pcl::PointXYZHSV> &ipRelations,
        int iterCount,
        int &ptNumNoDepthRec,
        int &ptNumWithDepthRec,
        double &meanValueWithDepthRec)
    {
        pcl::PointXYZHSV ipr;
        pcl::PointXYZHSV ipr2, ipr3, ipr4;

        int ipRelationsNum = ipRelations.size();
        ipRelations2_.clear();
        ipy2_.clear();
        ipRelations2_.reserve(ipRelationsNum);
        ipy2_.reserve(ipRelationsNum);

        if (iterCount == 0)
        {
            LOG(INFO) << "constraint size: " << ipRelations2_.size();
            LOG(INFO) << "T_cur_ref : \n"
                      << T_cur_ref;
        }
        int ptNumNoDepth = 0;
        int ptNumWithDepth = 0;
        double meanValueWithDepth = 0;

        for (int i = 0; i < ipRelationsNum; i++)
        {
            ipr = ipRelations[i];
            float u0 = ipr.x;
            float v0 = ipr.y;
            float u1 = ipr.z;
            float v1 = ipr.h;

            // 没有深度的点提供角度约束
            if ((fabs(ipr.v) < 0.5f || fabs(ipr.v - 2) < 0.5) && T_cur_ref.pos.norm() > 0.1f)
            {
                Eigen::Vector3f P0{u0, v0, 1.0f};
                Eigen::Vector3f P1{u1, v1, 1.0f};
                Eigen::Matrix<float, 1, 3> a;
                float t1 = T_cur_ref.pos[0];
                float t2 = T_cur_ref.pos[1];
                float t3 = T_cur_ref.pos[2];
                a[0] = -v1 * t3 + t2;
                a[1] = u1 * t3 - t1;
                a[2] = -u1 * t2 + v1 * t1;
                Eigen::Vector3f RP0 = T_cur_ref.rot * P0;

                // 提供的是点点极线的约束
                // 自动将1x3 赋值为3x1的向量
                Eigen::Vector3f Je_Jtheta = -a * Utility::skewSymmetric(RP0);
                Eigen::Vector3f Je_Jt;
                Je_Jt[0] = -RP0[1] + RP0[2] * v1;
                Je_Jt[1] = RP0[0] - RP0[2] * u1;
                Je_Jt[2] = -RP0[0] * v1 + RP0[1] * u1;

                float res = a * RP0;

                float huber_thresh_epipole = 0.5 / 760;
                float obs_std_dev_epipole = 0.75;

                float e = std::abs(res);

                Eigen::Vector3f epipo_line = Utility::skewSymmetric(T_cur_ref.pos) * RP0;
                float dist_to_epipoline = abs(P1.dot(epipo_line) / epipo_line.norm());

                float huber_weight =
                    dist_to_epipoline < huber_thresh_epipole ? 1.0 : huber_thresh_epipole / dist_to_epipoline;
                //            LOG(INFO) << "dist_to_epipoline: " << dist_to_epipoline << " ," << huber_weight;
                //            LOG(INFO) << "epipo_line inlier: " << (dist_to_epipoline < huber_thresh_epipole ? "true" : "false");
                //            LOG(INFO) << "huber_thresh_epipole: " << huber_thresh_epipole;
                float obs_weight =
                    huber_weight / (obs_std_dev_epipole);

                ipr2.x = Je_Jt[0] * obs_weight;
                ipr2.y = Je_Jt[0] * obs_weight;
                ipr2.z = Je_Jt[0] * obs_weight;
                ipr2.h = Je_Jtheta[0] * obs_weight;
                ipr2.s = Je_Jtheta[0] * obs_weight;
                ipr2.v = Je_Jtheta[0] * obs_weight;

                double y2 = res;

                //            if (ptNumWithDepthRec < 50 || (iterCount < 25 || fabs(y2) < 2 * meanValueWithDepthRec / 10000)) {
                if (ptNumWithDepthRec < 50 || iterCount < 25)
                {
                    float scale = 3.0;
                    ipr2.x *= scale;
                    ipr2.y *= scale;
                    ipr2.z *= scale;
                    ipr2.h *= scale;
                    ipr2.s *= scale;
                    ipr2.v *= scale;
                    y2 *= scale;

                    ipRelations2_.push_back(ipr2);
                    ipy2_.push_back(y2);

                    ptNumNoDepth++;
                }
                else
                {
                    ipRelations[i].v = -1;
                }
            } // end deal with no depth information feature point
            else if (std::fabs(ipr.v - 1) < 0.5 || fabs(ipr.v - 2) < 0.5)
            {
                // deal with 拥有深度的点
                float d0 = ipr.s;
                float t1 = T_cur_ref.pos[0];
                float t2 = T_cur_ref.pos[1];
                float t3 = T_cur_ref.pos[2];

                Eigen::Vector3f P0{u0 * d0, v0 * d0, d0};

                Eigen::Vector3f RP0 = T_cur_ref.rot * P0;
                Eigen::Vector3f P1 = RP0 + T_cur_ref.pos;

                float invz = 1.0 / P1[2];
                Eigen::Vector2f projP{P1[0] * invz, P1[1] * invz};

                Eigen::Vector2f res = Eigen::Vector2f{u1, v1} - projP;
                float e = res.norm();

                float huber_weight =
                    e < huber_thresh_ ? 1.0 : huber_thresh_ / e;
                //            LOG(INFO) << "reproj weight:" << huber_weight;
                float obs_weight =
                    huber_weight / (obs_std_dev_ * obs_std_dev_);

                float y3 = RP0[0] - u1 * RP0[2] + t1 - u1 * t3;
                float y4 = RP0[1] - v1 * RP0[2] + t2 - v1 * t3;

                float ey3 = abs(RP0[0] - u1 * RP0[2]);
                float ey4 = abs(RP0[1] - v1 * RP0[2]);

                float ey3y4 = sqrt(ey3 * ey3 + ey4 * ey4);
                //            LOG(INFO) << "ey3y4" << ey3y4;
                if (ey3y4 < 0.01)
                    obs_weight *= 0.1;

                Eigen::Matrix3f dRP_dtheta = -Utility::skewSymmetric(RP0);

                Eigen::Vector3f dy3_dtheta = dRP_dtheta.row(0) - u1 * dRP_dtheta.row(2);
                Eigen::Vector3f dy4_dtheta = dRP_dtheta.row(1) - v1 * dRP_dtheta.row(2);
                Eigen::Vector3f dy3_dt{1, 0, -u1};
                Eigen::Vector3f dy4_dt{0, 1, -v1};

                ipr3.x = dy3_dt[0] * obs_weight;
                ipr3.y = dy3_dt[1] * obs_weight;
                ipr3.z = dy3_dt[2] * obs_weight;
                ipr3.h = dy3_dtheta[0] * obs_weight;
                ipr3.s = dy3_dtheta[1] * obs_weight;
                ipr3.v = dy3_dtheta[2] * obs_weight;

                ipr4.x = dy4_dt[0] * obs_weight;
                ipr4.y = dy4_dt[1] * obs_weight;
                ipr4.z = dy4_dt[2] * obs_weight;
                ipr4.h = dy4_dtheta[0] * obs_weight;
                ipr4.s = dy4_dtheta[1] * obs_weight;
                ipr4.v = dy4_dtheta[2] * obs_weight;

                y3 *= obs_weight;
                y4 *= obs_weight;

                e = sqrt(y3 * y3 + y4 * y4);

                // 这个是用于排除外点, 迭代次数较少是
                if (ptNumWithDepthRec < 300 || iterCount < 70 ||
                    e < 2 * meanValueWithDepthRec)
                {
                    ipRelations2_.push_back(ipr3);
                    ipy2_.push_back(y3);

                    ipRelations2_.push_back(ipr4);
                    ipy2_.push_back(y4);

                    ptNumWithDepth++;
                    meanValueWithDepth += e;
                }
                else
                {
                    ipRelations.points[i].v = -1;
                }
            }
        }

        // 每次迭代都会剔除其中的外点
        meanValueWithDepth /= (ptNumWithDepth + 0.01);
        ptNumNoDepthRec = ptNumNoDepth;
        ptNumWithDepthRec = ptNumWithDepth;
        meanValueWithDepthRec = meanValueWithDepth;

        int ipRelations2Num = ipRelations2_.size();
        if (iterCount == 0)
        {
            LOG(INFO) << "constraint size: " << ipRelations2_.size();
            LOG(INFO) << "ptNumNoDepthRec: " << ptNumNoDepthRec;
            LOG(INFO) << "ptNumWithDepthRec: " << ptNumWithDepthRec;
        }
        if (ipRelations2Num > 10)
        {
            Eigen::Matrix<float, Eigen::Dynamic, 6> matA(ipRelations2Num, 6);
            Eigen::Matrix<float, 6, Eigen::Dynamic> matAt(6, ipRelations2Num);
            Eigen::VectorXf matB(ipRelations2Num);
            Eigen::Matrix<float, 6, 1> matX;

            for (int i = 0; i < ipRelations2Num; i++)
            {
                ipr2 = ipRelations2_[i];

                matA(i, 0) = ipr2.x;
                matA(i, 1) = ipr2.y;
                matA(i, 2) = ipr2.z;
                matA(i, 3) = ipr2.h;
                matA(i, 4) = ipr2.s;
                matA(i, 5) = ipr2.v;
                matB(i, 0) = -ipy2_[i];
            }

            matAt = matA.transpose();
            H = matAt * matA;
            b = matAt * matB;
        }

        return meanValueWithDepth;
    }

} // namespace vloam