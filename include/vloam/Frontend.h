//
// Created by ubuntu on 2020/3/9.
//
#pragma once

#include "Config.h"
#include "PinholeModel.h"
#include "Keyframe.h"
#include "sophus/se3.hpp"
#include "Tracker2.h"
#include "vloam/featureTracking.h"
#include "vloam/KeyframeWindow.h"
#include "vloam/WindowOptimizer.h"

namespace vloam
{
class CameraModel;

class UniqueId
{
public:
    UniqueId()
        : unique_id(0)
    {
    }
    int id()
    {
        return unique_id++;
    }

private:
    int unique_id;
};

static UniqueId id_manager;
// 设计理念是: 包含配置文件和追踪模块
// 数据处理模块是一个单独的模块

class Frontend
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Frontend> Ptr;
    Frontend();
    Frontend(const std::string &path, const std::string &fname);
    ~Frontend();

    bool track_camlidar(Frame::Ptr pcurrent);

    Transformf T_last_cur()
    { return T_last_cur_; }
    Transformf Tw_odom()
    { return Tw_odom_; }
    void
    trackfeature(double timestamap, const cv::Mat image, pcl::PointCloud<pcl::PointXYZI> &pc);

    CameraModel::Ptr camera()
    {
        return pcamera_;
    }

    float build_InDerectLinearSystem(
        Eigen::Matrix<float, 6, 6> &H,
        Eigen::Matrix<float, 6, 1> &b,
        Transformf &T_cur_ref,
        pcl::PointCloud<pcl::PointXYZHSV> &ipRelations,
        int iterCount,
        int &ptNumNoDepthRec,
        int &ptNumWithDepthRec,
        double &meanValueWithDepthRec
    );
private:
    Matrix3x4 extrinsics_;
    CameraModel::Ptr pcamera_;

    Keyframe::Ptr platestKf_;
    Frame::Ptr plastFrame_;

    // featuretracking module
    featureTracking::Ptr pfeaturetracker_;
    // Tracker2 module
    Tracker2::Ptr ptracker2_;

    int num_keyframe_;
    // KF Window
    KeyframeWindow::Ptr kf_window_;
    // Slide Window
    WindowOptimizer::Ptr window_optimizer_;

    bool btracking_status_;
    bool binitialize_;

    Transformf T_last_k_;
    Transformf dT_cur_last_;
    Transformf T_last_cur_;
    Transformf T_k_last_;
    Transformf Tw_odom_;
    Transformf T_cur_pre_;
private:

    pcl::PointCloud<ImagePoint> featurePointLast_;
    pcl::PointCloud<ImagePoint> featurePointCur_;

    featureTrackResult featureTrackResultLast_;
    featureTrackResult featureTrackResultCur_;

    pcl::PointCloud<pcl::PointXYZI> depthCloundLast_;
    pcl::PointCloud<pcl::PointXYZI> depthCloundCur_;

    featureTrackResult startPointLast_;
    featureTrackResult startPointCur_;

    Eigen::unordered_map<int, pcl::PointXYZHSV> startTransLast_;
    Eigen::unordered_map<int, pcl::PointXYZHSV> startTransCur_;

    pcl::PointCloud<pcl::PointXYZHSV> ipRelations_;
    pcl::PointCloud<pcl::PointXYZHSV> ipRelations2_;

    std::vector<int> ipInd_;
    std::vector<float> ipy2_;

    std::unordered_map<int, float> featureDepthLast_;
    std::unordered_map<int, float> featureDepthCur_;

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdTree_ = nullptr;

    int featurePointLastNum_ = 0;
    int featurePointCurNum_ = 0;

    double huber_thresh_ = 1.5 / 760;
    double obs_std_dev_ = 0.5;
    double huber_thresh_epipole_ = 1.2 / 760;
    double obs_std_dev_epipole_ = 0.75;
}; //class Frontend

} //namespace vloam
