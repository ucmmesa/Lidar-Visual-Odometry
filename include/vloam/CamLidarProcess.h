//
// Created by ubuntu on 2020/3/9.
//

#pragma once

#include <ros/ros.h>
#include <ros/package.h>

#include <thread>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/core/core.hpp>

#include <vloam/Datatypes.h>
#include <vloam/Point.h>
#include <vloam/CameraModel.h>

#include <vloam/Frame.h>
#include <vloam/Frontend.h>
#include <vloam/BackEndSolver.h>

#include <vloam/Resources.h>
#include <vloam/RosClient.h>

class CamLidarProcess
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef pair<sensor_msgs::ImageConstPtr, sensor_msgs::PointCloud2ConstPtr> CamlidarPair;

    CamLidarProcess(std::shared_ptr<ros::NodeHandle> node,
                    std::shared_ptr<ros::NodeHandle> privante_nh);

    // 线程需要join的join
    ~CamLidarProcess();

    // 准备数据
    void prepare_cam_lidar();
    // 数据准备好就需要开启当前帧和关键帧进行匹配
    void track_camlidar();
    // 开启数据线程
    void run();

public:
vloam::Matrix3x4 newPose;
Transformf T_last_cur;

private:

    shared_ptr<cam_lidar_queue_t> camlidar_queue_;

    RosClient *ros_client_;
    std::thread *ros_client_thread_;
    std::thread *camlidar_process_thread_;
    std::thread *system_thread_;

    // Data
    //    uint32_t timestamp_;
    sensor_msgs::ImageConstPtr img_ptr_;
    sensor_msgs::PointCloud2ConstPtr pc_ptr_;

    queue<CamlidarPair> camlidar_pair_queue_;
    std::mutex mtx_camlidar_;
    std::condition_variable cond_camlidar_;

    cv_bridge::CvImagePtr img_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_;

    // System
    vloam::Frontend::Ptr vloam_frontend_;
    vloam::CameraModel::Ptr camera_;
    vloam::Frame::Ptr reference_, current_;
    
    vloam::Matrix3x4 extrinsics_;

    bool status_;
    int cnt_ = 0;


};

