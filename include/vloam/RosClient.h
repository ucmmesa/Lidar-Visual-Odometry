//
// Created by ubuntu on 2020/3/9.
//
#pragma once

#include <memory>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>


#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include "vloam/Resources.h"
#include "vloam/Datatypes.h"

class RosClient
{

public:
    RosClient(std::shared_ptr<ros::NodeHandle> node,
              std::shared_ptr<ros::NodeHandle> privante_nh,
              camlidar_topic_t camlidar_topic,
              std::shared_ptr<cam_lidar_queue_t> camlidar_queue);

    ~RosClient();

    // publish odometry
    void publishOdometry(int64_t timestamp, const vloam::Matrix3x4& pose);
    // spin
    void run();
private:
    // callback function
    void image_callback(const sensor_msgs::ImageConstPtr &cam);
    void cloud_callback(const sensor_msgs::PointCloud2ConstPtr &lidar);

private:

    // topic name for image and lidar
    camlidar_topic_t camlidar_topic_;
    // message queue
    std::shared_ptr<cam_lidar_queue_t> camlidar_queue_;
    cv_bridge::CvImagePtr cv_ptr_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc_ptr_;

    std::shared_ptr<ros::NodeHandle> nh_;
    std::shared_ptr<ros::NodeHandle> privante_nh_;

    // image subscribe
    image_transport::ImageTransport *it_;
    image_transport::Subscriber sub_;
    // pointcloud sub
    ros::Subscriber cloud_sub_;

    // imu sub
    ros::Subscriber imu_sub_;
    // odometry publisher
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubLaserPath;
    nav_msgs::Path laserPath;

};


