//
// Created by ubuntu on 2020/3/9.
//

#include "vloam/RosClient.h"



RosClient::RosClient(std::shared_ptr<ros::NodeHandle> node,
                     std::shared_ptr<ros::NodeHandle> privante_nh,
                     camlidar_topic_t camlidar_topic,
                     std::shared_ptr<cam_lidar_queue_t> camlidar_queue)
    : camlidar_queue_(camlidar_queue)
{
    camlidar_topic_ = camlidar_topic;
    std::cerr << "[RosClient]\t Camera topic name : " << camlidar_topic_.cam_topic << std::endl;
    std::cerr << "[RosClient]\t Velodyne topic name : " << camlidar_topic_.lidar_topic << std::endl;
    // ROS setting
    this->nh_ = node;

    pc_ptr_.reset(new pcl::PointCloud<pcl::PointXYZI>);

    // For TUM
    it_ = new image_transport::ImageTransport(*nh_);
    sub_ = it_->subscribe(camlidar_topic_.cam_topic, 1, &RosClient::image_callback, this);
    cloud_sub_ = nh_->subscribe(camlidar_topic_.lidar_topic, 1, &RosClient::cloud_callback, this);

    pubLaserOdometry = nh_->advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    pubLaserPath = nh_->advertise<nav_msgs::Path>("/laser_odom_path", 100);
}


void RosClient::publishOdometry(int64_t timestamp, const vloam::Matrix3x4& pose)
{
    Eigen::Quaterniond q(pose.cast<double>().block<3,3>(0,0));
    Eigen::Vector3d t = pose.cast<double>().block<3,1>(0,3);

    // publish odometry
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "/camera_init";
    laserOdometry.child_frame_id = "/laser_odom";
    laserOdometry.header.stamp = ros::Time().fromNSec(timestamp);
    laserOdometry.pose.pose.orientation.x = q.x();
    laserOdometry.pose.pose.orientation.y = q.y();
    laserOdometry.pose.pose.orientation.z = q.z();
    laserOdometry.pose.pose.orientation.w = q.w();
    laserOdometry.pose.pose.position.x = t.x();
    laserOdometry.pose.pose.position.y = t.y();
    laserOdometry.pose.pose.position.z = t.z();
    pubLaserOdometry.publish(laserOdometry);

    geometry_msgs::PoseStamped laserPose;
    laserPose.header = laserOdometry.header;
    laserPose.pose = laserOdometry.pose.pose;
    laserPath.header.stamp = laserOdometry.header.stamp;
    laserPath.poses.push_back(laserPose);
    laserPath.header.frame_id = "/camera_init";
    pubLaserPath.publish(laserPath);
}
void RosClient::run()
{
    ros::spin();
}

void RosClient::image_callback(const sensor_msgs::ImageConstPtr& cam) {
    //    cv_ptr_ = cv_bridge::toCvCopy(cam, sensor_msgs::image_encodings::MONO8);
    //    cerr << "image_callback : " << cam->header.stamp.toNSec() << endl;

    //    cv_ptr_ = cv_bridge::toCvCopy(cam, cam->encoding);

    //    std::lock_guard<std::mutex> lg(camlidar_queue_->mtx_img);
    std::unique_lock<std::mutex> ul(camlidar_queue_->mtx_img);

    //    sensor_msgs::Image img = *cam;

    camlidar_queue_->img_queue.push(cam);
    //    camlidar_queue_->img_queue.push(cv_ptr_);

    //    camlidar_queue_->cond_camlidar.notify_one();


    ul.unlock();
    camlidar_queue_->cond_img.notify_one();
}

void RosClient::cloud_callback (const sensor_msgs::PointCloud2ConstPtr& lidar)
{
    //    cerr << "cloud_callback : " << lidar->header.stamp.toNSec() << endl;

    //    pcl::fromROSMsg(*lidar, *pc_ptr_);

    //    std::lock_guard<std::mutex> lg(camlidar_queue_->mtx_pc);
    std::unique_lock<std::mutex> ul(camlidar_queue_->mtx_pc);

    //    camlidar_queue_->pc_queue.push(lidar);
    //    sensor_msgs::PointCloud2 pc = *lidar;
    camlidar_queue_->pc_queue.push(lidar);


    //    pc = camlidar_queue_->pc_queue.front();
    //    camlidar_queue_->pc_queue.push(pc_ptr_);

    ul.unlock();
    camlidar_queue_->cond_pc.notify_one();
}

