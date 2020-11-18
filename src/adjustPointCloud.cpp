//
// Created by ubuntu on 2020/3/16.
//


#include <cmath>
#include <vector>
#include <queue>
#include <string>
#include <memory>

#include <mutex>
#include <condition_variable>
#include <thread>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/utility.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>


std::queue<std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>>> pointCloudBuf;

std::queue<double> pointCloudTimeStampBuf;

std::queue<std::shared_ptr<Eigen::Vector3d>> accBuf;

std::queue<std::shared_ptr<Eigen::Vector3d>> gyrBuf;

std::queue<std::shared_ptr<Eigen::Quaterniond>> orientationBuf;

std::queue<double> imuTimeStampBuf;

double td = 0;

struct pointClouData
{
    double timestamp;
    pcl::PointCloud<pcl::PointXYZI> pc;
};
struct ImuData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp;
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    Eigen::Quaterniond orientation;
};

using imuAndLidarData =std::vector<std::pair<std::vector<std::shared_ptr<ImuData>>, std::shared_ptr<pointClouData>>>;
std::mutex mBuf;
std::condition_variable con;
double curTime, prevTime;

// publish
ros::Publisher pubLaserCloud;
ros::Publisher pubLaserCloud2;
ros::Publisher pubSyncImu;
ros::Publisher pubImuOrientation;

std::vector<std::pair<std::vector<std::shared_ptr<ImuData>>, std::shared_ptr<pointClouData>>> measurements;

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    mBuf.lock();

    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::cout << " points: " <<laserCloudIn.points.size() << std::endl;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    std::cout << "remove NAN points: " <<laserCloudIn.points.size() << std::endl;
    pointCloudBuf.push(std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(laserCloudIn));
    pointCloudTimeStampBuf.push(laserCloudMsg->header.stamp.toSec());

    mBuf.unlock();
    con.notify_one();
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d acc(dx, dy, dz);
    Eigen::Vector3d gyr(rx, ry, rz);
    Eigen::Quaterniond q;
    q.x() = imu_msg->orientation.x;
    q.y() = imu_msg->orientation.y;
    q.z() = imu_msg->orientation.z;
    q.w() = imu_msg->orientation.w;

    Eigen::Vector3d ypr = Utility::R2ypr(q.toRotationMatrix());
    sensor_msgs::Imu msg = *imu_msg;
    msg.header.frame_id ="/camera_init";
    msg.linear_acceleration.z = ypr.x();
    msg.linear_acceleration.y = ypr.y();
    msg.linear_acceleration.x = ypr.z();

    pubImuOrientation.publish(msg);
    //    std::cout << "orientation1: "
    //              << q.x() << ","
    //              << q.y() << ","
    //              << q.z() << ","
    //              << q.w() << ","
    //              << std::endl;

    mBuf.lock();

    accBuf.push(std::make_shared<Eigen::Vector3d>(acc));
    gyrBuf.push(std::make_shared<Eigen::Vector3d>(gyr));
    orientationBuf.push(std::make_shared<Eigen::Quaterniond>(q));

    //    std::cout << "orientation2: "
    //              << orientationBuf.back()->x() << ","
    //              << orientationBuf.back()->y() << ","
    //              << orientationBuf.back()->z() << ","
    //              << orientationBuf.back()->w() << ","
    //              << std::endl;
    imuTimeStampBuf.push(imu_msg->header.stamp.toSec());

    mBuf.unlock();

    con.notify_one();
}

imuAndLidarData
getMeasurements()
{
    imuAndLidarData measurements;

    while (true) {
        // condition1: 没有可用的ＩＭＵ数据或者视觉测量数据
        if (pointCloudTimeStampBuf.empty() or imuTimeStampBuf.empty())
            return measurements;
        // condition2: IMU的数据时间戳小于视觉的时间戳，需要等待更多的imu数据
        if (!(imuTimeStampBuf.back() > pointCloudTimeStampBuf.front()+td))
            return measurements;
        // condition3: IMU数据的时间戳超前与视觉的时间戳，视觉测量需要剔除
        if (!(imuTimeStampBuf.front() < pointCloudTimeStampBuf.front()+td)) {
            ROS_WARN("throw img, only should happen at the beginning");

            pointCloudBuf.pop();
            pointCloudTimeStampBuf.pop();
            continue;
        }
        //　取出可用的激光数据
        auto pointCloud = pointCloudBuf.front();
        auto pointCloudTime = pointCloudTimeStampBuf.front();

        auto pointCloudMsg = std::make_shared<pointClouData>();
        pointCloudMsg->timestamp = pointCloudTime;
        pointCloudMsg->pc = *pointCloud;

        // 弹出激光数据
        pointCloudBuf.pop();
        pointCloudTimeStampBuf.pop();
        // 获取对应区间的IMU数据
        std::vector<std::shared_ptr<ImuData>> IMUs;
        while (imuTimeStampBuf.front() < pointCloudTime+td) {
            auto imudata = std::make_shared<ImuData>();
            imudata->timestamp = imuTimeStampBuf.front();
            imudata->accel = *accBuf.front();
            imudata->gyro = *gyrBuf.front();
            imudata->orientation = *orientationBuf.front();
            IMUs.emplace_back(imudata);

            // 弹出imu数据
            imuTimeStampBuf.pop();
            accBuf.pop();
            gyrBuf.pop();
            orientationBuf.pop();
        }
        // 多一帧imu数据用于插值
        auto imudata = std::make_shared<ImuData>();
        imudata->timestamp = imuTimeStampBuf.front();
        imudata->accel = *accBuf.front();
        imudata->gyro = *gyrBuf.front();
        imudata->orientation = *orientationBuf.front();
        IMUs.emplace_back(imudata);
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, pointCloudMsg);
    }
    return measurements;
}
void process()
{
    while (ros::ok()) {

        imuAndLidarData measurements;
        //notice 要对视觉测量和IMU缓冲队列进行写操作，注意上锁
        std::unique_lock<std::mutex> lk(mBuf);
        con.wait(lk, [&]
        {
            return (measurements = getMeasurements()).size() != 0;
        });
        lk.unlock(); // 解锁，可以让新的视觉测量和IMU输入压入缓冲队列

        for(auto &measurement : measurements)
        {
            // step1: 播放数据
            // step1.1: 播放激光数据
            auto pPointCloud = measurement.second;
            auto vImu = measurement.first;
            // step1.2 去除激光点云的roll和pitch 的旋转
            Eigen::Matrix3d Rwc;
            Rwc = vImu.back()->orientation.toRotationMatrix();
            sensor_msgs::PointCloud2 laserCloudOutMsg, laserCloudOutMsg2;
            pcl::toROSMsg(pPointCloud->pc, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = ros::Time().fromSec(pPointCloud->timestamp);
            laserCloudOutMsg.header.frame_id = "/vloam/base_link";

            for(auto& point:  pPointCloud->pc.points)
            {
                Eigen::Vector3d pt;
                pt << point.x, point.y, point.z;
                pt = Rwc*pt;
                point.x = pt.x();
                point.y = pt.y();
                point.z = pt.z();
            }

            pcl::toROSMsg(pPointCloud->pc, laserCloudOutMsg2);
            laserCloudOutMsg2.header.stamp = ros::Time().fromSec(pPointCloud->timestamp);
            laserCloudOutMsg2.header.frame_id = "/vloam/base_link";

            pubLaserCloud.publish(laserCloudOutMsg);
            pubLaserCloud2.publish(laserCloudOutMsg2);

            std::cout << "imu size: " << vImu.size() << std::endl;
            std::cout << "lidar timestamp: " << pPointCloud->timestamp << std::endl;
            std::cout << "imu timestamp: " << vImu.back()->timestamp << std::endl;
            std::cout << "imu and lidar timestamp diff: " << std::abs(pPointCloud->timestamp - vImu.back()->timestamp) << std::endl;


            // step1.2: 播放IMU数据
            sensor_msgs::Imu imuMsg;
            imuMsg.header.stamp = ros::Time().fromSec(vImu.back()->timestamp);
            imuMsg.header.frame_id = "/vloam/base_link";
            imuMsg.angular_velocity.x = vImu.back()->gyro.x();
            imuMsg.angular_velocity.y = vImu.back()->gyro.y();
            imuMsg.angular_velocity.z = vImu.back()->gyro.z();

            imuMsg.linear_acceleration.x = vImu.back()->accel.x();
            imuMsg.linear_acceleration.y = vImu.back()->accel.y();
            imuMsg.linear_acceleration.z = vImu.back()->accel.z();

            imuMsg.orientation.x = vImu.back()->orientation.x();
            imuMsg.orientation.y = vImu.back()->orientation.y();
            imuMsg.orientation.z = vImu.back()->orientation.z();
            imuMsg.orientation.w = vImu.back()->orientation.w();

            pubSyncImu.publish(imuMsg);

        }
    }
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "adjustPointCloud");
    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/points_raw", 100, laserCloudHandler);
    ros::Subscriber sub_imu = nh.subscribe("/imu_correct", 2000, imu_callback, ros::TransportHints().tcpNoDelay());

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/vloam/point_cloud", 100);
    pubLaserCloud2 = nh.advertise<sensor_msgs::PointCloud2>("/vloam/point_cloud1", 100);
    pubSyncImu = nh.advertise<sensor_msgs::Imu>("/vloam/imu", 100);
    pubImuOrientation = nh.advertise<sensor_msgs::Imu>("/imu/orient", 100);
    std::thread measurement_process{process};

    ros::spin();

    //measurement_process.join();
    return 0;
}
