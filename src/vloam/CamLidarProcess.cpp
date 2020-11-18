//
// Created by ubuntu on 2020/3/9.
//

#include "vloam/CamLidarProcess.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

// VLP-64
extern const int N_SCAN = 64;

extern const int Horizon_SCAN = 1800;

extern const float ang_res_x = 0.2;

extern const float ang_res_y = 0.427;

extern const float ang_bottom = 24.9;

extern const int groundScanInd = 16;

extern const bool loopClosureEnableFlag = false;

extern const double mappingProcessInterval = 0.3;

CamLidarProcess::CamLidarProcess(
    std::shared_ptr<ros::NodeHandle> node,
    std::shared_ptr<ros::NodeHandle> privante_nh)
{
    std::string pkg_path = ros::package::getPath("aloam_velodyne");
    std::string params_path = pkg_path + "/params/";
    std::cerr << "[CamLidarProcess]\t Parameter Path is " << params_path << std::endl;

    // read config file
    cv::FileStorage f_ros_settings(params_path + "camlidar_system2.yaml", cv::FileStorage::READ);

    // get camera and lidar topic
    camlidar_topic_t camlidar_topic;
    camlidar_topic.cam_topic = string(f_ros_settings["Ros.Camera"]);
    camlidar_topic.lidar_topic = string(f_ros_settings["Ros.Velodyne"]);

    std::cerr << "[CamLidarProcess]\t Camera topic name : " << camlidar_topic.cam_topic << std::endl;
    std::cerr << "[CamLidarProcess]\t Velodyne topic name : " << camlidar_topic.lidar_topic << std::endl;

    string param_fname = string(f_ros_settings["Parameter.File"]);
    std::cerr << "[CamLidarProcess]\t parameter files : " << params_path + param_fname << std::endl;

    // Frontend Setting
    vloam_frontend_.reset(new vloam::Frontend(params_path, param_fname));
    extrinsics_ = vloam::Config::cfg()->camlidar().extrinsic;
    camera_ = vloam::Config::cfg()->camera();

    static_pointer_cast<vloam::PinholeModel>(camera_)->show_intrinsic();
    camera_ = vloam_frontend_->camera();

    camlidar_queue_ = shared_ptr<cam_lidar_queue_t>(new cam_lidar_queue_t);
    ros_client_ = new RosClient(node, privante_nh, camlidar_topic, camlidar_queue_);
    status_ = true;

    pc_.reset(new pcl::PointCloud<pcl::PointXYZI>);
}

CamLidarProcess::~CamLidarProcess()
{
    vloam_frontend_.reset();
    // thread joint
    ros_client_thread_->join();
    camlidar_process_thread_->join();
    system_thread_->join();
}

///  camera 和 lidar的同步
void CamLidarProcess::prepare_cam_lidar()
{
    while (status_) {
        //        cerr << "[CamLidarProcess]\t preparing data" << endl;

        //        std::unique_lock<std::mutex> ul(camlidar_queue_->mtx_camlidar);

        // step1:获取激光点云数据
        std::unique_lock<std::mutex> ul_pc(camlidar_queue_->mtx_pc);

        while (camlidar_queue_->pc_queue.empty()) {
            camlidar_queue_->cond_pc.wait(ul_pc);
        }
        std::cout << "prepare_cam_lidar: lidar" << std::endl;
        pc_ptr_ = camlidar_queue_->pc_queue.front();

        // step2: 获取图像数据
        std::unique_lock<std::mutex> ul_img(camlidar_queue_->mtx_img);

        while (camlidar_queue_->img_queue.empty()) {
            camlidar_queue_->cond_img.wait(ul_img);
        }

        if (camlidar_queue_->img_queue.size() < 2) {
            //            cerr << "[CamLidarProcess]\t img_queue size < 2 " << endl;
            //            ul_pc.unlock();
            //            ul_img.unlock();
            //            continue;
        }

        // step3 获取最新点云数据的时间戳
        int64_t pc_timestamp = pc_ptr_->header.stamp.toNSec();

        // step4 获取最新图像数据的时间戳
        std::cout << "prepare_cam_lidar: get image timestamp" << std::endl;
        auto last_img = camlidar_queue_->img_queue.end() - 1;
        int64_t last_img_timestamp = (*last_img)->header.stamp.toNSec();

        if (last_img_timestamp < pc_timestamp) {
            //            cerr << "[CamlidarProcess]\t img_timestamp < pc_timestamp" << endl;
            //            cerr << last_img_timestamp << " , " << pc_timestamp << endl;
            //            cerr << cnt_++ << endl;

            //            ul_pc.unlock();
            //            ul_img.unlock();
            //            continue;
        }

        int match_idx = 0;
        int64_t min_diff = 1000000000; // 1s
        //        int64_t min_diff_thresh = 156650000; //12500000;
        int64_t min_diff_thresh = 226650000; // 0.22665
        bool exist_match = false;

        // 查找图像队列中与最新点云数据时间间隔最小的图像
        std::cout << "prepare_cam_lidar: timestamp match" << std::endl;

        for (auto iter = camlidar_queue_->img_queue.begin(); iter != camlidar_queue_->img_queue.end();
             ++iter, ++match_idx) {

            int64_t diff = pc_ptr_->header.stamp.toNSec() - (*iter)->header.stamp.toNSec();

            if (abs(diff) < min_diff) {
                min_diff = abs(diff);
                img_ptr_ = *iter;
            }
        }

        // 如果时间间隔小于0.2s,表示找到了与激光点云数据匹配的一帧图像
        if (min_diff < min_diff_thresh) {
            exist_match = true;
            // 将匹配的图像数据之前的image buffer 清空
            for (int i = 0; i < match_idx; ++i)
                camlidar_queue_->img_queue.pop();

            camlidar_queue_->pc_queue.pop();
        }

        ul_pc.unlock();
        ul_img.unlock();

        // 如果找到匹配的点云数据和,那么将图像和点云数据组合成一个pair
        if (exist_match) {
            //            cerr << pc_ptr_->header.stamp.toNSec() << " - " << img_ptr_->header.stamp.toNSec() << " = " << static_cast<double> (min_diff) * 1e-9 << endl;

            std::cout << "prepare_cam_lidar: compose image lidar pair" << std::endl;
            std::unique_lock<std::mutex> ul_camlidar(mtx_camlidar_);
            CamlidarPair camlidar_pair = make_pair(img_ptr_, pc_ptr_);
            camlidar_pair_queue_.push(camlidar_pair);
            std::cout << "prepare_cam_lidar: compose image lidar pair successfully" << std::endl;
            ul_camlidar.unlock();
            cond_camlidar_.notify_one();
        }
    }

    cerr << "[CamLidarProcess]\t Out of preparing data" << endl;
}

void CamLidarProcess::track_camlidar()
{
    while (status_) {
        cerr << "    ---------------------------------------" << std::endl;
        cerr << "      [CamLidarProcess]\t track_camlidar()" << endl;
        cerr << "    ---------------------------------------" << std::endl;

        bool data_ready = false;
        CamlidarPair camlidar_pair;
        sensor_msgs::ImageConstPtr img_ptr;
        sensor_msgs::PointCloud2ConstPtr pc_ptr;

        if (!data_ready) {
            std::unique_lock<std::mutex> ul(mtx_camlidar_);

            while (camlidar_pair_queue_.empty()) {
                //                ul.unlock();
                //                continue;
                cond_camlidar_.wait(ul);
            }
            std::cout << "cam-lidar ready1" << std::endl;
            camlidar_pair = camlidar_pair_queue_.front();
            camlidar_pair_queue_.pop();

            ul.unlock();
            data_ready = true;
        }

        if (data_ready) {
            std::cout << "cam-lidar ready2" << std::endl;
            img_ptr = camlidar_pair.first;
            pc_ptr = camlidar_pair.second;

            //            ofstream WriteFile;
            //            WriteFile.open("frame_stamp.csv",ios_base::app);
            //            WriteFile << img_ptr->header.stamp << endl;
            //            WriteFile.close();

            std::cout << "step1: get image" << std::endl;
            img_ = cv_bridge::toCvCopy(img_ptr, img_ptr->encoding);
            std::cout << "step2: get pointcloud" << std::endl;
            // 将ros的激光数据消息转换为pcl::PointCloud<pcl::PointXYZI>::Ptr 格式
            pcl::fromROSMsg(*pc_ptr, *pc_);
            pcl::PointCloud<pcl::PointXYZI> downsampled_cloud;

            pcl::PointXYZI thisPoint;

            auto cloudSize = pc_->points.size();
            float verticalAngle;
            size_t rowIdn;
            for (size_t i = 0; i < cloudSize; ++i) {
                thisPoint.x = pc_->points[i].x;
                thisPoint.y = pc_->points[i].y;
                thisPoint.z = pc_->points[i].z;
                thisPoint.intensity = pc_->points[i].intensity;

                verticalAngle =
                    atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
                if (rowIdn < 0 || rowIdn >= N_SCAN)
                    continue;

                downsampled_cloud.push_back(thisPoint);
                if (rowIdn % 4 == 0) {
                    downsampled_cloud.push_back(thisPoint);
                }
            }

            // TODO:
            // 将降采样的点云赋值给pc_
            // *pc_ = downsampled_cloud;


            std::cout << "step3: copy pointcloud" << std::endl;
            // 将PointXYZI类型的激光数据转化为PointXYZRGBA格式的点云数据
            // 这一块可能会报警告
            vloam::PointCloud pointcloud;
            pcl::copyPointCloud(*pc_, pointcloud);

            // 将激光数据转化到相机坐标系下
            Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
            extrinsic.block<3, 4>(0, 0) = extrinsics_;
            pcl::transformPointCloud(pointcloud, pointcloud, extrinsic);

            std::cout << "rgb lidar: " << __FILE__ << __LINE__ << std::endl;

            reference_ = current_;

            std::cout << "sweep two frame: " << __FILE__ << __LINE__ << std::endl;
            cv::Mat rgb, rectified;
            // cerr << img_->image.type() << endl;
            rgb = img_->image; // for color kitti
            // camera_->undistort_image(rgb, rectified);
            // img_->image.copyTo(rectified); // for kitti

            cvtColor(img_->image, rgb, CV_GRAY2BGR);

            rectified = rgb;
            cerr << "[CamLidarProcess]\t Initialize Frame" << endl;

            current_ = std::make_shared<vloam::Frame>(pc_ptr_->header.stamp.toNSec(), rectified, pointcloud, camera_);
            cerr << "[CamLidarProcess]\t constructe Frame successfully" << endl;

            //TODO: 增加特征追踪的结果
            LOG(INFO) << "[CamLidarProcess]\t track feature point";
            pcl::PointCloud<pcl::PointXYZI> &pc_in_cameraview = current_->xyzipc();

#if 0

            vloam_frontend_->track_camlidar(current_);

            ros_client_->publishOdometry(pc_ptr_->header.stamp.toSec(), current_->Twc().matrix3x4());
#else
            vloam_frontend_->trackfeature(pc_ptr_->header.stamp.toSec(), img_->image, pc_in_cameraview);

            vloam::Matrix3x4 T_cam_liar = vloam::Config::cfg()->camlidar().extrinsic;

            vloam::Matrix3x4 T_w_liar;
            T_w_liar.block<3, 3>(0, 0) = vloam_frontend_->Tw_odom().rotationMatrix() * T_cam_liar.block<3, 3>(0, 0);
            T_w_liar.block<3, 1>(0, 3) = vloam_frontend_->Tw_odom().rotationMatrix() * T_cam_liar.block<3, 1>(0, 3)
                + vloam_frontend_->Tw_odom().pos;

            ros_client_->publishOdometry(current_->timestamp(), T_w_liar);
#endif
            //            cout << setprecision(20) << pc_ptr_->header.stamp.toSec() << endl;
            //            //            cerr << "[CamLidarProcess]\t Set Frame" << endl;
            //

#if 0
            auto tic = ros::Time::now();
            if (vloam_frontend_->track_camlidar(current_)) {
                newPose = current_->Twlidar();
                T_last_cur = vloam_frontend_->T_last_cur();
                ros_client_->publishOdometry(current_->timestamp(), current_->Twlidar());
            }
            auto toc = ros::Time::now();
#endif
            //            cerr << "[CamLidarProcess]\t tracking time and rate : " << (toc - tic).toSec() << ", "
            //                 << 1 / ((toc - tic).toSec()) << endl;
        }

        usleep(5); // 这个休眠挺好的,因为激光的频率为10hz
    }
}

void CamLidarProcess::run()
{
    std::cout << "start CamLidarProcess" << std::endl;
    ros_client_thread_ = new std::thread(&RosClient::run, ros_client_);                    // 接受消息的线程
    camlidar_process_thread_ = new std::thread(&CamLidarProcess::prepare_cam_lidar, this); // 数据的时间戳对齐
    system_thread_ = new std::thread(&CamLidarProcess::track_camlidar, this);              // dometry
}