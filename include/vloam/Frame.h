#pragma once

#include <thread>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

// #include <opencv2/features2d.hpp>
// #include "ORBExtractor.h"
// #include "ORBVocabulary.h"
// #include "DBoW2/BowVector.h"
// #include "DBoW2/FeatureVector.h"

#include "Datatypes.h"
#include "Config.h"
#include "Point.h"
#include "CameraModel.h"
#include "vloam/Twist.h"
#include <mutex>


namespace vloam
{

class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 内存对齐

    typedef shared_ptr<Frame> Ptr;

    Frame(const cv::Mat &img);
    Frame(const int64_t timestamp, const cv::Mat &img, PointCloud &pointcloud, CameraModel::Ptr camera);
    ~Frame();

    cv::Mat &original_img()
    { return original_img_; }
    cv::Mat &level(size_t idx);

    pcl::PointCloud<pcl::PointXYZI> &xyzipc();
    PointCloud &pointcloud();
    PointCloud &original_pointcloud();
    int64_t timestamp()
    { return utime_; } //
    CameraModel::Ptr camera()
    { return camera_; }

    Transformf &Twc()
    { return Twc_; }
    Transformd dTwc()
    { return Twc_.cast<double>(); }
    Matrix3x4 Twlidar();
    void Twc(const Transformf &Twc);

    int get_pointcloud_size()
    { return pointcloud_.size(); }

    void show_pointcloud();
    void show_image_with_points(cv::Mat &img, size_t num_level);
    void save_image_with_points(size_t num_level, int id);

    inline
    static void jacobian_xyz2uv(const Eigen::Vector3f &xyz_in_f, Matrix2x6 &J)
    {
        const float x = xyz_in_f[0];
        const float y = xyz_in_f[1];
        const float z_inv = 1. / xyz_in_f[2];
        const float z_inv_2 = z_inv * z_inv;

        J(0, 0) = -z_inv;               // -1/z
        J(0, 1) = 0.0;                  // 0
        J(0, 2) = x * z_inv_2;          // x/z^2
        J(0, 3) = y * J(0, 2);          // x*y/z^2
        J(0, 4) = -(1.0 + x * J(0, 2)); // -(1.0 + x^2/z^2)
        J(0, 5) = y * z_inv;            // y/z

        J(1, 0) = 0.0;               // 0
        J(1, 1) = -z_inv;            // -1/z
        J(1, 2) = y * z_inv_2;       // y/z^2
        J(1, 3) = 1.0 + y * J(1, 2); // 1.0 + y^2/z^2
        J(1, 4) = -J(0, 3);          // -x*y/z^2
        J(1, 5) = -x * z_inv;        // x/z
    }

private:
    void initialize_img(const cv::Mat img);
    void initialize_pc();
    void initialize_pc_OMP();

    int64_t utime_;

    CameraModel::Ptr camera_ = nullptr;

    cv::Mat original_img_;
    ImgPyramid img_pyramid_;

    pcl::PointCloud<pcl::PointXYZI> xyzipc_;
    PointCloud pointcloud_;
    PointCloud original_pointcloud_;

    int num_levels_;
    int max_level_;

    Transformf Twc_;
    mutex Twc_mutex_;
};

/// Creates an image pyramid of half-sampled images.
void create_image_pyramid(const cv::Mat &img_level_0, int n_levels, ImgPyramid &pyr);

class FrameDB
{
public:
    typedef shared_ptr<FrameDB> Ptr;

    FrameDB()
    {}
    ~FrameDB()
    {}

    void add(Frame::Ptr frame)
    {
        unique_lock<mutex> ul{DB_mutex_};
        frameDB_.push_back(frame);
        ul.unlock();
    }

    vector<Frame::Ptr>::iterator begin()
    { return frameDB_.begin(); }
    vector<Frame::Ptr>::iterator end()
    { return frameDB_.end(); }
    size_t size()
    { return frameDB_.size(); }
    vector<Frame::Ptr> &frameDB()
    { return frameDB_; }

private:
    vector<Frame::Ptr> frameDB_;
    mutex DB_mutex_;
};

} // namespace dedvo
