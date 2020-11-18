#pragma once
#include <utility>
#include "Frame.h"
#include "CameraModel.h"
#include "PinholeModel.h"

namespace vloam
{

class Keyframe
{
public:
    typedef shared_ptr<Keyframe> Ptr;
    Keyframe(Frame::Ptr frame, int id_, CameraModel::Ptr camera);
    Keyframe(Frame::Ptr frame, CameraModel::Ptr camera);
    ~Keyframe();

    Frame::Ptr frame()
    {
        return frame_;
    }

    int pointcloud_size() { return pointcloud_.size(); }
    //    vector<Point, Eigen::aligned_allocator<Eigen::Vector3f>>& pointcloud() { return pointcloud_; }
    PointCloud &pointcloud() { return pointcloud_; }
    //    cv::Subdiv2D& subdiv() { return subdiv_; }

    void id(int id) { id_ = id; }
    int &id() { return id_; }

    void point_sampling();
    void parent(Keyframe::Ptr parent) { parent_ = parent; }
    void child(Keyframe::Ptr child) { child_ = child; }

    float get_visible_ratio(const Keyframe::Ptr keyframe);
    Keyframe::Ptr parent() { return parent_; }
    Keyframe::Ptr child() { return child_; }

    void first_connection(bool first_connection) { first_connection_ = first_connection; }
    bool first_connection() { return first_connection_; }

    void show_image_with_points(cv::Mat &img, size_t num_level);

private:
    int id_;
    Frame::Ptr frame_;
    CameraModel::Ptr camera_;
    PinholeModel::Ptr pinhole_model_;

    PointCloud pointcloud_; // 降采样的点云

    bool first_connection_;
    Keyframe::Ptr parent_;
    Keyframe::Ptr child_;
}; //class Keyframe

} // namespace vloam
