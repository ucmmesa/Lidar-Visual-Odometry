#pragma once
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

namespace vloam
{

typedef pcl::PointXYZRGBA POINT;
typedef pcl::PointCloud<POINT> PointCloud;

class Point
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Point();
    Point(Eigen::Vector3f pnt);
    Point(float x, float y, float z);
    ~Point();

    Eigen::Vector3f point;

    Point& operator=(const Point& lhs);
    // Point& operator=(const Eigen::Vector3f& lhs);
};

} // vloam
