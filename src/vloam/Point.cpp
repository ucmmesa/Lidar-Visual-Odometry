//
// Created by ubuntu on 2020/3/9.
//

#include <vloam/Point.h>

namespace vloam
{
Point::Point()
{

}

Point::Point(Eigen::Vector3f pnt)
    :point(pnt)
{
}

Point::Point(float x, float y, float z)
{
    point << x, y, z;
}

Point::~Point()
{

}

Point& Point::operator=(const Point& lhs)
{
    point = lhs.point;

    return *this;
}

} //vloam
