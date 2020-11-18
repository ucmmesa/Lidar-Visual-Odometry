//
// Created by ubuntu on 2020/4/2.
//

#ifndef POINTDEFINITION_H
#define POINTDEFINITION_H

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

struct ImagePoint {
    float u, v;
    int ind;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (ImagePoint,
                                   (float, u, u)
                                       (float, v, v)
                                       (int, ind, ind))

struct DepthPoint {
    float u, v;  //图像坐标
    float depth; //深度值
    int label;   //用来判断深度值是三角化还是深度图直接得到
    int ind;     //用来判断是角度还是平移
};

POINT_CLOUD_REGISTER_POINT_STRUCT (DepthPoint,
                                   (float, u, u)
                                       (float, v, v)
                                       (float, depth, depth)
                                       (int, label, label)
                                       (int, ind, ind))


#endif //POINTDEFINITION_H
