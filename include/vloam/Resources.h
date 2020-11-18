//
// Created by ubuntu on 2020/3/9.
//

#pragma once
#include <string>
#include <boost/lockfree/queue.hpp>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <Eigen/Core>

#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>

template<typename T, typename Container=std::deque<T> >
class iterable_queue : public std::queue<T,Container>
{
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;

    iterator begin() { return this->c.begin(); }
    iterator end() { return this->c.end(); }
    const_iterator begin() const { return this->c.begin(); }
    const_iterator end() const { return this->c.end(); }
};

typedef struct _camlidar_topic_t camlidar_topic_t;
struct _camlidar_topic_t {
    std::string cam_topic;
    std::string lidar_topic;
};


typedef struct _cam_lidar_queue_t cam_lidar_queue_t;
struct _cam_lidar_queue_t {
    _cam_lidar_queue_t() {}
    ~_cam_lidar_queue_t() {}
    iterable_queue<sensor_msgs::ImageConstPtr> img_queue;
    iterable_queue<sensor_msgs::PointCloud2ConstPtr> pc_queue;
    //    std::vector<sensor_msgs::Image> img_queue;
    //    std::vector<sensor_msgs::PointCloud2> pc_queue;
    std::mutex mtx_camlidar;
    std::condition_variable cond_camlidar;
    std::mutex mtx_img;
    std::condition_variable cond_img;
    std::mutex mtx_pc;
    std::condition_variable cond_pc;
};

