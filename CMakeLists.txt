cmake_minimum_required(VERSION 2.8.3)
project(aloam_velodyne)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wcomment -g")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
  )

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Gflags)
find_package(Glog)
find_package(GTSAM REQUIRED)
# find_package(MKL REQUIRED)
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})

find_package(Boost COMPONENTS system thread timer)
include_directories(${GFLAGS_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})

include_directories(thirdparty/Sophus)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS include
)

link_libraries(${GFLAGS_LIBRARIES})
link_libraries(${GLOG_LIBRARIES})

add_library(vloam
  src/aloam_velodyne/utility.cpp
  src/vloam/Config.cpp
  src/vloam/PinholeModel.cpp
  src/vloam/Point.cpp
  src/vloam/Frame.cpp
  src/vloam/Keyframe.cpp
  src/vloam/KeyframeDB.cpp
  src/vloam/WeightFunction.cpp
  src/vloam/featureTracking.cpp
  src/vloam/Tracker2.cpp
  src/vloam/Frontend.cpp
  src/vloam/KeyframeWindow.cpp
  src/vloam/WindowOptimizer.cpp
  src/vloam/BackEndSolver.cpp
  src/vloam/RosClient.cpp
  src/vloam/CamLidarProcess.cpp
  src/Optimization/FrameParameterization.cpp
  src/Optimization/FrameTracker.cpp
  )

target_link_libraries(vloam
  ${TBB_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  gtsam
 )

add_executable(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry
  vloam
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  gtsam
  Boost::system
  Boost::thread
  Boost::timer
  ${TBB_LIBRARIES}
  )

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  Boost::system
  Boost::thread
  Boost::timer
  ${TBB_LIBRARIES})

add_executable(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Boost::system
  Boost::thread
  Boost::timer
  ${TBB_LIBRARIES})

add_executable(adjustPointCloud src/adjustPointCloud.cpp)
target_link_libraries(adjustPointCloud
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Boost::system
  Boost::thread
  Boost::timer
  ${TBB_LIBRARIES})


add_executable(test_quaterion src/test_quaterion.cpp)