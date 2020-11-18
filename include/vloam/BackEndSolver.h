//
// Created by ubuntu on 2020/4/29.
//

#pragma once

// std
#include <vector>
#include <condition_variable>
// tbb
#include <tbb/concurrent_queue.h>
#include <tbb/tbb.h>
// Graph
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

// factor
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>


#include "vloam/State.h"
#include "vloam/types.h"
#include "vloam/Config.h"
#include "vloam/Twist.h"

namespace vloam
{

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z, r, p, y)
using gtsam::symbol_shorthand::V; // Vel (xdot, ydot, zdot)
using gtsam::symbol_shorthand::B; // Bias (ax, ay, az, gx, gy, gz)

using Trajectory = std::vector<std::pair<double, gtsam::State>,
                               Eigen::aligned_allocator<std::pair<double, gtsam::State>>>;

class BackEndSolver
{
public:

    BackEndSolver()
    {
        //step1: 构造非线性因子图,　用于存储构造的因子约束
        this->graph_ = new gtsam::NonlinearFactorGraph();
        this->graph_new_ = new gtsam::NonlinearFactorGraph();

        //step2: 配置优化器
        gtsam::ISAM2Params isam_params;
        //　优化器参数的配置
        isam_params.relinearizeThreshold = 0.001;
        isam_params.relinearizeSkip = 1;
        isam_params.cacheLinearizedFactors = false;
        isam_params.enableDetailedResults = true;
        isam_params.factorization = gtsam::ISAM2Params::QR;
        isam_params.print();

        this->isam2_ = new gtsam::ISAM2(isam_params);
        // 为IMU数据的队列预留内存空间
        tbb_imu_times_.set_capacity(300);
        tbb_imu_linaccs_.set_capacity(300);
        tbb_imu_angvel_.set_capacity(300);
    }

    ~BackEndSolver()
    {
        if (this->graph_)
            delete this->graph_;
        if (this->graph_new_)
            delete this->graph_new_;
        if (this->isam2_)
            delete this->isam2_;
    }
    inline bool is_initialized()
    { return systeminitalized_; }

    /// Function that takes in IMU measurement for use in
    /// preintegration measurements
    void addmeasurement_imu(double timestamp, const Eigen::Vector3d &linacc, const Eigen::Vector3d &angvel);

    void addmeasurement_directvisual(double timestamp, const Transformd &T_cur_pre);
    // set initial guess at current state.
    void addImuValues(const FrameId &cur_id,
                      const gtsam::PreintegrationType &pim);

    /// \brief add imu factor between two  frame
    /// \param from_id
    /// \param to_id
    /// \param pim
    void addImuFactor(const FrameId &from_id,
                      const FrameId &to_id,
                      const gtsam::PreintegrationType &pim);

    /// \brief 在两帧图像之间加入相对位姿约束
    /// \param from_id
    /// \param to_id
    /// \param from_id_POSE_to_id
    void addBetweenFactor(const FrameId &from_id,
                          const FrameId &to_id,
                          const gtsam::Pose3 &from_id_POSE_to_id);

    /// \brief this function will optimize the graph
    void optimize();

    // This function return the current state,
    // return origin if we have not initialized yet
    gtsam::State get_state(size_t ct)
    {
        // Ensure valid state
        if (!values_initial_.exists(X(ct)))
            return gtsam::State();
        return gtsam::State(values_initial_.at<gtsam::Pose3>(X(ct)),
                            values_initial_.at<gtsam::Vector3>(V(ct)),
                            values_initial_.at<gtsam::Bias>(B(ct)));
    }
    Trajectory get_trajectory()
    {
        // Return if we do not have any nodes yet
        if (values_initial_.empty()) {
            Trajectory traj;
            traj.push_back(std::make_pair(0., gtsam::State()));

            return traj;
        }

        Trajectory trajectory;
        // Else loop through the state and return them
        for (size_t i = 1; i <= ct_state; ++i) {
            double timestamp = timestamp_lookup[i];
            trajectory.push_back(std::make_pair(timestamp, get_state(i)));
        }

        return trajectory;
    }

private:

    /// Function which will try to initial our graph
    /// using the current IMU measurement
    void trytoinitialize(double timestamp);

    /// Function which imu preintegration initialized
    /// \param prior_state
    /// \return
    bool set_imu_preintegration(const gtsam::State &prior_state);

    /// Function that will reset imu preintegration
    void resetIMUIntegration();
    /// \brief Function will get the predicted Navigation State
    /// based on this generated measurement
    /// \param values_initial
    /// \return
    gtsam::State get_predicted_state(gtsam::Values &values_initial);

    // Function that will create a discrete IMU factor using the GTSAM
    // preintegratior class. This will integrate the current state time
    // up to the new update time
    gtsam::CombinedImuFactor create_imu_factor(double updatetime, gtsam::Values &values_initial);


private:

    //////////////////////////////
    ///////////////
    // 新的因子图
    // New factors that have not been optimized yet
    gtsam::NonlinearFactorGraph *graph_new_;
    // 旧的因子图
    // Master non-linear GTSAM graph, all created factor
    gtsam::NonlinearFactorGraph *graph_;
    ///////////////

    ///////////////
    // 优化的变量
    // New nodes that have not been optimized
    gtsam::Values values_new_;
    // All created nodes
    gtsam::Values values_initial_;
    ///////////////

    ///////////////
    // ISAM2 solvers
    gtsam::ISAM2 *isam2_;
    bool systeminitalized_ = false;
    ///////////////

    std::unordered_map<double, size_t> ct_state_lookup; // ct_state based on timestamp
    std::unordered_map<size_t, double> timestamp_lookup;
    //////////////////////////////

    // IMU Preintegration
    gtsam::PreintegratedCombinedMeasurements *preint_gtsam_;

    // Current ID of state
    size_t ct_state = 0;

    // IMU data from the sensor
    mutable std::mutex imu_mutex_;
    std::condition_variable cv_imu_;
    std::deque<double> imu_times_;
    std::deque<Eigen::Vector3d> imu_linaccs_;
    std::deque<Eigen::Vector3d> imu_angvel_;

    tbb::concurrent_bounded_queue<double> tbb_imu_times_;
    tbb::concurrent_bounded_queue<Eigen::Vector3d> tbb_imu_linaccs_;
    tbb::concurrent_bounded_queue<Eigen::Vector3d> tbb_imu_angvel_;

}; // class BackEndSolver
} // namespce vloam


