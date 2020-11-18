//
// Created by ubuntu on 2020/4/29.
//

#include <ros/ros.h>

#include "vloam/BackEndSolver.h"

namespace vloam
{
void BackEndSolver::addmeasurement_imu(double timestamp, const Eigen::Vector3d &linacc, const Eigen::Vector3d &angvel)
{
    // request access to the imu measurements
    std::unique_lock<std::mutex> lock(imu_mutex_);
    cv_imu_.notify_one();
    // Append this new measurement to the array
    imu_times_.push_back(timestamp);
    imu_linaccs_.push_back(linacc);
    imu_angvel_.push_back(angvel);
}

gtsam::CombinedImuFactor BackEndSolver::create_imu_factor(double updatetime, gtsam::Values &values_initial)
{
    int imucompound = 0;
    {
        // IMU预积分(使用的是前向积分的方法)
        std::unique_lock<std::mutex> lock(imu_mutex_);
//        con.wait(lk, [&]
//        {
//            return (measurements = getMeasurements()).size() != 0;
//        });

        cv_imu_.wait(lock, [&]{
            return imu_times_.size() > 15;
        });

        while (imu_linaccs_.size() > 1 && imu_times_.at(1) <= updatetime) {
            double dt = imu_times_.at(1) - imu_times_.at(0);
            if (dt >= 0) {
                // Our IMU measurement
                Eigen::Vector3d meas_acc;
                Eigen::Vector3d meas_angvel;

                meas_acc = imu_linaccs_.at(0);
                meas_angvel = imu_angvel_.at(0);
                // Preintegrate this measurement!

                preint_gtsam_->integrateMeasurement(meas_acc, meas_angvel, dt);
            }

            imu_linaccs_.erase(imu_linaccs_.begin());
            imu_angvel_.erase(imu_angvel_.begin());
            imu_times_.erase(imu_times_.begin());
        }// 将所有合法的IMU数据进行积分

        double dt_f = updatetime - imu_times_.at(0);

        if (dt_f > 0) {
            // Our IMU measurement
            Eigen::Vector3d meas_acc;
            Eigen::Vector3d meas_angvel;
            meas_acc = imu_linaccs_.at(0);
            meas_angvel = imu_angvel_.at(0);
            // Preintegrate thie measurement
            preint_gtsam_->integrateMeasurement(meas_acc, meas_angvel, dt_f);
            imu_times_.at(0) = updatetime;
            imucompound++;
        }
    }

    return gtsam::CombinedImuFactor(
        X(ct_state), V(ct_state),
        X(ct_state + 1), V(ct_state + 1),
        B(ct_state), B(ct_state + 1),
        *preint_gtsam_);
}

/// \This function will get the predicted state based on the IMU measurement
/// \param values_initial
/// \return
gtsam::State BackEndSolver::get_predicted_state(gtsam::Values &values_initial)
{
    // Get the current state (t=k)
    gtsam::State stateK = gtsam::State(values_initial.at<gtsam::Pose3>(X(ct_state)),
                                       values_initial.at<gtsam::Vector3>(V(ct_state)),
                                       values_initial.at<gtsam::Bias>(B(ct_state)));

    // From this we should predict where we will be at the next time (t=K+1)
    gtsam::NavState stateK1 = preint_gtsam_->predict(gtsam::NavState(stateK.pose(), stateK.v()), stateK.b());

    return gtsam::State(stateK1.pose(), stateK1.v(), stateK.b());
}
void BackEndSolver::addmeasurement_directvisual(double timestamp, const Transformd &T_cur_pre)
{
    // Return if the node already exsit
    if (ct_state_lookup.find(timestamp) != ct_state_lookup.end())
        return;

    // Return if we don't actually have any IMU measurements
    // 因为图像的帧率是10hz,IMU的频率是100hz,所以两帧图像之间至少有10帧IMU数据
    if (imu_times_.size() < 2)
        return;

    // We should try to initialize now
    // Or add the current a new IMU measurement and state

    if (!systeminitalized_) {
        trytoinitialize(timestamp);

        //　Return if we have not initialized the system yet
        if (!systeminitalized_)
            return;
    }
    else {
        // Store our IMU messages as
        // Forster2 discrete pre-integration
        gtsam::CombinedImuFactor imuFactor = create_imu_factor(timestamp, values_initial_);

        graph_new_->add(imuFactor);

        // Predict current state
        gtsam::State newstate = get_predicted_state(values_initial_);

        // Move node count forward in time
        ct_state++;

        // Append to our node vector
        values_new_.insert(X(ct_state), newstate.pose());
        values_new_.insert(V(ct_state), newstate.v());
        values_new_.insert(B(ct_state), newstate.b());
        values_initial_.insert(X(ct_state), newstate.pose());
        values_initial_.insert(V(ct_state), newstate.v());
        values_initial_.insert(B(ct_state), newstate.b());

        // Add ct state to map
        ct_state_lookup[timestamp] = ct_state;
        timestamp_lookup[ct_state] = timestamp;
    }

    // 插入视觉的的相对位姿因子
    // 将transform 转换为gtsam pose 的数据格式,
    Eigen::Matrix3d relaR = T_cur_pre.rotationMatrix();
    Eigen::Vector3d relat = T_cur_pre.pos;
    gtsam::Pose3 relaPose(gtsam::Rot3(relaR), relat);
    addBetweenFactor(ct_state-1, ct_state, relaPose);
}

/// This will try to take the current IMU vector and initialize
/// If there are enough IMU, we should find the current orientation

/// \param timestamp
void BackEndSolver::trytoinitialize(double timestamp)
{
    /// If we have already initialized, then just return
    if (systeminitalized_)
        return;

    // wait for enough IMu reading
    if (imu_times_.size() < (size_t) (Config::cfg()->backEndParams().initialImuWait_));

    //================================
    // INITIALIZE!
    //================================

    std::unique_lock<std::mutex> lock(imu_mutex_);

    // Sum up our current accleration and velocities
    Eigen::Vector3d linsum = Eigen::Vector3d::Zero();
    Eigen::Vector3d angsum = Eigen::Vector3d::Zero();


    for (size_t i = 0; i < imu_times_.size() and imu_times_[i] <= timestamp; i++) {
        linsum += imu_linaccs_.at(i);
        angsum += imu_angvel_.at(i);
    }

    // Calculate the mean of the linear acceleration and angular velocity
    Eigen::Vector3d linavg = Eigen::Vector3d::Zero();
    Eigen::Vector3d angavg = Eigen::Vector3d::Zero();
    linavg = linsum / imu_times_.size();
    angavg = angsum / imu_times_.size();

    ROS_INFO("\033[0;32m[INIT]:accel mean= %.4f, %.4f, %.4f\033[0m", linavg(0), linavg(1), linavg(2));
    ROS_INFO("\033[0;32m[INIT]:gyro mean= %.4f, %.4f, %.4f\033[0m", angavg(0), angavg(1), angavg(2));

    Eigen::Vector3d normal_g;
    normal_g << 0., 0., -9.81;

    Eigen::Quaterniond q_IG = Eigen::Quaterniond::FromTwoVectors(-normal_g, linavg);
    Matrix3x3d R_IG = q_IG.toRotationMatrix();
    Eigen::Quaterniond q_GI = q_IG.inverse();

    Eigen::Vector3d Rg = q_IG * normal_g;
    ROS_INFO("\033[0;32m[INIT]:Rg= %.4f, %.4f, %.4f\033[0m", Rg(0), Rg(1), Rg(2));
    Eigen::Vector3d ba = q_IG * normal_g + linavg;
    Eigen::Vector3d bg = angavg;
    gtsam::Point3 p_IinG(0., 0., 0.);
    Eigen::Vector3d prior_vel{0., 0., 0.};
    gtsam::Pose3 prior_pose =
        gtsam::Pose3(gtsam::Quaternion(q_GI.w(), q_GI.x(), q_GI.y(), q_GI.z()),
                     p_IinG);

    //===========================================
    // CREATE PRIOR FRACTORS AND INITIALZE GRAPHS
    //===========================================

    // Create our prior factor and add it to our graph

    gtsam::State prior_state = gtsam::State(prior_pose,
                                            prior_vel,
                                            gtsam::Bias(ba, bg));

    // Set initial covariance for inertial factors
    // Set initial pose uncertainty: constrain mainly position and global yaw.
    // roll and pitch is observable, therefore low variance.

    const auto &backend_params = Config::cfg()->backEndParams();
    Matrix6x6d pose_prior_covariance = Matrix6x6d::Zero();
    pose_prior_covariance.diagonal()[0] = pow(backend_params.initialRollPitchSigma_, 2);
    pose_prior_covariance.diagonal()[1] = pow(backend_params.initialRollPitchSigma_, 2);
    pose_prior_covariance.diagonal()[2] = pow(backend_params.initialYawSigma_, 2);
    pose_prior_covariance.diagonal()[3] = pow(backend_params.initialPositionSigma_, 2);
    pose_prior_covariance.diagonal()[4] = pow(backend_params.initialPositionSigma_, 2);
    pose_prior_covariance.diagonal()[5] = pow(backend_params.initialPositionSigma_, 2);


    // Rotate initial uncertainty into local frame, where the uncertainty is specified.
    pose_prior_covariance.topLeftCorner(3, 3) = R_IG * pose_prior_covariance.topLeftCorner(3, 3) * R_IG.transpose();
    // Add pose prior.
    gtsam::SharedNoiseModel noise_init_pose =
        gtsam::noiseModel::Gaussian::Covariance(pose_prior_covariance);

    graph_new_->add(gtsam::PriorFactor<gtsam::Pose3>(X(ct_state), prior_state.pose(), noise_init_pose));
    graph_->add(gtsam::PriorFactor<gtsam::Pose3>(X(ct_state), prior_state.pose(), noise_init_pose));
    // Add initial velocity prior
    gtsam::SharedNoiseModel noise_init_vel_prior
        = gtsam::noiseModel::Isotropic::Sigma(3, backend_params.initialVelocitySigma_);

    graph_new_->add(gtsam::PriorFactor<gtsam::Vector3>(V(ct_state), prior_state.v(), noise_init_vel_prior));
    graph_->add(gtsam::PriorFactor<gtsam::Vector3>(V(ct_state), prior_state.v(), noise_init_vel_prior));

    // Add initial bias priors
    Vector6d prior_biasSigmas;
    prior_biasSigmas.head<3>().setConstant(backend_params.initialAccBiasSigma_);
    prior_biasSigmas.tail<3>().setConstant(backend_params.initialGyroBiasSigma_);

    gtsam::SharedNoiseModel imu_bias_prior_noise =
        gtsam::noiseModel::Diagonal::Sigmas(prior_biasSigmas);

    graph_new_->add(gtsam::PriorFactor<gtsam::Bias>(B(ct_state), prior_state.b(), imu_bias_prior_noise));
    graph_->add(gtsam::PriorFactor<gtsam::Bias>(B(ct_state), prior_state.b(), imu_bias_prior_noise));

    //Add initai state

    // Add initial state to the graph
    values_new_.insert(X(ct_state), prior_state.pose());
    values_new_.insert(V(ct_state), prior_state.v());
    values_new_.insert(B(ct_state), prior_state.b());
    values_initial_.insert(X(ct_state), prior_state.pose());
    values_initial_.insert(V(ct_state), prior_state.v());
    values_initial_.insert(B(ct_state), prior_state.b());

    // Add ct_state to the map
    ct_state_lookup[timestamp] = ct_state;
    timestamp_lookup[ct_state] = timestamp;

    // clear all old imu messages (keep the last two)
    imu_times_.erase(imu_times_.begin(), imu_times_.end() - 1);
    imu_linaccs_.erase(imu_linaccs_.begin(), imu_linaccs_.end() - 1);
    imu_angvel_.erase(imu_angvel_.begin(), imu_angvel_.end() - 1);

    // set our initialized to true
    systeminitalized_ = true;

    // Debug info
    ROS_INFO("\033[0;32m[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\033[0m", q_GI.x(), q_GI.y(), q_GI.z(), q_GI.w());
    ROS_INFO("\033[0;32m[INIT]: position = %.4f, %.4f, %.4f\033[0m", p_IinG(0), p_IinG(1), p_IinG(2));
    ROS_INFO("\033[0;32m[INIT]: velocity = %.4f, %.4f, %.4f\033[0m", prior_vel(0), prior_vel(1), prior_vel(2));
    ROS_INFO("\033[0;32m[INIT]: bias gyro = %.4f, %.4f, %.4f\033[0m", bg(0), bg(1), bg(2));
    ROS_INFO("\033[0;32m[INIT]: bias accel = %.4f, %.4f, %.4f\033[0m", ba(0), ba(1), ba(2));
}

bool BackEndSolver::set_imu_preintegration(const gtsam::State &prior_state)
{

    boost::shared_ptr<gtsam::PreintegratedCombinedMeasurements::Params> params;

    const auto &backendParams = Config::cfg()->backEndParams();
    const auto &gravity = Config::cfg()->backEndParams().n_gravity_;
    const auto &imuParams = Config::cfg()->imuParams();
    params = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(gravity[2]);

    params->setAccelerometerCovariance(gtsam::I_3x3 * std::pow(imuParams.accel_noise, 2));
    params->setGyroscopeCovariance(gtsam::I_3x3 * std::pow(imuParams.gyro_noise, 2));
    params->setIntegrationCovariance(gtsam::I_3x3 * std::pow(imuParams.imu_integration_sigma, 2));
    params->setUse2ndOrderCoriolis(false);

    params->biasAccOmegaInt = 1e-5 * gtsam::I_6x6;
    ///< continuous-time "Covariance" describing
    ///< accelerometer bias random walk
    params->biasAccCovariance =
        std::pow(imuParams.accel_walk, 2.0) * Eigen::Matrix3d::Identity();
    params->biasOmegaCovariance =
        std::pow(imuParams.gyro_walk, 2.0) * Eigen::Matrix3d::Identity();

    // Actually create the GTSAM preintegration
    preint_gtsam_ = new gtsam::PreintegratedCombinedMeasurements(params, prior_state.b());

    return true;
}

void BackEndSolver::resetIMUIntegration()
{
    // use the optimized bias to reset integration
    if (values_initial_.exists(B(ct_state))) {
        preint_gtsam_->resetIntegrationAndSetBias(
            values_initial_.at<gtsam::Bias>(B(ct_state)));
    }
}
/*
 * If we should optimize using the fixed lag smoothers
 * We will send in new measurements and nodes, and
 * get back the estimate of the current state
 */
void BackEndSolver::optimize()
{
    // Return if not initialized
    if (!systeminitalized_ && ct_state < 2)
        return;

    // Perform smoothing update
    try {
        gtsam::ISAM2Result result = isam2_->update(*graph_new_, values_new_);
        values_initial_ = isam2_->calculateEstimate();
    }
    catch (gtsam::IndeterminantLinearSystemException &e) {
        ROS_ERROR("FORSTER2 gtsam indeterminate linear system exception!");
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    // remove the used up nodes
    values_new_.clear();
    // remove the ussed up factors
    graph_new_->resize(0);
    // reset imu preintegration
    resetIMUIntegration();
}


void
BackEndSolver::addBetweenFactor(
    const FrameId &from_id,
    const FrameId &to_id,
    const gtsam::Pose3 &from_id_POSE_to_id)
{
    const auto &backendParams = Config::cfg()->backEndParams();

    Vector6d precisions;
    precisions[0] = backendParams.betweenRotXSigma_;
    precisions[1] = backendParams.betweenRotYSigma_;
    precisions[2] = backendParams.betweenRotZSigma_;
    precisions[3] = backendParams.betweenTransXSigma_;
    precisions[4] = backendParams.betweenTransYSigma_;
    precisions[5] = backendParams.betweenTransZSigma_;

    Matrix6x6d between_covariance = Matrix6x6d::Zero();
    between_covariance.diagonal()[0] = std::pow(precisions[0],2);
    between_covariance.diagonal()[1] = std::pow(precisions[1],2);
    between_covariance.diagonal()[2] = std::pow(precisions[2],2);
    between_covariance.diagonal()[3] = std::pow(precisions[3],2);
    between_covariance.diagonal()[4] = std::pow(precisions[4],2);
    between_covariance.diagonal()[5] = std::pow(precisions[5],2);

    gtsam::SharedNoiseModel between_noise =
        gtsam::noiseModel::Gaussian::Covariance(between_covariance);

    auto betweenfactor = gtsam::BetweenFactor<gtsam::Pose3>(
        X(from_id),
        X(to_id),
        from_id_POSE_to_id);

    graph_new_->add(betweenfactor);
}

} //namespace vloam