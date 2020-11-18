#include <vloam/Config.h>

using namespace std;

namespace vloam
{

Config *Config::instance_ = nullptr;

Config::Config(string path, string fname)
    : path_(path), fname_(fname)
{
    std::cerr << std::endl;
    std::cerr << "[Configuration]\t Load configuration from \"" << fname_ << "\"" << std::endl;

    cv::FileStorage f_hw_settings(path_ + fname_, cv::FileStorage::READ);
    camera_info_.width = f_hw_settings["Camera.width"];
    camera_info_.height = f_hw_settings["Camera.height"];
    camera_info_.fx = f_hw_settings["Camera.fx"];
    camera_info_.fy = f_hw_settings["Camera.fy"];
    camera_info_.cx = f_hw_settings["Camera.cx"];
    camera_info_.cy = f_hw_settings["Camera.cy"];
    camera_info_.k1 = f_hw_settings["Camera.k1"];
    camera_info_.k2 = f_hw_settings["Camera.k2"];
    camera_info_.p1 = f_hw_settings["Camera.p1"];
    camera_info_.p2 = f_hw_settings["Camera.p2"];
    camera_info_.d0 = f_hw_settings["Camera.d0"];
    camera_info_.d1 = f_hw_settings["Camera.d1"];
    camera_info_.d2 = f_hw_settings["Camera.d2"];
    camera_info_.d3 = f_hw_settings["Camera.d3"];
    camera_info_.d4 = f_hw_settings["Camera.d4"];
    camera_info_.image_type = f_hw_settings["Camera.RGB"];

    cv::Mat T;
    f_hw_settings["extrinsicMatrix"] >> T;
    cv::cv2eigen(T, camlidar_.extrinsic);

    tracker_.levels = f_hw_settings["Tracker.levels"];
    tracker_.min_level = f_hw_settings["Tracker.min_level"];
    tracker_.max_level = f_hw_settings["Tracker.max_level"];
    tracker_.max_iteration = f_hw_settings["Tracker.max_iteration"];

    tracker_.scale_estimator = string(f_hw_settings["Tracker.scale_estimator"]);
    tracker_.weight_function = string(f_hw_settings["Tracker.weight_function"]);

    tracker_.set_scale_estimator_type();
    tracker_.set_weight_function_type();

    // step read featuretracking paramter
    featuretracking_.bflowBack = int(f_hw_settings["FeatureTracking.flowBack"]) > 0;
    featuretracking_.ishowSkipNum = f_hw_settings["FeatureTracking.showSkipNum"];
    featuretracking_.ishowDSRate = f_hw_settings["FeatureTracking.showDSRate"];
    featuretracking_.imageWidth = f_hw_settings["FeatureTracking.imageWidth"];
    featuretracking_.imageHeight = f_hw_settings["FeatureTracking.imageHeight"];
    featuretracking_.imaxFeatureNumPerSubregion = f_hw_settings["FeatureTracking.maxFeatureNumPerSubregion"];
    featuretracking_.ixSubregionNum = f_hw_settings["FeatureTracking.xSubregionNum"];
    featuretracking_.iySubregionNum = f_hw_settings["FeatureTracking.ySubregionNum"];
    featuretracking_.ixBoundary = f_hw_settings["FeatureTracking.xBoundary"];
    featuretracking_.iyBoundary = f_hw_settings["FeatureTracking.yBoundary"];
    featuretracking_.dmaxTrackDis = f_hw_settings["FeatureTracking.maxTrackDis"];

    //////////////////
    // 后端的参数
    backEndParams_.autoInitialize_ = f_hw_settings["backEndParams.autoInitialize"];
    backEndParams_.initialPositionSigma_ = f_hw_settings["backEndParams.initialPositionSigma"];
    backEndParams_.initialRollPitchSigma_ = f_hw_settings["backEndParams.initialRollPitchSigma"];
    backEndParams_.initialYawSigma_ = f_hw_settings["backEndParams.initialYawSigma"];
    backEndParams_.initialVelocitySigma_ = f_hw_settings["backEndParams.initialVelocitySigma"];
    backEndParams_.initialAccBiasSigma_ = f_hw_settings["backEndParams.initialAccBiasSigma"];
    backEndParams_.initialGyroBiasSigma_ = f_hw_settings["backEndParams.initialGyroBiasSigma"];
    f_hw_settings["backEndParams.n_gravity"] >>backEndParams_.n_gravity_;

    std::cout << "gravity: [";
    for(const auto& ele : backEndParams_.n_gravity_)
    {
        std::cout <<" " << ele;
    }
    std::cout << " ]" << std::endl;

    backEndParams_.roundOnAutoInitialize_ = int(f_hw_settings["backEndParams.roundOnAutoInitialize"]) != 0;

    // BetweenPoseFactor variance
    backEndParams_.betweenRotXSigma_ = f_hw_settings["backEndParams.betweenRotXSigma"];
    backEndParams_.betweenRotYSigma_ = f_hw_settings["backEndParams.betweenRotYSigma"];
    backEndParams_.betweenRotZSigma_ = f_hw_settings["backEndParams.betweenRotZSigma"];
    backEndParams_.betweenTransXSigma_ = f_hw_settings["backEndParams.betweenTransXSigma"];
    backEndParams_.betweenTransYSigma_ = f_hw_settings["backEndParams.betweenTransYSigma"];
    backEndParams_.betweenTransZSigma_ = f_hw_settings["backEndParams.betweenTransZSigma"];
    //////////////////

    /////////////////
    /// IMU Parameter
    imuParams_.gyro_noise = f_hw_settings["imuParams.gyro_noise"];    // [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
    imuParams_.gyro_walk = f_hw_settings["imuParams.gyro_walk"];      // [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
    imuParams_.accel_noise = f_hw_settings["imuParams.accel_noise"];  // [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
    imuParams_.accel_walk = f_hw_settings["imuParams.accel_walk"];    // [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )

    // Extra imu parameters
    imuParams_.imu_integration_sigma = f_hw_settings["imuParams.imu_integration_sigma"];
    imuParams_.imu_time_shift = f_hw_settings["imuParams.imu_time_shift"];
    /////////////////

    std::cout << "--------IMU Paramter--------";
    std::cout << "\n   gyro_noise: \t" << imuParams_.gyro_noise;
    std::cout << "\n   gyro_walk: \t" << imuParams_.gyro_walk;
    std::cout << "\n   accel_noise: \t" << imuParams_.accel_noise;
    std::cout << "\n   accel_walk: \t" << imuParams_.accel_walk;
    std::cout << "\n   imu_integration_sigma: \t" << imuParams_.imu_integration_sigma;
    std::cout << "\n   imu_time_shift: \t" << imuParams_.imu_time_shift;

    std::cout << "\n--------IMU Paramter--------" << std::endl;

    std::string voca_fname = string(f_hw_settings["LoopClosure.f_vocabulary"]);

    loopclosure_.BoW_fname = path_ + voca_fname;
    std::cerr << "[CamLidarProcess]\t Set vocabulary file : " << path_ + voca_fname << std::endl;

    camera_.reset(new PinholeModel(camera_info_.width,
                                   camera_info_.height,
                                   camera_info_.fx,
                                   camera_info_.fy,
                                   camera_info_.cx,
                                   camera_info_.cy,
                                   camera_info_.d0,
                                   camera_info_.d1,
                                   camera_info_.d2,
                                   camera_info_.d3,
                                   camera_info_.d4));

    show_configuration();
}

Config::Config()
//    :num_levels_(5), min_level_(2), max_level_(4), max_iterations_(100)
{
    show_configuration();
}

Config::Config(int num_levels, int min_level, int max_level, int max_iterations)
//    :num_levels_(num_levels), min_level_(min_level), max_level_(max_level), max_iterations_(max_iterations)
{
    show_configuration();
}

void Config::show_configuration()
{
    std::cerr << std::endl;
    std::cerr << "[Configuration]\t Camera information" << std::endl;
    std::cerr << "[Configuration]\t width : " << camera_info_.width << std::endl;
    std::cerr << "[Configuration]\t height : " << camera_info_.height << std::endl;
    std::cerr << "[Configuration]\t fx : " << camera_info_.fx << std::endl;
    std::cerr << "[Configuration]\t fy : " << camera_info_.fy << std::endl;
    std::cerr << "[Configuration]\t cx : " << camera_info_.cx << std::endl;
    std::cerr << "[Configuration]\t cy : " << camera_info_.cy << std::endl;
    std::cerr << "[Configuration]\t distortion d[5] : [ " << camera_info_.d0 << ", " << camera_info_.d1 << ", "
              << camera_info_.d2 << ", " << camera_info_.d3 << ", " << camera_info_.d4 << "]" << std::endl;

    std::cerr << std::endl;
    std::cerr << "[Configuration]\t camera-lidar information" << std::endl;
    std::cerr << camlidar_.extrinsic.matrix() << std::endl;

    std::cerr << std::endl;
    std::cerr << "[Configuration]\t Tracker information" << std::endl;
    std::cerr << "[Configuration]\t The number of pyramid level: " << tracker_.levels << std::endl;
    std::cerr << "[Configuration]\t The minimum pyramid level: " << tracker_.min_level << std::endl;
    std::cerr << "[Configuration]\t The maximum pyramid level: " << tracker_.max_level << std::endl;
    std::cerr << "[Configuration]\t The maximum pyramid level: " << tracker_.max_iteration << std::endl;

    string tmp_weight_scale = (tracker_.use_weight_scale) ? "true" : "false";
    std::cerr << "[Configuration]\t Use weight scale: " << tmp_weight_scale << std::endl;
    std::cerr << "[Configuration]\t Scale Estimator: " << tracker_.scale_estimator << std::endl;
    std::cerr << "[Configuration]\t Weight Function: " << tracker_.weight_function << std::endl;

    std::cerr << std::endl;
    std::cerr << "[Configuration]\t LoopClosure information" << std::endl;
    std::cerr << "[Configuration]\t Vocaburaly file : " << loopclosure_.BoW_fname << std::endl;
}

Config *Config::cfg()
{
    if (instance_ == NULL)
        instance_ = new Config();

    return instance_;
}

Config *Config::cfg(string path, string fname)
{
    if (instance_ == NULL)
        instance_ = new Config(path, fname);

    return instance_;
}

Config *Config::cfg(int num_levels, int min_level, int max_level, int max_iterations)
{
    if (instance_ == NULL)
        instance_ = new Config(num_levels, min_level, max_level, max_iterations);

    return instance_;
}

} // namespace vloam
