#pragma once
#include <iostream>
#include <string>

#include "Datatypes.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "vloam/CameraModel.h"
#include "vloam/PinholeModel.h"
#include "vloam/WeightFunction.h"


#define Deg2rad(x) ((x) /180.0*M_PI)
namespace vloam
{
typedef struct _camera_t camera_t;

struct _camera_t
{
    int width;
    int height;

    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
    float d0, d1, d2, d3, d4;

    int image_type;
};

typedef struct _featuretracking featuretracking_t;

struct _featuretracking
{
    bool bflowBack = true;
    int ishowSkipNum = 2;
    int ishowDSRate = 2;

    int imageWidth = 1241;
    int imageHeight = 376;

    int imaxFeatureNumPerSubregion = 5;
    int ixSubregionNum = 28;
    int iySubregionNum = 10;

    int ixBoundary = 20;
    int iyBoundary = 20;

    double dmaxTrackDis = 255;
};

typedef struct _imuParams imuParams_t;

struct _imuParams
{
    double gyro_noise = 01.6968e-04;    // [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
    double gyro_walk = 1.9393e-05;       // [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
    double accel_noise = 2.0000e-3;    // [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
    double accel_walk = 3.0000e-3;   // [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )

    double imu_integration_sigma = 1.0e-8;
    double imu_time_shift = 0.0;
};

typedef struct _backEndParams backEndParams_t;

struct _backEndParams
{
public:
    int autoInitialize_ = 0;
    int initialImuWait_ = 30;
    double initialPositionSigma_ = 0.00001;
    double initialRollPitchSigma_ = 10.0 / 180 * M_PI;
    double initialYawSigma_ = 0.1 / 180 * M_PI;
    double initialVelocitySigma_ = 1e-3;
    double initialAccBiasSigma_ = 0.1;
    double initialGyroBiasSigma_ = 0.01;

    bool roundOnAutoInitialize_ = false;

    // BetweenPoseFactor variance
    double betweenRotXSigma_ = Deg2rad(3);
    double betweenRotYSigma_ = Deg2rad(3);
    double betweenRotZSigma_ = Deg2rad(3);
    double betweenTransXSigma_ = 0.1;
    double betweenTransYSigma_ = 0.1;
    double betweenTransZSigma_ = 0.2;

    vector<double> n_gravity_;
};

typedef struct _camlidar_t camlidar_t;

struct _camlidar_t
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3x4 extrinsic;
};

typedef struct _loopclosure_t loopclosure_t;

struct _loopclosure_t
{
    string BoW_fname;
};

typedef struct _ORBExtractor_t ORBExtractor_t;

struct _ORBExtractor_t
{
    int num_features;
    int scale_factor;
    int num_levels;
    int iniThFAST;
    int minThFAST;
};

typedef struct _tracker_t tracker_t;

struct _tracker_t
{
    int levels;
    int min_level;
    int max_level;
    int max_iteration;

    bool use_weight_scale = true;
    string scale_estimator;
    string weight_function;

    ScaleEstimatorType scale_estimator_type;
    WeightFunctionType weight_function_type;

    void set_scale_estimator_type()
    {
        if (!scale_estimator.compare("None"))
            use_weight_scale = false;
        if (!scale_estimator.compare("TDistributionScale"))
            scale_estimator_type = ScaleEstimatorType::TDistributionScale;

        cerr << "ScaleType : " << static_cast<int>(scale_estimator_type);
    }

    void set_weight_function_type()
    {
        if (!weight_function.compare("TDistributionWeight"))
            weight_function_type = WeightFunctionType::TDistributionWeight;
    }
};

class Config
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static Config *cfg();
    static Config *cfg(std::string path, std::string fname);
    static Config *cfg(int num_levels, int min_level, int max_level, int max_iterations);

    static std::string &path()
    { return cfg()->path_; }
    static std::string &fname()
    { return cfg()->fname_; }
    static camera_t &camera_info()
    { return cfg()->camera_info_; }
    static camlidar_t &camlidar()
    { return cfg()->camlidar_; }
    static loopclosure_t &loopclosure()
    { return cfg()->loopclosure_; }
    static tracker_t &tracker()
    { return cfg()->tracker_; }

    static featuretracking_t &featuretracking()
    { return cfg()->featuretracking_; }

    static backEndParams_t &backEndParams()
    {
        return cfg()->backEndParams_;
    }

    static imuParams_t & imuParams()
    {
        return cfg()->imuParams_;
    }
    static CameraModel::Ptr &camera()
    { return cfg()->camera_; }

    static int &num_levels()
    { return cfg()->tracker_.levels; }
    static int &min_level()
    { return cfg()->tracker_.min_level; }
    static int &max_level()
    { return cfg()->tracker_.max_level; }

    static int &max_iterations()
    { return cfg()->tracker_.max_iteration; }

    void show_configuration();

private:
    Config();
    Config(std::string path, std::string fname);
    Config(int num_levels, int min_level, int max_level, int max_iterations);
    Config(const Config &other);
    ~Config()
    {}

    static Config *instance_;

    // configure from file
    std::string path_;
    std::string fname_;
    camera_t camera_info_;
    camlidar_t camlidar_;
    loopclosure_t loopclosure_;
    tracker_t tracker_;
    featuretracking_t featuretracking_;
    // add 后端的配置参数
    backEndParams_t backEndParams_;
    // add IMU params
    imuParams_t imuParams_;
    // camera model
    CameraModel::Ptr camera_;
};

} // namespace vloam
