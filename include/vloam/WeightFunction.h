#pragma once
#include <vector>
#include <memory>
#include <cmath>

#include <opencv2/opencv.hpp>

namespace vloam
{

enum class ScaleEstimatorType
{
    None,
    TDistributionScale
};
enum class WeightFunctionType
{
    None,
    TDistributionWeight,
    HuberWeight
};

class ScaleEstimator
{
public:
    typedef std::shared_ptr<ScaleEstimator> Ptr;

    ScaleEstimator()
    {}
    virtual ~ScaleEstimator()
    {}
    virtual float compute(std::vector<float> &errors) = 0;
    virtual float compute(cv::Mat &errors) = 0;
};

class TDistributionScaleEstimator: public ScaleEstimator
{
public:
    typedef std::shared_ptr<TDistributionScaleEstimator> Ptr;
    TDistributionScaleEstimator(const float dof = DEFAULT_DOF);
    virtual ~TDistributionScaleEstimator()
    {}
    virtual float compute(std::vector<float> &errors);
    virtual float compute(cv::Mat &errors);

    static const float DEFAULT_DOF;
    static const float INITIAL_SIGMA;

protected:
    float dof_;
    float initial_sigma_;
};

class WeightFunction
{
public:
    typedef std::shared_ptr<WeightFunction> Ptr;
    virtual ~WeightFunction()
    {}
    virtual void parameters(const float &param) = 0;
    virtual float weight(const float &x) = 0;
};

class TDistributionWeightFunction: public WeightFunction
{
public:
    typedef std::shared_ptr<TDistributionWeightFunction> Ptr;
    TDistributionWeightFunction(const float dof = DEFAULT_DOF);
    virtual ~TDistributionWeightFunction()
    {}
    virtual void parameters(const float &param);
    virtual float weight(const float &res);

    static const float DEFAULT_DOF;

private:
    float dof_{};
    float normalizer_{};
};

class HuberWeightFunction: public WeightFunction
{
public:
    typedef std::shared_ptr<HuberWeightFunction> Ptr;
    HuberWeightFunction(const float k = DEFAULT_K);
    virtual ~HuberWeightFunction()
    {}
    virtual void parameters(const float &param);
    virtual float weight(const float &res);

    static const float DEFAULT_K;

private:
    float k;
};

} // namespace vloam
