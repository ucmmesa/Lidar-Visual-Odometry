//
// Created by ubuntu on 2020/3/9.
//

#include <vloam/WeightFunction.h>

namespace vloam {

const float TDistributionScaleEstimator::INITIAL_SIGMA = 5.0;
const float TDistributionScaleEstimator::DEFAULT_DOF = 5.0;

const float TDistributionWeightFunction::DEFAULT_DOF = 5.0f;

TDistributionScaleEstimator::TDistributionScaleEstimator(const float dof)
    : dof_(dof), initial_sigma_(INITIAL_SIGMA)
{

}

float TDistributionScaleEstimator::compute(cv::Mat& errors)
{
    float initial_lamda = 1.0f / (initial_sigma_ * initial_sigma_);
    int num = 0;
    float lambda = initial_lamda;
    int iterations = 0;
    do
    {
        ++iterations;
        initial_lamda = lambda;
        num = 0.0f;
        lambda = 0.0f;

        const float* data_ptr = errors.ptr<float>();

        for(size_t idx = 0; idx < errors.size().area(); ++idx, ++data_ptr)
        {
            const float& data = *data_ptr;

            if(std::isfinite(data))
            {
                ++num;
                lambda += data * data * ( (dof_ + 1.0f) / (dof_ + initial_lamda * data * data) );
            }
        }

        lambda = float(num) / lambda;
    } while(std::abs(lambda - initial_lamda) > 1e-3);

    return std::sqrt(1.0f / lambda);
}

float TDistributionScaleEstimator::compute(std::vector<float>& errors)
{
    float initial_lamda = 1.0f / (initial_sigma_ * initial_sigma_);
    int num = 0;
    float lambda = initial_lamda;
    int iterations = 0;
    do
    {
        ++iterations;
        initial_lamda = lambda;
        num = 0;
        lambda = 0.0f;

        for(auto it=errors.begin(); it!=errors.end(); ++it)
        {
            if(std::isfinite(*it))
            {
                ++num;
                const float error2 = (*it)*(*it);
                lambda += error2 * ( (dof_ + 1.0f) / (dof_ + initial_lamda * error2) );
            }
        }
        lambda = float(num) / lambda;
    } while(std::abs(lambda - initial_lamda) > 1e-3);

    return std::sqrt(1.0f / lambda);
}

TDistributionWeightFunction::TDistributionWeightFunction(const float dof)
{
    parameters(dof);
}

void TDistributionWeightFunction::parameters(const float &param)
{
    dof_ = param;
    normalizer_ = dof_ / (dof_ + 1.0);
}

float TDistributionWeightFunction::weight(const float &res)
{
    // return std::max ( (dof_ + 1.0) / (dof_ + res*res), 0.001 );
    return (dof_ + 1.0) / (dof_ + res*res);
}

const float HuberWeightFunction::DEFAULT_K = 1.345f;

HuberWeightFunction::HuberWeightFunction(const float k)
{
    parameters(k);
}

void HuberWeightFunction::parameters(const float& param)
{
    k = param;
}

float HuberWeightFunction::weight(const float& res)
{
    float sig = 5.000/255.00;
    const float t_abs = std::abs(res);
    if(t_abs < k*sig)
        return 1.0f;
    else
        return k / t_abs;
}

} // namespace vloam
