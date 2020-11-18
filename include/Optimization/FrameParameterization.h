//
// Created by ubuntu on 2020/5/10.
//

#ifndef FRAMEPARAMETERIZATION_H
#define FRAMEPARAMETERIZATION_H

#include "ceres/local_parameterization.h"
#include "sophus/se3.hpp"

#include <memory>


namespace  vloam
{
class FrameParameterization: public ceres::LocalParameterization
{
public:
    FrameParameterization();
    virtual ~FrameParameterization();

    virtual bool Plus(const double* x, const double* delta,
        double* x_plus_delta)const;

    virtual bool ComputeJacobian(const double* x, double* jacobian) const;

    virtual int GlobalSize() const;
    virtual int LocalSize() const;

};
} // namespce vloam

#endif //FRAMEPARAMETERIZATION_H
