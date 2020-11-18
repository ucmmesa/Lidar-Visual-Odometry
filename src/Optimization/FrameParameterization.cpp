//
// Created by ubuntu on 2020/5/10.
//

#include "Optimization/FrameParameterization.h"


namespace vloam
{
// Parameterization

FrameParameterization::FrameParameterization()
{

}

FrameParameterization::~FrameParameterization()
{

}

bool FrameParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    //map buffers
    Eigen::Map<Sophus::SE3d const> T(x);
    Eigen::Map<Sophus::Vector6d const> const d(delta);
    Eigen::Map<Sophus::SE3d> T_plus_delta(x_plus_delta);

//    LOG(INFO) << "increment d: " << d.transpose();
    T_plus_delta = Sophus::SE3d::exp(d) * T;

    return true;
}

bool FrameParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    // trick to work directly in the tangent space
    // compute jacobians relativate to the tangent space
    // and let this jacobian be the identity
    // J = [I(6x6); 0]

    Eigen::Map<Eigen::Matrix<double, 7,6, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<6,6>(0,0).setIdentity();
    return true;
}

int FrameParameterization::GlobalSize() const
{
    return (Sophus::SE3d::num_parameters);
}

int FrameParameterization::LocalSize() const
{
    return (Sophus::SE3d::DoF);
}

} // namespace vloam