//
// Created by ubuntu on 2020/10/27.
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

int main(int argc, char **argv) {

  Eigen::Matrix3d R;
  R <<  0.00875116, -0.00479609, 0.99995014,
      -0.99986423, -0.01400249, 0.00868325,
      0.01396015,-0.99989048, -0.00491798;

  std::cout << "R:\n" << R << std::endl;

  Eigen::Quaterniond q(R);

  std::cout << "q[xyzw]: " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();

  return 0;
}