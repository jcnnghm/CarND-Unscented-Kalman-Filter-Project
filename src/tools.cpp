#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  for (int i=0; i<estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    rmse = rmse.array() + (residual.array() * residual.array());
  }

  // Take the mean
  rmse = rmse / estimations.size();
  // Then the square root
  rmse = rmse.array().sqrt();
  return rmse;
}
