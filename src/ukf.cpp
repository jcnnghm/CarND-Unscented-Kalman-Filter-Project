#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.fill(0);
  P_(0,0) = 1;
  P_(1,1) = 1;
  P_(2,2) = 10;
  P_(3,3) = 10;
  P_(4,4) = 10;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.55;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.475;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.125;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.125;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(1.0 / (2.0 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Set not initialized
  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Process measurements, predicting and updating for each measurement.
  */

  if (!is_initialized_) {

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float range = meas_package.raw_measurements_[0];
      float bearing = meas_package.raw_measurements_[1];
      x_ << cos(bearing) * range, sin(bearing) * range, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;

    // Finished initializing, skip predict and update on the first measurement
    is_initialized_ = true;
    return;
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) return;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) return;

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Predict
  Prediction(delta_t);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // Generate Sigma Points
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig.col(0) = x_;
  auto llt = P_.llt();
  MatrixXd A = llt.matrixL();
  for (int i=0; i<n_x_; i++) {
    Xsig.col(i+1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i+n_x_+1) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }


  // Augment Sigma Points
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.fill(0);
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  auto aug_llt = P_aug.llt();
  MatrixXd square = aug_llt.matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for(int i=0; i<n_aug_;i++) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * square.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * square.col(i);
  }

  // predict sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  for(int i=0; i<2*n_aug_+1; i++) {
      double px = Xsig_aug(0, i);
      double py = Xsig_aug(1, i);
      double v = Xsig_aug(2, i);
      double psi = Xsig_aug(3, i);
      double psi_dot = Xsig_aug(4, i);
      double nu_a = Xsig_aug(5, i);
      double nu_psi = Xsig_aug(6, i);

      if (psi_dot != 0) {
        Xsig_pred_(0, i) = px + (v / psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi)) +
            0.5 * (delta_t * delta_t) * cos(psi) * nu_a;
        Xsig_pred_(1, i) = py + (v / psi_dot) * (-cos(psi + psi_dot * delta_t) + cos(psi)) +
            0.5 * (delta_t * delta_t) * sin(psi) * nu_a;
      } else {
        Xsig_pred_(0, i) = px + v * cos(psi) * delta_t +
            0.5 * (delta_t * delta_t) * cos(psi) * nu_a;
        Xsig_pred_(1, i) = py + v * sin(psi) * delta_t +
            0.5 * (delta_t * delta_t) * sin(psi) * nu_a;
      }


      Xsig_pred_(2, i) = v + delta_t * nu_a;
      Xsig_pred_(3, i) = psi + (psi_dot * delta_t) +
        0.5 * delta_t * delta_t * nu_psi;
      Xsig_pred_(4, i) = psi_dot + delta_t * nu_psi;
  }

  //predict state mean
  //x = Xsig_pred * weights;
  x_.fill(0);
  for (int i=0; i<2*n_aug_+1; i++) {
      x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.fill(0);
  for (int i=0; i<2*n_aug_+1; i++) {
    MatrixXd diff = Xsig_pred_.col(i) - x_;
    while (diff(3) > M_PI) diff(3) -= 2.0 * M_PI;
    while (diff(3) < -M_PI) diff(3) += 2.0 * M_PI;

    P_ += weights_(i) * diff * diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);

  //transform sigma points into measurement space
  for (int i=0; i<2*n_aug_+1; i++){
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      Zsig(0, i) = px;
      Zsig(1, i) = py;
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(2, 2);
  R.fill(0);
  R(0,0) = std_laspx_ * std_laspx_;
  R(1,1) = std_laspy_ * std_laspy_;

  for(int i=0; i<2*n_aug_+1; i++) {
      MatrixXd diff = Zsig.col(i) - z_pred;
      S += weights_(i) * diff * diff.transpose();
  }
  S += R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  VectorXd z = meas_package.raw_measurements_;

  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.0 * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  while (z_diff(1)> M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.0 * M_PI;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //create matrix for sigma points in measurement space
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);

  //transform sigma points into measurement space
  for (int i=0; i<2*n_aug_+1; i++){
      double px = Xsig_pred_(0, i);
      double py = Xsig_pred_(1, i);
      double v = Xsig_pred_(2, i);
      double psi = Xsig_pred_(3, i);

      if (px == 0 && py == 0) {
        Zsig(0, i) = 0;
        Zsig(1, i) = 0;
        Zsig(2, i) = 0;
      } else {
        Zsig(0, i) = sqrt(px * px + py * py);
        Zsig(1, i) = atan(py / px);
        Zsig(2, i) = (px * cos(psi) * v + py * sin(psi) * v) / sqrt(px * px + py * py);
      }
  }

  //calculate mean predicted measurement
  z_pred = Zsig * weights_;

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(3, 3);
  R.fill(0);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;

  for(int i=0; i<2*n_aug_+1; i++) {
      MatrixXd diff = Zsig.col(i) - z_pred;
      while (diff(1) > M_PI) diff(1) -= 2.0 * M_PI;
      while (diff(1) < -M_PI) diff(1) += 2.0 * M_PI;

      S += weights_(i) * diff * diff.transpose();
  }
  S += R;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  VectorXd z = meas_package.raw_measurements_;

  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.0 * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.0 * M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.0 * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.0 * M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.0 * M_PI;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
