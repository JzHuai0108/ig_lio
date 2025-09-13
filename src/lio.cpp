#include "ig_lio/lio.h"

#include "ig_lio/common.h"
#include "ig_lio/timer.h"

extern Timer timer;

bool LIO::MeasurementUpdate(SensorMeasurement& sensor_measurement) {
  if (sensor_measurement.measurement_type_ == MeasurementType::LIDAR) {
    // range filter
    CloudPtr filtered_cloud_ptr(new CloudType());
    filtered_cloud_ptr->points.reserve(sensor_measurement.cloud_ptr_->size());
    for (const auto& pt : sensor_measurement.cloud_ptr_->points) {
      if (InRadius(pt)) {
        filtered_cloud_ptr->points.emplace_back(pt);
      }
    }
    sensor_measurement.cloud_ptr_ = filtered_cloud_ptr;
    sensor_measurement.lidar_cloud_ptr_ = pcl::make_shared<CloudType>(*sensor_measurement.cloud_ptr_);

    timer.Evaluate(
        [&, this]() {
          // transform scan from lidar's frame to imu's frame
          CloudPtr cloud_body_ptr(new CloudType());
          pcl::transformPointCloud(*sensor_measurement.cloud_ptr_,
                                   *cloud_body_ptr,
                                   config_.T_imu_lidar);
          sensor_measurement.cloud_ptr_ = std::move(cloud_body_ptr);

          // undistort
          if (config_.enable_undistort) {
            UndistortPointCloud(sensor_measurement.bag_time_,
                                sensor_measurement.lidar_end_time_,
                                sensor_measurement.cloud_ptr_);
          }
        },
        "undistort");

    timer.Evaluate(
        [&, this]() {
          fast_voxel_grid_ptr_->Filter(
              sensor_measurement.cloud_ptr_, cloud_DS_ptr_, cloud_cov_ptr_);
        },
        "downsample");
  }

  // Make sure the local map is dense enought to measurement update
  if (lidar_frame_count_ <= 10) {
    CloudPtr trans_cloud_ptr(new CloudType());
    pcl::transformPointCloud(
        *sensor_measurement.cloud_ptr_, *trans_cloud_ptr, curr_state_.pose);
    voxel_map_ptr_->AddCloud(trans_cloud_ptr);
    lidar_frame_count_++;
    return true;
  }

  // measurement update
  prev_state_ = curr_state_;
  iter_num_ = 0;
  need_converge_ = false;
  Eigen::Matrix<double, 15, 1> delta_x = Eigen::Matrix<double, 15, 1>::Zero();
  while (iter_num_ < config_.max_iterations) {
    StepOptimize(sensor_measurement, delta_x);

    if (IsConverged(delta_x)) {
      // Optimization convergence, exit
      break;
    } else {
      // The first three iterations perform KNN, then no longer perform, thus
      // accelerating the problem convergence
      if (iter_num_ < 3) {
        need_converge_ = false;
      } else {
        need_converge_ = true;
      }
    }

    iter_num_++;
  }

  // LOG(INFO) << "final hessian: " << std::endl << final_hessian_;
  // P_ = final_hessian_.inverse();
  ComputeFinalCovariance(delta_x);
  prev_state_ = curr_state_;

  timer.Evaluate(
      [&, this]() {
        if (lidar_frame_count_ < 10) {
          CloudPtr trans_cloud_ptr(new CloudType());
          pcl::transformPointCloud(*sensor_measurement.cloud_ptr_,
                                   *trans_cloud_ptr,
                                   curr_state_.pose);
          voxel_map_ptr_->AddCloud(trans_cloud_ptr);

          last_keyframe_pose_ = curr_state_.pose;
        } else {
          Eigen::Matrix4d delta_p = SE3Inverse(last_keyframe_pose_) * curr_state_.pose;
          // The keyframe strategy ensures an appropriate spatial pattern of the
          // points in each voxel
          if (effect_feat_num_ < 1000 ||
              delta_p.block<3, 1>(0, 3).norm() > 0.5 ||
              Sophus::SO3d(delta_p.block<3, 3>(0, 0)).log().norm() > 0.18) {
            CloudPtr trans_cloud_DS_ptr(new CloudType());
            pcl::transformPointCloud(
                *cloud_DS_ptr_, *trans_cloud_DS_ptr, curr_state_.pose);
            voxel_map_ptr_->AddCloud(trans_cloud_DS_ptr);

            last_keyframe_pose_ = curr_state_.pose;
            keyframe_count_++;
          }
        }
      },
      "update voxel map");

  lidar_frame_count_++;

  ava_effect_feat_num_ += (effect_feat_num_ - ava_effect_feat_num_) /
                          static_cast<double>(lidar_frame_count_);
  VLOG(2) << "curr_feat_num: " << effect_feat_num_
          << " ava_feat_num: " << ava_effect_feat_num_
          << " keyframe_count: " << keyframe_count_
          << " lidar_frame_count: " << lidar_frame_count_
          << " grid_size: " << voxel_map_ptr_->GetVoxelMapSize();
  return true;
}

bool LIO::StepOptimize(const SensorMeasurement& sensor_measurement,
                       Eigen::Matrix<double, 15, 1>& delta_x) {
  Eigen::Matrix<double, 15, 15> H = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 1> b = Eigen::Matrix<double, 15, 1>::Zero();

  double y0 = 0;
  switch (sensor_measurement.measurement_type_) {
  case MeasurementType::LIDAR: {
    double y0_lidar = 0.0;

    timer.Evaluate(
        [&, this]() {
          // After LIO has moved some distance, each voxel is already well
          // formulate
          // the surrounding environments
          if (keyframe_count_ > 20) {
            y0_lidar = ConstructGICPConstraints(H, b);
          }
          // In the initial state, the probability of each voxel is poor
          // use point-to-plane instead of GICP
          else {
            y0_lidar = ConstructPoint2PlaneConstraints(H, b);
          }
        },
        "lidar constraints");

    y0 += y0_lidar;
    break;
  }

  default: {
    LOG(ERROR) << "error measurement type!";
    exit(0);
  }
  }

  // LOG(INFO) << "lidar H: " << std::endl << H << std::endl;

  timer.Evaluate(
      [&, this]() {
        double y0_imu = ConstructImuPriorConstraints(H, b);
        y0 += y0_imu;
      },
      "imu constraint");

  GNStep(sensor_measurement, H, b, y0, delta_x);

  return true;
}

bool LIO::GNStep(const SensorMeasurement& sensor_measurement,
                 Eigen::Matrix<double, 15, 15>& H,
                 Eigen::Matrix<double, 15, 1>& b,
                 const double y0,
                 Eigen::Matrix<double, 15, 1>& delta_x) {
  timer.Evaluate(
      [&, this]() {
        // The function inverse() has better numerical stability
        // And the dimension is small, direct inversion is not time-consuming
        Eigen::Matrix<double, 15, 1> dir = -H.inverse() * b;

        State new_state;
        delta_x = dir;
        CorrectState(curr_state_, delta_x, new_state);
        curr_state_ = new_state;

        final_hessian_ = H;
      },
      "gn step");

  return true;
}

double LIO::ConstructGICPConstraints(Eigen::Matrix<double, 15, 15>& H,
                                     Eigen::Matrix<double, 15, 1>& b) {
  Eigen::Matrix<double, 8, 6> result_matrix =
      Eigen::Matrix<double, 8, 6>::Zero();
  Eigen::Matrix<double, 8, 6> init_matrix = Eigen::Matrix<double, 8, 6>::Zero();

  if (need_converge_) {
    result_matrix = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, correspondences_array_.size()),
        init_matrix,
        [&, this](tbb::blocked_range<size_t> r,
                  Eigen::Matrix<double, 8, 6> local_result) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            Eigen::Vector3d trans_mean_A =
                curr_state_.pose.block<3, 3>(0, 0) *
                    correspondences_array_[i]->mean_A +
                curr_state_.pose.block<3, 1>(0, 3);

            Eigen::Vector3d error =
                correspondences_array_[i]->mean_B - trans_mean_A;

            // without loss function
            // local_result(7, 0) += gicp_constraint_gain_ * error.transpose() *
            //                       correspondences_array_[i]->mahalanobis *
            //                       error;

            // // The residual takes the partial derivative of the state
            // Eigen::Matrix<double, 3, 6> dres_dx =
            //     Eigen::Matrix<double, 3, 6>::Zero();

            // // The residual takes the partial derivative of rotation
            // dres_dx.block<3, 3>(0, 0) =
            //     curr_state_.pose.block<3, 3>(0, 0) *
            //     Sophus::SO3d::hat(correspondences_array_[i]->mean_A);

            // // The residual takes the partial derivative of position
            // dres_dx.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

            // local_result.block(0, 0, 6, 6) +=
            //     gicp_constraint_gain_ * dres_dx.transpose() *
            //     correspondences_array_[i]->mahalanobis * dres_dx;

            // local_result.block(6, 0, 1, 6) +=
            //     (gicp_constraint_gain_ * dres_dx.transpose() *
            //      correspondences_array_[i]->mahalanobis * error)
            //         .transpose();

            // loss function
            Eigen::Matrix3d mahalanobis =
                correspondences_array_[i]->mahalanobis;
            double cost_function = error.transpose() * mahalanobis * error;
            Eigen::Vector3d rho;
            CauchyLossFunction(cost_function, 10.0, rho);

            local_result(7, 0) += config_.gicp_constraint_gain * rho[0];

            // The residual takes the partial derivative of the state
            Eigen::Matrix<double, 3, 6> dres_dx =
                Eigen::Matrix<double, 3, 6>::Zero();

            // The residual takes the partial derivative of rotation
            dres_dx.block<3, 3>(0, 0) =
                curr_state_.pose.block<3, 3>(0, 0) *
                Sophus::SO3d::hat(correspondences_array_[i]->mean_A);

            // The residual takes the partial derivative of position
            dres_dx.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

            Eigen::Matrix3d robust_information_matrix =
                config_.gicp_constraint_gain *
                (rho[1] * mahalanobis + 2.0 * rho[2] * mahalanobis * error *
                                            error.transpose() * mahalanobis);
            local_result.block(0, 0, 6, 6) +=
                dres_dx.transpose() * robust_information_matrix * dres_dx;

            local_result.block(6, 0, 1, 6) +=
                (config_.gicp_constraint_gain * rho[1] * dres_dx.transpose() *
                 mahalanobis * error)
                    .transpose();
          }

          return local_result;
        },
        [](Eigen::Matrix<double, 8, 6> x, Eigen::Matrix<double, 8, 6> y) {
          return x + y;
        });

    H.block<6, 6>(IndexErrorOri, IndexErrorOri) +=
        result_matrix.block<6, 6>(0, 0);
    b.block<6, 1>(IndexErrorOri, 0) +=
        result_matrix.block<1, 6>(6, 0).transpose();

    return result_matrix(7, 0);
  }

  size_t delta_p_size = voxel_map_ptr_->delta_P_.size();
  size_t N = cloud_cov_ptr_->size();
  correspondences_array_.clear();
  result_matrix = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, N),
      init_matrix,
      [&, this](tbb::blocked_range<size_t> r,
                Eigen::Matrix<double, 8, 6> local_result) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          const PointCovType& point_cov = cloud_cov_ptr_->points[i];
          const Eigen::Vector3d mean_A =
              point_cov.getVector3fMap().cast<double>();
          const Eigen::Vector3d trans_mean_A =
              curr_state_.pose.block<3, 3>(0, 0) * mean_A +
              curr_state_.pose.block<3, 1>(0, 3);

          Eigen::Matrix3d cov_A;
          cov_A << point_cov.cov[0], point_cov.cov[1], point_cov.cov[2],
              point_cov.cov[1], point_cov.cov[3], point_cov.cov[4],
              point_cov.cov[2], point_cov.cov[4], point_cov.cov[5];

          Eigen::Vector3d mean_B = Eigen::Vector3d::Zero();
          Eigen::Matrix3d cov_B = Eigen::Matrix3d::Zero();

          for (size_t i = 0; i < delta_p_size; ++i) {
            Eigen::Vector3d nearby_point =
                trans_mean_A + voxel_map_ptr_->delta_P_[i];
            size_t hash_idx = voxel_map_ptr_->ComputeHashIndex(nearby_point);
            if (voxel_map_ptr_->GetCentroidAndCovariance(
                    hash_idx, mean_B, cov_B) &&
                voxel_map_ptr_->IsSameGrid(nearby_point, mean_B)) {
              Eigen::Matrix3d mahalanobis =
                  (cov_B +
                   curr_state_.pose.block<3, 3>(0, 0) * cov_A *
                       curr_state_.pose.block<3, 3>(0, 0).transpose() +
                   Eigen::Matrix3d::Identity() * 1e-3)
                      .inverse();

              Eigen::Vector3d error = mean_B - trans_mean_A;
              double chi2_error = error.transpose() * mahalanobis * error;
              if (config_.enable_outlier_rejection) {
                if (iter_num_ > 2 && chi2_error > 7.815) {
                  continue;
                }
              }

              std::shared_ptr<Correspondence> corr_ptr =
                  std::make_shared<Correspondence>();
              corr_ptr->mean_A = mean_A;
              corr_ptr->mean_B = mean_B;
              corr_ptr->mahalanobis = mahalanobis;
              correspondences_array_.emplace_back(corr_ptr);

              // without loss function
              // local_result(7, 0) += gicp_constraint_gain_ * chi2_error;

              // // The residual takes the partial derivative of the state
              // Eigen::Matrix<double, 3, 6> dres_dx =
              //     Eigen::Matrix<double, 3, 6>::Zero();

              // // The residual takes the partial derivative of rotation
              // dres_dx.block<3, 3>(0, 0) = curr_state_.pose.block<3, 3>(0, 0)
              // *
              //                             Sophus::SO3d::hat(mean_A);

              // // The residual takes the partial derivative of position
              // dres_dx.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

              // local_result.block(0, 0, 6, 6) += gicp_constraint_gain_ *
              //                                   dres_dx.transpose() *
              //                                   mahalanobis * dres_dx;

              // local_result.block(6, 0, 1, 6) +=
              //     (gicp_constraint_gain_ * dres_dx.transpose() * mahalanobis
              //     *
              //      error)
              //         .transpose();

              // loss function
              double cost_function = chi2_error;
              Eigen::Vector3d rho;
              CauchyLossFunction(cost_function, 10.0, rho);

              local_result(7, 0) += config_.gicp_constraint_gain * rho[0];

              // The residual takes the partial derivative of the state
              Eigen::Matrix<double, 3, 6> dres_dx =
                  Eigen::Matrix<double, 3, 6>::Zero();

              // The residual takes the partial derivative of rotation
              dres_dx.block<3, 3>(0, 0) = curr_state_.pose.block<3, 3>(0, 0) *
                                          Sophus::SO3d::hat(mean_A);

              // The residual takes the partial derivative of position
              dres_dx.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

              Eigen::Matrix3d robust_information_matrix =
                  config_.gicp_constraint_gain *
                  (rho[1] * mahalanobis + 2.0 * rho[2] * mahalanobis * error *
                                              error.transpose() * mahalanobis);
              local_result.block(0, 0, 6, 6) +=
                  dres_dx.transpose() * robust_information_matrix * dres_dx;

              local_result.block(6, 0, 1, 6) +=
                  (config_.gicp_constraint_gain * rho[1] * dres_dx.transpose() *
                   mahalanobis * error)
                      .transpose();

              break;
            }
          }
        }

        return local_result;
      },
      [](Eigen::Matrix<double, 8, 6> x, Eigen::Matrix<double, 8, 6> y) {
        return x + y;
      });

  effect_feat_num_ = correspondences_array_.size();

  H.block<6, 6>(IndexErrorOri, IndexErrorOri) +=
      result_matrix.block<6, 6>(0, 0);
  b.block<6, 1>(IndexErrorOri, 0) +=
      result_matrix.block<1, 6>(6, 0).transpose();

  return result_matrix(7, 0);
}

double LIO::ConstructPoint2PlaneConstraints(Eigen::Matrix<double, 15, 15>& H,
                                            Eigen::Matrix<double, 15, 1>& b) {
  Eigen::Matrix<double, 8, 6> result_matrix =
      Eigen::Matrix<double, 8, 6>::Zero();
  Eigen::Matrix<double, 8, 6> init_matrix = Eigen::Matrix<double, 8, 6>::Zero();

  // Skip the KNN to accelerate convergence
  if (need_converge_) {
    result_matrix = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, correspondences_array_.size()),
        init_matrix,
        [&, this](tbb::blocked_range<size_t> r,
                  Eigen::Matrix<double, 8, 6> local_result) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            const Eigen::Vector3d trans_pt =
                curr_state_.pose.block<3, 3>(0, 0) *
                    correspondences_array_[i]->mean_A +
                curr_state_.pose.block<3, 1>(0, 3);
            const Eigen::Vector4d& plane_coeff =
                correspondences_array_[i]->plane_coeff;

            double error =
                plane_coeff.head(3).dot(trans_pt) + plane_coeff(3, 0);

            local_result(7, 0) +=
                config_.point2plane_constraint_gain * error * error;

            // The residual takes the partial derivative of the state
            Eigen::Matrix<double, 1, 6> dres_dx =
                Eigen::Matrix<double, 1, 6>::Zero();

            // The residual takes the partial derivative of rotation
            dres_dx.block<1, 3>(0, 0) =
                -plane_coeff.head(3).transpose() *
                curr_state_.pose.block<3, 3>(0, 0) *
                Sophus::SO3d::hat(correspondences_array_[i]->mean_A);

            // The residual takes the partial derivative of position
            dres_dx.block<1, 3>(0, 3) = plane_coeff.head(3).transpose();

            local_result.block(0, 0, 6, 6) +=
                config_.point2plane_constraint_gain * dres_dx.transpose() *
                dres_dx;

            local_result.block(6, 0, 1, 6) +=
                config_.point2plane_constraint_gain * dres_dx * error;
          }

          return local_result;
        },
        [](Eigen::Matrix<double, 8, 6> x, Eigen::Matrix<double, 8, 6> y) {
          return x + y;
        });

    H.block<6, 6>(IndexErrorOri, IndexErrorOri) +=
        result_matrix.block<6, 6>(0, 0);
    b.block<6, 1>(IndexErrorOri, 0) +=
        result_matrix.block<1, 6>(6, 0).transpose();

    return result_matrix(7, 0);
  }

  size_t N = cloud_cov_ptr_->size();
  correspondences_array_.clear();
  result_matrix = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, N),
      init_matrix,
      [&, this](tbb::blocked_range<size_t> r,
                Eigen::Matrix<double, 8, 6> local_result) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          const Eigen::Vector3d p =
              cloud_cov_ptr_->points[i].getVector3fMap().cast<double>();
          const Eigen::Vector3d p_w = curr_state_.pose.block<3, 3>(0, 0) * p +
                                      curr_state_.pose.block<3, 1>(0, 3);

          std::vector<Eigen::Vector3d> nearest_points;
          nearest_points.reserve(10);

          voxel_map_ptr_->KNNByCondition(p_w, 5, 5.0, nearest_points);

          Eigen::Vector4d plane_coeff;
          if (nearest_points.size() >= 3 &&
              EstimatePlane(plane_coeff, nearest_points)) {
            double error = plane_coeff.head(3).dot(p_w) + plane_coeff(3, 0);

            bool is_vaild = p.norm() > (81 * error * error);
            if (is_vaild) {
              std::shared_ptr<Correspondence> corr_ptr =
                  std::make_shared<Correspondence>();
              corr_ptr->mean_A = p;
              corr_ptr->plane_coeff = plane_coeff;
              correspondences_array_.emplace_back(corr_ptr);

              local_result(7, 0) +=
                  config_.point2plane_constraint_gain * error * error;

              // The residual takes the partial derivative of the state
              Eigen::Matrix<double, 1, 6> dres_dx =
                  Eigen::Matrix<double, 1, 6>::Zero();

              // The residual takes the partial derivative of rotation
              dres_dx.block<1, 3>(0, 0) = -plane_coeff.head(3).transpose() *
                                          curr_state_.pose.block<3, 3>(0, 0) *
                                          Sophus::SO3d::hat(p);

              // The residual takes the partial derivative of position
              dres_dx.block<1, 3>(0, 3) = plane_coeff.head(3).transpose();

              local_result.block(0, 0, 6, 6) +=
                  config_.point2plane_constraint_gain * dres_dx.transpose() *
                  dres_dx;

              local_result.block(6, 0, 1, 6) +=
                  config_.point2plane_constraint_gain * dres_dx * error;
            }
          }
        }

        return local_result;
      },
      [](Eigen::Matrix<double, 8, 6> x, Eigen::Matrix<double, 8, 6> y) {
        return x + y;
      });

  effect_feat_num_ = correspondences_array_.size();

  H.block<6, 6>(IndexErrorOri, IndexErrorOri) +=
      result_matrix.block<6, 6>(0, 0);
  b.block<6, 1>(IndexErrorOri, 0) +=
      result_matrix.block<1, 6>(6, 0).transpose();

  return result_matrix(7, 0);
}

double LIO::ConstructImuPriorConstraints(Eigen::Matrix<double, 15, 15>& H,
                                         Eigen::Matrix<double, 15, 1>& b) {
  Sophus::SO3d ori_diff =
      Sophus::SO3d(prev_state_.pose.block<3, 3>(0, 0).transpose() *
                   curr_state_.pose.block<3, 3>(0, 0));
  Eigen::Vector3d ori_error = ori_diff.log();

  Eigen::Matrix3d right_jacoiban_inv = Sophus::SO3d::jr_inv(ori_diff);

  Eigen::Matrix<double, 15, 15> jacobian =
      Eigen::Matrix<double, 15, 15>::Identity();
  jacobian.block<3, 3>(IndexErrorOri, IndexErrorOri) = right_jacoiban_inv;

  // LOG(INFO) << "imu jacobian: " << std::endl << jacobian;

  Eigen::Matrix<double, 15, 1> residual = Eigen::Matrix<double, 15, 1>::Zero();
  residual.block<3, 1>(IndexErrorOri, 0) = ori_error;
  residual.block<3, 1>(IndexErrorPos, 0) =
      curr_state_.pose.block<3, 1>(0, 3) - prev_state_.pose.block<3, 1>(0, 3);
  residual.block<3, 1>(IndexErrorVel, 0) = curr_state_.vel - prev_state_.vel;
  residual.block<3, 1>(IndexErrorBiasAcc, 0) = curr_state_.ba - prev_state_.ba;
  residual.block<3, 1>(IndexErrorBiasGyr, 0) = curr_state_.bg - prev_state_.bg;

  Eigen::Matrix<double, 15, 15> inv_P = P_.inverse();

  // LOG(INFO) << "inv_P: " << std::endl << inv_P;

  H += jacobian.transpose() * inv_P * jacobian;
  b += jacobian.transpose() * inv_P * residual;

  double errors = residual.transpose() * inv_P * residual;

  return errors;
}

bool LIO::Predict(const double time,
                  const Eigen::Vector3d& acc_1,
                  const Eigen::Vector3d& gyr_1) {
  double dt = time - lio_time_;

  Eigen::Vector3d un_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d un_gyr = Eigen::Vector3d::Zero();

  NominalStateUpdate(dt,
                     acc_0_,
                     acc_1,
                     gyr_0_,
                     gyr_1,
                     prev_state_.pose,
                     prev_state_.vel,
                     curr_state_.pose,
                     curr_state_.vel,
                     un_acc,
                     un_gyr);

  ErrorStateUpdate(dt, acc_0_, acc_1, gyr_0_, gyr_1);

  if (config_.enable_undistort) {
    // save the predicted pose for scan undistortion
    PoseHistory ph;
    ph.time_ = time;
    ph.T_ = curr_state_.pose;
    ph.un_acc_ = un_acc;
    ph.un_gyr_ = un_gyr;
    ph.vel_ = curr_state_.vel;
    pose_history_.push_back(ph);
  }

  // save the data for next median integral
  prev_state_ = curr_state_;
  acc_0_ = acc_1;
  gyr_0_ = gyr_1;
  lio_time_ = time;

  return true;
}

bool LIO::NominalStateUpdate(const double dt,
                             const Eigen::Vector3d& acc_0,
                             const Eigen::Vector3d& acc_1,
                             const Eigen::Vector3d& gyr_0,
                             const Eigen::Vector3d& gyr_1,
                             const Eigen::Matrix4d& T_prev,
                             const Eigen::Vector3d& vel_prev,
                             Eigen::Matrix4d& T_curr,
                             Eigen::Vector3d& vel_curr,
                             Eigen::Vector3d& un_acc,
                             Eigen::Vector3d& un_gyr) {
  // update ori
  un_gyr = 0.5 * (gyr_0 + gyr_1) - curr_state_.bg;
  T_curr.block<3, 3>(0, 0) =
      T_prev.block<3, 3>(0, 0) * Sophus::SO3d::exp(un_gyr * dt).matrix();

  Eigen::Vector3d un_acc_0 =
      T_prev.block<3, 3>(0, 0) * (acc_0 - curr_state_.ba);
  Eigen::Vector3d un_acc_1 =
      T_curr.block<3, 3>(0, 0) * (acc_1 - curr_state_.ba);
  un_acc = 0.5 * (un_acc_0 + un_acc_1) - g_;

  // update vel
  vel_curr = vel_prev + un_acc * dt;
  // update pos
  T_curr.block<3, 1>(0, 3) =
      T_prev.block<3, 1>(0, 3) + vel_prev * dt + 0.5 * dt * dt * un_acc;

  return true;
}

bool LIO::ErrorStateUpdate(const double dt,
                           const Eigen::Vector3d& acc_0,
                           const Eigen::Vector3d& acc_1,
                           const Eigen::Vector3d& gyr_0,
                           const Eigen::Vector3d& gyr_1) {
  Eigen::Vector3d w = 0.5 * (gyr_0 + gyr_1) - curr_state_.bg;
  Eigen::Vector3d a0 = acc_0 - curr_state_.ba;
  Eigen::Vector3d a1 = acc_1 - curr_state_.ba;

  // Eigen::Matrix3d w_x = Sophus::SO3d::hat(w).matrix();
  Eigen::Matrix3d a0_x = Sophus::SO3d::hat(a0).matrix();
  Eigen::Matrix3d a1_x = Sophus::SO3d::hat(a1).matrix();
  Eigen::Matrix3d I_w_x = Sophus::SO3d::exp(-w * dt).matrix();

  F_.setZero();
  // F_.block<3,3>(IndexErrorVel,IndexErrorOri) =
  //     -0.5 * dt * prev_state_.pose.block<3,3>(0,0) * a0_x
  //     -0.5 * dt * curr_state_.pose.block<3,3>(0,0) * a1_x *
  //     (Eigen::Matrix3d::Identity() - w_x * dt);
  F_.block<3, 3>(IndexErrorVel, IndexErrorOri) =
      -0.5 * dt * prev_state_.pose.block<3, 3>(0, 0) * a0_x -
      0.5 * dt * curr_state_.pose.block<3, 3>(0, 0) * a1_x *
          I_w_x;  // More accurate than above
  F_.block<3, 3>(IndexErrorVel, IndexErrorVel) = Eigen::Matrix3d::Identity();
  F_.block<3, 3>(IndexErrorVel, IndexErrorBiasAcc) =
      -0.5 *
      (prev_state_.pose.block<3, 3>(0, 0) +
       curr_state_.pose.block<3, 3>(0, 0)) *
      dt;
  F_.block<3, 3>(IndexErrorVel, IndexErrorBiasGyr) =
      0.5 * curr_state_.pose.block<3, 3>(0, 0) * a1_x * dt * dt;

  F_.block<3, 3>(IndexErrorPos, IndexErrorPos) = Eigen::Matrix3d::Identity();
  F_.block<3, 3>(IndexErrorPos, IndexErrorOri) =
      0.5 * dt * F_.block<3, 3>(IndexErrorVel, IndexErrorOri);
  F_.block<3, 3>(IndexErrorPos, IndexErrorVel) =
      Eigen::Matrix3d::Identity() * dt;
  F_.block<3, 3>(IndexErrorPos, IndexErrorBiasAcc) =
      0.5 * dt * F_.block<3, 3>(IndexErrorVel, IndexErrorBiasAcc);
  F_.block<3, 3>(IndexErrorPos, IndexErrorBiasGyr) =
      0.5 * dt * F_.block<3, 3>(IndexErrorVel, IndexErrorBiasGyr);

  // F_.block<3,3>(IndexErrorOri,IndexErrorOri) = Eigen::Matrix3d::Identity()
  // - w_x * dt;
  F_.block<3, 3>(IndexErrorOri, IndexErrorOri) =
      I_w_x;  // More accurate than above
  F_.block<3, 3>(IndexErrorOri, IndexErrorBiasGyr) =
      -Eigen::Matrix3d::Identity() * dt;

  F_.block<3, 3>(IndexErrorBiasAcc, IndexErrorBiasAcc) =
      Eigen::Matrix3d::Identity();
  F_.block<3, 3>(IndexErrorBiasGyr, IndexErrorBiasGyr) =
      Eigen::Matrix3d::Identity();

  B_.setZero();
  B_.block<3, 3>(IndexErrorVel, IndexNoiseAccLast) =
      0.5 * prev_state_.pose.block<3, 3>(0, 0) * dt;
  B_.block<3, 3>(IndexErrorVel, IndexNoiseGyrLast) =
      -0.25 * curr_state_.pose.block<3, 3>(0, 0) * a1_x * dt * dt;
  B_.block<3, 3>(IndexErrorVel, IndexNoiseAccCurr) =
      0.5 * curr_state_.pose.block<3, 3>(0, 0) * dt;
  B_.block<3, 3>(IndexErrorVel, IndexNoiseGyrCurr) =
      B_.block<3, 3>(IndexErrorVel, IndexNoiseGyrLast);

  B_.block<3, 3>(IndexErrorOri, IndexNoiseGyrLast) =
      0.5 * Eigen::Matrix3d::Identity() * dt;  // inaccuracy
  B_.block<3, 3>(IndexErrorOri, IndexNoiseGyrCurr) =
      B_.block<3, 3>(IndexErrorOri, IndexNoiseGyrLast);

  B_.block<3, 3>(IndexErrorPos, IndexNoiseAccLast) =
      0.5 * B_.block<3, 3>(IndexErrorVel, IndexNoiseAccLast) * dt;
  B_.block<3, 3>(IndexErrorPos, IndexNoiseGyrLast) =
      0.5 * B_.block<3, 3>(IndexErrorVel, IndexNoiseGyrLast) * dt;
  B_.block<3, 3>(IndexErrorPos, IndexNoiseAccCurr) =
      0.5 * B_.block<3, 3>(IndexErrorVel, IndexNoiseAccCurr) * dt;
  B_.block<3, 3>(IndexErrorPos, IndexNoiseGyrCurr) =
      B_.block<3, 3>(IndexErrorPos, IndexNoiseGyrLast);

  B_.block<3, 3>(IndexErrorBiasAcc, IndexNoiseBiasAcc) =
      Eigen::Matrix3d::Identity() * dt;
  B_.block<3, 3>(IndexErrorBiasGyr, IndexNoiseBiasGyr) =
      B_.block<3, 3>(IndexErrorBiasAcc, IndexNoiseBiasAcc);

  P_ = F_ * P_ * F_.transpose() + B_ * Q_ * B_.transpose();
  return true;
}

// Undistortion based on median integral
bool LIO::UndistortPointCloud(const double bag_time,
                              const double lidar_end_time,
                              CloudPtr& cloud_ptr) {
  Eigen::Matrix3d R_w_be = curr_state_.pose.block<3, 3>(0, 0);
  Eigen::Vector3d t_w_be = curr_state_.pose.block<3, 1>(0, 3);
  auto it_pt = cloud_ptr->points.end() - 1;
  bool finshed_flag = false;
  for (auto it_pose = pose_history_.end() - 1; it_pose != pose_history_.begin();
       --it_pose) {
    auto bi = it_pose - 1;
    auto bj = it_pose;
    Eigen::Matrix3d R_w_bi = bi->T_.block<3, 3>(0, 0);
    Eigen::Vector3d t_w_bi = bi->T_.block<3, 1>(0, 3);
    Eigen::Vector3d v_w_bi = bi->vel_;
    Eigen::Vector3d un_acc_bj = bj->un_acc_;
    Eigen::Vector3d un_gyr_bj = bj->un_gyr_;

    for (; (it_pt->curvature / (double)(1000) + bag_time) > bi->time_;
         --it_pt) {
      double dt = (it_pt->curvature / (double)(1000) + bag_time) - bi->time_;

      Eigen::Matrix3d R_w_bk =
          R_w_bi * Sophus::SO3d::exp(un_gyr_bj * dt).matrix();
      Eigen::Vector3d t_w_bebk =
          t_w_bi + v_w_bi * dt + 0.5 * dt * dt * un_acc_bj - t_w_be;
      // point_K
      Eigen::Vector3d P_bk_bkK(it_pt->x, it_pt->y, it_pt->z);
      Eigen::Vector3d P_w_beK =
          R_w_be.transpose() * (R_w_bk * P_bk_bkK + t_w_bebk);

      it_pt->x = P_w_beK.x();
      it_pt->y = P_w_beK.y();
      it_pt->z = P_w_beK.z();

      if (it_pt == cloud_ptr->points.begin()) {
        finshed_flag = true;
        break;
      }
    }

    if (finshed_flag) {
      break;
    }
  }

  // Remove excess history imu_pose
  while (!pose_history_.empty() &&
         (pose_history_.front().time_ < lidar_end_time)) {
    pose_history_.pop_front();
  }

  return true;
}

bool LIO::LinUndistortPointCloud(const double lidar_start_time,
                               const double lidar_end_time,
                               CloudPtr& cloud_ptr) {
  // Check time consistency of the penultimate and last lidar states
  double t_prev = lidar_states_[lidar_states_.size() - 2].time;
  double t_curr = lidar_states_.back().time;

  if (std::abs(t_prev - lidar_start_time) > 10e-3) {
    std::cerr << "[Warning] Penultimate lidar state time mismatch:\n"
              << "  Expected: " << std::fixed << std::setprecision(9) << lidar_start_time << "  Actual: " << t_prev
              << "  Diff: " << std::abs(t_prev - lidar_start_time) << " sec\n";
  }

  if (std::abs(t_curr - lidar_end_time) > 1e-4) {
    std::cerr << "[Warning] Last lidar state time mismatch:\n"
              << "  Expected: " << std::fixed << std::setprecision(9) << lidar_end_time << "  Actual: " << t_curr
              << "  Diff: " << std::abs(t_curr - lidar_end_time) << " sec\n";
  }

  // Get the last two lidar poses
  const auto& state_prev = lidar_states_[lidar_states_.size() - 2];
  const auto& state_curr = lidar_states_.back();

  // Compute angular and linear velocities between the last two poses
  Eigen::Matrix3d R_prev = state_prev.T.block<3, 3>(0, 0);
  Eigen::Matrix3d R_curr = state_curr.T.block<3, 3>(0, 0);
  Eigen::Matrix3d R_delta = R_prev.transpose() * R_curr;

  Eigen::AngleAxisd angle_axis(R_delta);
  double delta_t = t_curr - t_prev;
  Eigen::Vector3d angular_velocity = angle_axis.axis() * angle_axis.angle() / delta_t;
  Eigen::Vector3d linear_velocity = state_prev.vel;

  // Apply motion compensation to each point in the cloud
  for (auto& pt : cloud_ptr->points) {
    // Convert point timestamp relative to motion start
    double point_time = lidar_start_time + pt.curvature / 1000.0 - t_prev;

    // Pose of lidar at point time
    Eigen::Vector3d delta_angle = angular_velocity * point_time;
    Eigen::Matrix3d R_w_li = R_prev;
    if (delta_angle.norm() > 1e-10) {
        R_w_li *= Eigen::AngleAxisd(delta_angle.norm(), delta_angle.normalized()).toRotationMatrix();
    }
    Eigen::Vector3d t_w_li = state_prev.T.block<3, 1>(0, 3) + linear_velocity * point_time;

    // Pose of lidar at scan end time
    Eigen::Matrix3d R_w_le = R_curr;
    Eigen::Vector3d t_w_le = state_curr.T.block<3, 1>(0, 3);

    // Transform point from time-interpolated pose to scan-end pose
    Eigen::Vector3d p_li(pt.x, pt.y, pt.z);
    Eigen::Vector3d p_le = R_w_le.transpose() * (R_w_li * p_li + t_w_li - t_w_le);

    // Overwrite point with undistorted coordinates
    pt.x = p_le.x();
    pt.y = p_le.y();
    pt.z = p_le.z();
  }

  return true;
}

bool LIO::StaticInitialization(SensorMeasurement& sensor_measurement) {
  if (first_imu_frame_) {
    const auto& acc = sensor_measurement.imu_buff_.front().linear_acceleration;
    const auto& gyr = sensor_measurement.imu_buff_.front().angular_velocity;
    imu_init_buff_.emplace_back(Eigen::Vector3d(acc.x, acc.y, acc.z),
                                Eigen::Vector3d(gyr.x, gyr.y, gyr.z));
    first_imu_frame_ = false;
  }

  for (const auto& imu_msg : sensor_measurement.imu_buff_) {
    Eigen::Vector3d acc(imu_msg.linear_acceleration.x,
                        imu_msg.linear_acceleration.y,
                        imu_msg.linear_acceleration.z);
    Eigen::Vector3d gyr(imu_msg.angular_velocity.x,
                        imu_msg.angular_velocity.y,
                        imu_msg.angular_velocity.z);

    imu_init_buff_.emplace_back(acc, gyr);
  }

  if (imu_init_buff_.size() < max_init_count_) {
    return false;
  }

  Eigen::Vector3d acc_cov, gyr_cov;
  ComputeMeanAndCovDiag(
      imu_init_buff_,
      mean_acc_,
      acc_cov,
      [](const std::pair<Eigen::Vector3d, Eigen::Vector3d>& imu_data) {
        return imu_data.first;
      });
  ComputeMeanAndCovDiag(
      imu_init_buff_,
      mean_gyr_,
      gyr_cov,
      [](const std::pair<Eigen::Vector3d, Eigen::Vector3d>& imu_data) {
        return imu_data.second;
      });

  // Compute initial attitude via Schmidt orthogonalization.
  // The roll and pitch are aligned with the direction of gravity, but the yaw
  // is random.
  Eigen::Vector3d z_axis = mean_acc_.normalized();
  Eigen::Vector3d e1(1, 0, 0);
  Eigen::Vector3d x_axis = e1 - z_axis * z_axis.transpose() * e1;
  x_axis.normalize();
  Eigen::Vector3d y_axis = Sophus::SO3d::hat(z_axis).matrix() * x_axis;
  y_axis.normalize();

  Eigen::Matrix3d init_R;
  init_R.block<3, 1>(0, 0) = x_axis;
  init_R.block<3, 1>(0, 1) = y_axis;
  init_R.block<3, 1>(0, 2) = z_axis;
  Eigen::Quaterniond init_q(init_R);
  curr_state_.pose.block<3, 3>(0, 0) =
      init_q.normalized().toRotationMatrix().transpose();

  Eigen::Vector3d init_ba = Eigen::Vector3d::Zero();
  ComputeMeanAndCovDiag(
      imu_init_buff_,
      init_ba,
      acc_cov,
      [this](const std::pair<Eigen::Vector3d, Eigen::Vector3d>& imu_data) {
        Eigen::Vector3d temp_ba =
            imu_data.first -
            curr_state_.pose.block<3, 3>(0, 0).transpose() * g_;
        return temp_ba;
      });

  // init pose
  curr_state_.pose.block<3, 1>(0, 3).setZero();
  // init velocity
  curr_state_.vel.setZero();
  // init bg
  curr_state_.bg = mean_gyr_;
  // init ba
  curr_state_.ba = init_ba;

  prev_state_ = curr_state_;

  P_.setIdentity();
  P_.block<3, 3>(IndexErrorOri, IndexErrorOri) =
      config_.init_ori_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorPos, IndexErrorPos) =
      config_.init_pos_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorVel, IndexErrorVel) =
      config_.init_vel_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorBiasAcc, IndexErrorBiasAcc) =
      config_.init_ba_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorBiasGyr, IndexErrorBiasGyr) =
      config_.init_bg_cov * Eigen::Matrix3d::Identity();

  Q_.setIdentity();
  Q_.block<3, 3>(IndexNoiseAccLast, IndexNoiseAccLast) =
      config_.acc_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseGyrLast, IndexNoiseGyrLast) =
      config_.gyr_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseAccCurr, IndexNoiseAccCurr) =
      config_.acc_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseGyrCurr, IndexNoiseGyrCurr) =
      config_.gyr_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseBiasAcc, IndexNoiseBiasAcc) =
      config_.ba_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseBiasGyr, IndexNoiseBiasGyr) =
      config_.bg_cov * Eigen::Matrix3d::Identity();

  lio_time_ = sensor_measurement.imu_buff_.back().header.stamp.toSec();
  lio_init_ = true;

  LOG(INFO) << "imu static, mean_acc_: " << mean_acc_.transpose()
            << " init_ba: " << init_ba.transpose() << ", ori: " << std::endl
            << curr_state_.pose.block<3, 3>(0, 0);

  return true;
}

bool LIO::AHRSInitialization(SensorMeasurement& sensor_measurement) {
  const auto& back_imu = sensor_measurement.imu_buff_.back();
  double mag = (back_imu.orientation.w * back_imu.orientation.w +
       back_imu.orientation.x * back_imu.orientation.x +
       back_imu.orientation.y * back_imu.orientation.y +
       back_imu.orientation.z * back_imu.orientation.z);
  if (std::fabs(mag - 1.0) > 1e-2) {
    LOG(ERROR) << "AHRS initialization falid with an unusual magnitude "
       << std::fixed << std::setprecision(6) << mag << ", please use static initalizaiton!";
    return false;
  }

  Eigen::Quaterniond temp_q;
  if (first_imu_frame_) {
    const auto& acc = sensor_measurement.imu_buff_.front().linear_acceleration;
    const auto& gyr = sensor_measurement.imu_buff_.front().angular_velocity;
    temp_q = Eigen::Quaterniond(sensor_measurement.imu_buff_.front().orientation.w,
            sensor_measurement.imu_buff_.front().orientation.x,
            sensor_measurement.imu_buff_.front().orientation.y,
            sensor_measurement.imu_buff_.front().orientation.z);
    imu_init_buff_.emplace_back(Eigen::Vector3d(acc.x, acc.y, acc.z) - temp_q.toRotationMatrix().transpose() * g_,
                                Eigen::Vector3d(gyr.x, gyr.y, gyr.z));
    first_imu_frame_ = false;
  }

  for (const auto& imu_msg : sensor_measurement.imu_buff_) {
    temp_q = Eigen::Quaterniond(imu_msg.orientation.w,
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z);
    Eigen::Vector3d acc = Eigen::Vector3d(imu_msg.linear_acceleration.x,
                        imu_msg.linear_acceleration.y,
                        imu_msg.linear_acceleration.z) - temp_q.toRotationMatrix().transpose() * g_;
    Eigen::Vector3d gyr(imu_msg.angular_velocity.x,
                        imu_msg.angular_velocity.y,
                        imu_msg.angular_velocity.z);

    imu_init_buff_.emplace_back(acc, gyr);
  }

  if (imu_init_buff_.size() < max_init_count_) {
    return false;
  }

  Eigen::Vector3d acc_cov, gyr_cov;
  ComputeMeanAndCovDiag(
      imu_init_buff_,
      mean_acc_,
      acc_cov,
      [](const std::pair<Eigen::Vector3d, Eigen::Vector3d>& imu_data) {
        return imu_data.first;
      });
  ComputeMeanAndCovDiag(
      imu_init_buff_,
      mean_gyr_,
      gyr_cov,
      [](const std::pair<Eigen::Vector3d, Eigen::Vector3d>& imu_data) {
        return imu_data.second;
      });

  temp_q = Eigen::Quaterniond(back_imu.orientation.w,
                            back_imu.orientation.x,
                            back_imu.orientation.y,
                            back_imu.orientation.z);

  curr_state_.pose.block<3, 3>(0, 0) = temp_q.normalized().toRotationMatrix();

  curr_state_.pose.block<3, 1>(0, 3).setZero();

  // init velocity
  curr_state_.vel.setZero();
  // init bg
  curr_state_.bg.setZero();
  // init ba
  curr_state_.ba = mean_acc_;

  prev_state_ = curr_state_;

  P_.setIdentity();
  P_.block<3, 3>(IndexErrorOri, IndexErrorOri) =
      config_.init_ori_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorPos, IndexErrorPos) =
      config_.init_pos_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorVel, IndexErrorVel) =
      config_.init_vel_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorBiasAcc, IndexErrorBiasAcc) =
      config_.init_ba_cov * Eigen::Matrix3d::Identity();
  P_.block<3, 3>(IndexErrorBiasGyr, IndexErrorBiasGyr) =
      config_.init_bg_cov * Eigen::Matrix3d::Identity();

  Q_.setIdentity();
  Q_.block<3, 3>(IndexNoiseAccLast, IndexNoiseAccLast) =
      config_.acc_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseGyrLast, IndexNoiseGyrLast) =
      config_.gyr_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseAccCurr, IndexNoiseAccCurr) =
      config_.acc_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseGyrCurr, IndexNoiseGyrCurr) =
      config_.gyr_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseBiasAcc, IndexNoiseBiasAcc) =
      config_.ba_cov * Eigen::Matrix3d::Identity();
  Q_.block<3, 3>(IndexNoiseBiasGyr, IndexNoiseBiasGyr) =
      config_.bg_cov * Eigen::Matrix3d::Identity();

  lio_time_ = sensor_measurement.imu_buff_.back().header.stamp.toSec();
  lio_init_ = true;

  return true;
}

void LIO::RecordState(double lid_end_time) {
  LidarState lidar_state;
  lidar_state.time = lid_end_time;
  lidar_state.T = curr_state_.pose * config_.T_imu_lidar;
  lidar_state.vel = curr_state_.vel; // we ignore the Coriolis effect on lidar velocity
  lidar_states_.push_back(lidar_state);
  if (lidar_states_.size() > 2) {
    lidar_states_.pop_front();
  }
}

bool LIO::CorrectState(const State& state,
                       const Eigen::Matrix<double, 15, 1>& delta_x,
                       State& corrected_state) {
  // ori
  Eigen::Matrix3d delta_R =
      Sophus::SO3d::exp(delta_x.block<3, 1>(IndexErrorOri, 0)).matrix();
  // The normalization is employed after each update to pervent numerical
  // stability
  Eigen::Quaterniond temp_q(state.pose.block<3, 3>(0, 0) * delta_R);
  temp_q.normalize();
  corrected_state.pose.block<3, 3>(0, 0) = temp_q.toRotationMatrix();
  // pos
  corrected_state.pose.block<3, 1>(0, 3) =
      state.pose.block<3, 1>(0, 3) + delta_x.block<3, 1>(IndexErrorPos, 0);
  // vel
  corrected_state.vel = state.vel + delta_x.block<3, 1>(IndexErrorVel, 0);
  // ba
  corrected_state.ba = state.ba + delta_x.block<3, 1>(IndexErrorBiasAcc, 0);
  // bg
  corrected_state.bg = state.bg + delta_x.block<3, 1>(IndexErrorBiasGyr, 0);

  return true;
}

bool LIO::ComputeFinalCovariance(const Eigen::Matrix<double, 15, 1>& delta_x) {
  Eigen::Matrix<double, 15, 15> temp_P = final_hessian_.inverse();

  // project covariance
  Eigen::Matrix<double, 15, 15> L = Eigen::Matrix<double, 15, 15>::Identity();
  L.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() -
                        0.5 * Sophus::SO3d::hat(delta_x.block<3, 1>(0, 0));
  P_ = L * temp_P * L;

  return true;
}

bool LIO::IsConverged(const Eigen::Matrix<double, 15, 1>& delta_x) {
  return delta_x.lpNorm<Eigen::Infinity>() < transformation_epsilon_;
}

Eigen::Matrix4d SE3Inverse(const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block<3,3>(0,0);
  Eigen::Vector3d t = T.block<3,1>(0,3);
  Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();

  T_inv.block<3,3>(0,0) = R.transpose();
  T_inv.block<3,1>(0,3) = -R.transpose() * t;

  return T_inv;
}

bool extractImuToCsv(const std::string& bag_path,
                     const std::vector<std::string>& imu_topics,
                     const std::string& csv_path, double gyro_scale, double acc_scale)
{
    try {
        rosbag::Bag bag;
        bag.open(bag_path, rosbag::bagmode::Read);

        std::vector<std::string> topics = imu_topics;
        if (topics.empty()) {
            std::cerr << "[extractImuToCsv] No topics specified.\n";
            bag.close();
            return false;
        }

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        std::ofstream out(csv_path);
        if (!out) {
            std::cerr << "[extractImuToCsv] Cannot open CSV: " << csv_path << "\n";
            bag.close();
            return false;
        }

        // --- Header ---
        out << "time,gx,gy,gz,ax,ay,az,ox,oy,oz,ow,ros_time\n";

        // --- Messages ---
        size_t count = 0;
        char timebuf[64], rostimebuf[64];

        for (const rosbag::MessageInstance& m : view) {
            auto imu = m.instantiate<sensor_msgs::Imu>();
            if (!imu) continue;

            const ros::Time& t = imu->header.stamp;   // message time
            const ros::Time& rt = m.getTime();        // rosbag time

            // format times as sec.nsec with 9 digits
            snprintf(timebuf, sizeof(timebuf), "%d.%09d", t.sec, t.nsec);
            snprintf(rostimebuf, sizeof(rostimebuf), "%d.%09d", rt.sec, rt.nsec);

            out << timebuf << ','
                << imu->angular_velocity.x * gyro_scale << ','
                << imu->angular_velocity.y * gyro_scale << ','
                << imu->angular_velocity.z * gyro_scale << ','
                << imu->linear_acceleration.x * acc_scale << ','
                << imu->linear_acceleration.y * acc_scale << ','
                << imu->linear_acceleration.z * acc_scale << ','
                << imu->orientation.x << ','
                << imu->orientation.y << ','
                << imu->orientation.z << ','
                << imu->orientation.w << ','
                << rostimebuf << '\n';

            ++count;
        }

        std::cerr << "[extractImuToCsv] Wrote " << count
                  << " rows to " << csv_path << "\n";

        bag.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[extractImuToCsv] Exception: " << e.what() << "\n";
        return false;
    }
}


static std::pair<int,int> bracketStates(const cba::StampedStateVector& states, const ros::Time& t) {
  if (states.empty()) return {-1,-1};
  if (t <= states.front().time) return {0,0};
  if (t >= states.back().time)  return {int(states.size()-1), int(states.size()-1)};
  int lo = 0, hi = int(states.size()) - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (states[mid].time <= t) lo = mid; else hi = mid;
  }
  return {lo, lo+1};
}

static Eigen::Vector3d interpGyroBiasB(const cba::StampedStateVector& states, const ros::Time& t) {
  auto [i0, i1] = bracketStates(states, t);
  if (i0 < 0) return Eigen::Vector3d::Zero();
  if (i0 == i1) return states[i0].bg;
  const double t0 = states[i0].time.toSec(), t1 = states[i1].time.toSec();
  const double tau = (t1 > t0) ? ((t.toSec() - t0) / (t1 - t0)) : 0.0;
  return (1.0 - tau) * states[i0].bg + tau * states[i1].bg;
}

static Eigen::Vector3d interpAccBiasB(const cba::StampedStateVector& states, const ros::Time& t) {
  auto [i0, i1] = bracketStates(states, t);
  if (i0 < 0) return Eigen::Vector3d::Zero();
  if (i0 == i1) return states[i0].ba;
  const double t0 = states[i0].time.toSec(), t1 = states[i1].time.toSec();
  const double tau = (t1 > t0) ? ((t.toSec() - t0) / (t1 - t0)) : 0.0;
  return (1.0 - tau) * states[i0].ba + tau * states[i1].ba;
}

static inline Eigen::Vector3d tangential(const Eigen::Vector3d& alpha, const Eigen::Vector3d& r) {
  return alpha.cross(r);
}
static inline Eigen::Vector3d centripetal(const Eigen::Vector3d& omega, const Eigen::Vector3d& r) {
  return omega.cross(omega.cross(r));
}

struct AlphaEstimator {
  bool has_prev = false;
  ros::Time t_prev;
  Eigen::Vector3d w_prev = Eigen::Vector3d::Zero();
  Eigen::Vector3d update(const ros::Time& t, const Eigen::Vector3d& w_L) {
    if (!has_prev) { has_prev = true; t_prev = t; w_prev = w_L; return Eigen::Vector3d::Zero(); }
    const double dt = (t.toSec() - t_prev.toSec());
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    if (dt > 0) a = (w_L - w_prev) / dt;
    t_prev = t; w_prev = w_L;
    return a;
  }
};

bool extractAndConvertImu(const std::string& ros1_bag, const std::string& lio_states_txt, 
                          const std::string& imu_topic, const Eigen::Isometry3d& B_T_L,
                          const std::string& csv_path, double gyro_scale, double acc_scale) {
  // first extract all imu messages from rosbag, these msgs are omega_{WB,s}^B and a_{WB,s}^B
  // second compute the omega_{WL,s}^L = L_R_B * (omega_{WB,s}^B - bg), the suberscript s means sensed.
  // third compute the specific force acceleration in the L frame by
  // a_{WL,s}^L = L_R_B * a_{WB,s}^B - (\Omega_{WL}^{L, 2} L_p_B + \dot{\Omega}_{WL}^L L_p_B)
  // In the below interpolation for \Omega_{WL}^{L}, we will omega_{WB}^{B} = omega_{WB,s}^{B} - b_g, then 
  // omega_{WL}^{L} = R_LB * omega_{WB}^{B}
  // Load states (for bias interpolation)
  cba::StampedStateVector states;
  if (!loadStates(lio_states_txt, &states)) {
    std::cerr << "No states loaded from " << lio_states_txt << "\n";
    return false;
  }
  std::sort(states.begin(), states.end(),
            [](const cba::StampedState& a, const cba::StampedState& b){ return a.time < b.time; });

  // Extrinsics: x_B = R_BL x_L + t_BL
  const Eigen::Matrix3d R_BL = B_T_L.rotation();
  const Eigen::Matrix3d R_LB = R_BL.transpose();
  const Eigen::Vector3d t_BL = B_T_L.translation();
  const Eigen::Vector3d r_L  = - R_LB * t_BL;   // ^L p_B

  rosbag::Bag bag;
  try { bag.open(ros1_bag, rosbag::bagmode::Read); }
  catch (const rosbag::BagException& e) {
    std::cerr << "Failed to open bag: " << e.what() << "\n";
    return false;
  }
  rosbag::View view(bag, rosbag::TopicQuery({imu_topic}));

  std::ofstream ofs(csv_path);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open csv for write: " << csv_path << "\n";
    bag.close();
    return false;
  }
  ofs << std::fixed << std::setprecision(9);
  ofs << "# sec.nsec,gx,gy,gz,ax,ay,az (L-frame, bg/ba-comp, lever-arm corrected)\n";

  AlphaEstimator alpha_est;
  size_t n_written = 0;
  size_t n_large_accel = 0;

  for (const rosbag::MessageInstance& m : view) {
    auto imu = m.instantiate<sensor_msgs::Imu>();
    if (!imu) continue;

    const ros::Time t = imu->header.stamp;

    // Raw in B (sensor outputs)
    Eigen::Vector3d w_B_meas(imu->angular_velocity.x,
                             imu->angular_velocity.y,
                             imu->angular_velocity.z);
    Eigen::Vector3d a_B_meas(imu->linear_acceleration.x,
                             imu->linear_acceleration.y,
                             imu->linear_acceleration.z);

    // Apply optional global scales to measurements ONLY (biases are in SI already)
    w_B_meas *= gyro_scale;
    a_B_meas *= acc_scale;

    // Interpolate biases in B and subtract
    const Eigen::Vector3d bg_B = interpGyroBiasB(states, t);
    const Eigen::Vector3d ba_B = interpAccBiasB(states, t);
    const Eigen::Vector3d w_B  = w_B_meas - bg_B;
    const Eigen::Vector3d f_B  = a_B_meas - ba_B;   // specific force in B after bias removal

    // Rotate to L
    const Eigen::Vector3d w_L     = R_LB * w_B;
    const Eigen::Vector3d f_L_rot = R_LB * f_B;

    // Angular acceleration in L by finite differencing
    const Eigen::Vector3d alpha_L = alpha_est.update(t, w_L);

    // Lever-arm correction: f_L = R_LB f_B - α×r - ω×(ω×r)
    const Eigen::Vector3d f_L = f_L_rot
                              - tangential(alpha_L, r_L)
                              - centripetal(w_L, r_L);

    if (f_L.lpNorm<Eigen::Infinity>() > 40) {
      std::cerr << "Warn: large acceleration " << t.sec << "." << std::setw(9) << std::setfill('0') << t.nsec
        << ", w_L " << std::setprecision(9) << w_L.x() << "," << w_L.y() << "," << w_L.z() << ", f_L "
        << f_L.x() << "," << f_L.y() << "," << f_L.z() << ", alpha_L " << alpha_L.transpose()
        << "\n\tw_L " << w_L.transpose() << ", f_L_rot " << f_L_rot.transpose()
        << ", tangential " << tangential(alpha_L, r_L).transpose()
        << ", centripetal " << centripetal(w_L, r_L).transpose() << "\n\tbg_B " << bg_B.transpose() 
        << ", ba_B " << ba_B.transpose() << ", w_B "<< w_B.transpose() << ", f_B " << f_B.transpose() << std::endl;
      n_large_accel++;
    }
    // CSV line
    ofs << t.sec << "." << std::setw(9) << std::setfill('0') << t.nsec
        << "," << std::setprecision(9)
        << w_L.x() << "," << w_L.y() << "," << w_L.z() << ","
        << f_L.x() << "," << f_L.y() << "," << f_L.z() << "\n";
    ++n_written;
  }

  ofs.close();
  bag.close();
  std::cout << "Wrote " << n_written << " rows to " << csv_path << "\n";
  if (n_large_accel) {
    std::cout << "Warn: detected " << n_large_accel
          << (n_large_accel == 1 ? " large-acceleration event. " : " large-acceleration events. ")
          << "Possible cause: clustered IMU timestamps (tiny/irregular dt)."
          << std::endl;
  }
  return (n_written > 0);
}


// --- main function: bias removal only, output in B frame ---
bool extractAndCompensateImu(const std::string& ros1_bagfile,
                             const std::string& lio_states_txt,
                             const std::string& imu_topic,
                             const std::string& csv_path,
                             double gyro_scale,
                             double acc_scale)
{
  // 1) Load states for bias interpolation
  cba::StampedStateVector states;
  if (!loadStates(lio_states_txt, &states)) {
    ROS_ERROR_STREAM("Failed to load states from " << lio_states_txt
                     << ". Will proceed with zero biases.");
  } else {
    std::sort(states.begin(), states.end(),
              [](const cba::StampedState& a, const cba::StampedState& b){ return a.time < b.time; });
  }

  // 2) Open bag/view
  rosbag::Bag bag;
  try { bag.open(ros1_bagfile, rosbag::bagmode::Read); }
  catch (const rosbag::BagException& e) {
    ROS_ERROR_STREAM("Open bag failed: " << e.what());
    return false;
  }
  rosbag::View view(bag, rosbag::TopicQuery({imu_topic}));

  // 3) CSV
  std::ofstream ofs(csv_path);
  if (!ofs.is_open()) {
    ROS_ERROR_STREAM("Open CSV failed: " << csv_path);
    bag.close();
    return false;
  }
  ofs << std::fixed;
  ofs << "# time(gps sec.nsec), gx, gy, gz, ax, ay, az  (B-frame, bias-compensated)\n";

  // 4) Iterate IMU
  size_t n = 0;
  for (const rosbag::MessageInstance& mi : view) {
    auto imu = mi.instantiate<sensor_msgs::Imu>();
    if (!imu) continue;
    const ros::Time t = imu->header.stamp;

    // Measured (sensor outputs) in B
    Eigen::Vector3d w_meas(imu->angular_velocity.x,
                           imu->angular_velocity.y,
                           imu->angular_velocity.z);
    Eigen::Vector3d a_meas(imu->linear_acceleration.x,
                           imu->linear_acceleration.y,
                           imu->linear_acceleration.z);

    // Optional scalar calibration factors (apply to measurements, not biases)
    w_meas *= gyro_scale;
    a_meas *= acc_scale;

    // Interpolate biases (in B) and subtract
    const Eigen::Vector3d bg_B = interpGyroBiasB(states, t);
    const Eigen::Vector3d ba_B = interpAccBiasB(states, t);

    const Eigen::Vector3d w_B = w_meas - bg_B;   // rad/s
    const Eigen::Vector3d f_B = a_meas - ba_B;   // m/s^2 (specific force if driver publishes that)

    // Write time as sec.nsec with 9 digits
    ofs << t.sec << "." << std::setw(9) << std::setfill('0') << t.nsec
        << std::setfill(' ') << std::setprecision(9)
        << "," << w_B.x() << "," << w_B.y() << "," << w_B.z()
        << "," << f_B.x() << "," << f_B.y() << "," << f_B.z() << "\n";
    ++n;
  }

  ofs.close();
  bag.close();
  ROS_INFO_STREAM("extractAndCompensateImu: wrote " << n << " rows to " << csv_path);
  return n > 0;
}
