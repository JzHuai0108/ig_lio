#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <iostream>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <ros/time.h>
// #include <sensor_msgs/PointCloud2.h>

namespace cba {

enum class CostType : std::uint8_t {
  PriorPose = 0,
  RelativePose,
  ControlPointPosition,
  ImuErrorWithGravity,
  PriorSpeedAndBias,
};

inline std::vector<uint32_t> makePower10() {
  int len = 10;
  std::vector<uint32_t> pow10(len);
  pow10[0] = 1;
  for (int i = 1; i < len; ++i) {
    pow10[i] = pow10[i - 1] * 10;
  }
  return pow10;
}

std::pair<uint32_t, uint32_t> parseTime(const std::string &time);

void testParseTime();

struct StampedPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ros::Time time;
  Eigen::Vector3d r;
  Eigen::Quaterniond q;

  StampedPose() {}
  explicit StampedPose(const ros::Time& _time) : time(_time) {}
  StampedPose(const ros::Time& _time, const Eigen::Vector3d& _r, const Eigen::Quaterniond& _q) : time(_time), r(_r), q(_q) {}
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pose() const {
      return std::make_pair(r, q);
  }
  
  void setPose(const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& pose) {
      r = pose.first;
      q = pose.second;
  }

  Eigen::Isometry3d betweenPose(const StampedPose& y) const {
    const Eigen::Quaterniond q_rel = q.conjugate() * y.q;
    const Eigen::Vector3d t_rel = q.conjugate() * (y.r - r);
    Eigen::Isometry3d T_rel = Eigen::Isometry3d::Identity();
    T_rel.linear()      = q_rel.toRotationMatrix();
    T_rel.translation() = t_rel;
    return T_rel;
  }

  inline Eigen::Isometry3f isof() const {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.linear() = q.toRotationMatrix().cast<float>();
    T.translation() = Eigen::Vector3f(static_cast<float>(r.x()),
                                      static_cast<float>(r.y()),
                                      static_cast<float>(r.z()));
    return T;
  }

  /**
   * @brief to string without annotations.
   * @return
   */
  std::string toString() const;

  friend bool operator< (const StampedPose& c1, const StampedPose& c2);
  /**
   * @brief to pretty string with annotations.
   * @return
   */
  friend std::ostream& operator<<(std::ostream& os, const StampedPose& p);
};

inline bool operator< (const StampedPose& c1, const StampedPose& c2) {
    return c1.time < c2.time;
}

struct IncrementMotion {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d pij;  // q_wi * p_ij = p_wj - p_wi
    Eigen::Quaterniond qij; // q_wi * q_ij = q_wj
    Eigen::Vector3d vij; // q_wi * v_ij + v_wi = v_wj

    Eigen::Vector3d ratio(const IncrementMotion &ref) const;

    void multiply(const Eigen::Vector3d &coeffs);
};

struct StampedState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ros::Time time;
  Eigen::Vector3d r;
  Eigen::Quaterniond q;
  Eigen::Vector3d v;
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  Eigen::Vector3d gW;

  StampedState() {}

  StampedState(const ros::Time &_t, const Eigen::Vector3d &_r,
               const Eigen::Quaterniond &_q)
      : time(_t), r(_r), q(_q) {
    v.setZero();
    bg.setZero();
    ba.setZero();
    gW = Eigen::Vector3d(0, 0, -9.81);
  }

  StampedState leftTransform(const Eigen::Quaterniond& q_left) const {
    StampedState result(*this);
    result.r  = q_left * result.r;      // position
    result.q  = q_left * result.q;      // orientation quaternion
    result.v  = q_left * result.v;      // velocity vector
    result.gW = q_left * result.gW;     // gravity vector
    return result;
  }

  void updatePose(const Eigen::Matrix3d& _R, const Eigen::Vector3d& _p) {
    r = _p;
    q = Eigen::Quaterniond(_R);
  }

  void updatePose(const Eigen::Quaterniond& _q, const Eigen::Vector3d& _p) {
    r = _p;
    q = _q;
  }

  void setGravity(const Eigen::Vector3d& _gW) {
    gW = _gW;
  }

  Eigen::Vector3d trans() const {
    return r;
  }

  Eigen::Isometry3d toIsometry3d() const {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = r;
    T.linear() = q.toRotationMatrix();
    return T;
  }

  Eigen::Isometry3f toIsometry3f() const {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.translation() = r.cast<float>();
    T.linear() = q.toRotationMatrix().cast<float>();
    return T;
  }

  IncrementMotion between(const StampedState &sj) const;

  Eigen::Isometry3d betweenPose(const StampedState &sj) const;

  void retract(const IncrementMotion &im);

  bool operator<(const StampedState &rhs) const {
    return time < rhs.time;
  }

  /**
   * @brief to pretty string without annotations.
   * @return
   */
  std::string toString() const;
  /**
   * @brief operator << print with annotations.
   * @param os
   * @param s
   * @return
   */
  friend std::ostream& operator<<(std::ostream& os, const StampedState& s);
  friend std::istream& operator>>(std::istream& is, StampedState& s);
};

/// Compute median of a non-empty vector of doubles.
inline double median_of(std::vector<double> v) {
    size_t n = v.size();
    auto mid = v.begin() + n/2;
    std::nth_element(v.begin(), mid, v.end());
    double m = *mid;
    if ((n & 1) == 0) {
        auto midm1 = v.begin() + (n/2 - 1);
        std::nth_element(v.begin(), midm1, v.end());
        m = 0.5 * (m + *midm1);
    }
    return m;
}

// Quantile helper (q in [0,1])
inline double quantile_of(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    q = std::min(std::max(q, 0.0), 1.0);
    size_t k = static_cast<size_t>(q * (v.size()-1));
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
}


struct ImuData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ros::Time time;
  Eigen::Vector3d g;
  Eigen::Vector3d a;
  friend std::ostream& operator<<(std::ostream& os, const ImuData& d);
  friend std::istream& operator>>(std::istream& is, ImuData& d);
};

typedef std::vector<StampedPose, Eigen::aligned_allocator<StampedPose>>
    StampedPoseVector;

typedef std::vector<StampedState, Eigen::aligned_allocator<StampedState>>
    StampedStateVector;

typedef std::vector<ImuData, Eigen::aligned_allocator<ImuData>>
    ImuDataVector;

size_t loadStates(const std::string &stateFile,
                  StampedStateVector *states);

// Inputs: prop.aabb_min (mn), prop.aabb_max (mx), prop.n (unit normal)
double approxAreaFromAabbAndNormal(const Eigen::Vector3d& mn,
                                   const Eigen::Vector3d& mx,
                                   const Eigen::Vector3d& n);
}  // namespace cba

#endif // COMMON_H
