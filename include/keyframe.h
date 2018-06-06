

#ifndef VISO_KEYFRAME_H
#define VISO_KEYFRAME_H

#include "types.h"
#include <memory>
#include <opencv2/opencv.hpp>

class Keyframe
{
private:
  static long next_id_;
  cv::Mat mat_;
  long id_;
  std::vector<cv::KeyPoint> keypoints_;
  M3d R_;
  V3d T_;

public:
  using Ptr = std::shared_ptr<Keyframe>;

  Keyframe(cv::Mat mat) : mat_(mat)
  {
    id_ = next_id_;
    next_id_++;

    R_ = M3d::Identity();
    T_ = V3d::Zero();
  }

  ~Keyframe() = default;

  inline double GetPixelValue(const double &x, const double &y)
  {
    U8 *data = &mat_.data[int(y) * mat_.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[mat_.step] +
        xx * yy * data[mat_.step + 1]);
  }

  inline V2d GetGradient(const double &u, const double &v)
  {
    double dx = 0.5 * (GetPixelValue(u + 1, v) - GetPixelValue(u - 1, v));
    double dy = 0.5 * (GetPixelValue(u, v + 1) - GetPixelValue(u, v - 1));
    return V2d(dx, dy);
  }

  inline long GetId()
  {
    return id_;
  }

  inline std::vector<cv::KeyPoint> &Keypoints() { return keypoints_; }
  inline cv::Mat Mat() { return mat_; }
  inline M3d &R() { return R_; }
  inline V3d &T() { return T_; }

  static long GetNextId()
  {
    return next_id_;
  }
};

#endif
