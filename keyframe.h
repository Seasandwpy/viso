

#include "types.h"
#include <memory>

class Keyframe
{
private:
  static long next_id_ = 0;
  cv::Mat mat_;
  long id_;
  std::vector<cv::Keypoint> keypoints_;
  M3d R_;
  V3d T_;

public:
  using Ptr = std::shared_ptr<Keyframe>;

  Keyframe(cv::Mat mat) : mat_(mat)
  {
    id_ = next_id_;
    next_id++;

    R_ = M3d::Identity();
    T_ = V3d::Zero();
  }

  ~Keyframe() = default;

  inline double GetPixelValue(const double &x, const double &y)
  {
    U8 *data = &mat.data[int(y) * mat.step + int(x)];
    double xx = x - floor(x);
    double yy = y - floor(y);
    return double(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[mat.step] +
        xx * yy * data[mat.step + 1]);
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

  inline std::vector<cv::Keypoint> &Keypoints() { return keypoints_; }
  inline cv::Mat Mat() { return mat_; }
  inline M3d &R() { return R_; }
  inline V3d &T() { return T_; }

  static long GetNextId()
  {
    return next_id_;
  }
};
