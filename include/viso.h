
#ifndef VISO_VISO_H
#define VISO_VISO_H

#include "keyframe.h"
#include "frame_sequence.h"

class Viso : public FrameSequence::FrameHandler
{
private:
  /* data */
  M3d K;
  M3d K_inv;

  std::vector<V3d> map_;
  Keyframe::Ptr last_frame;

  enum State {
    kInitialization = 0,
    kRunning = 1,
  };

  State state_;

public:
  Viso(double fx, double fy, double cx, double cy) {
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    K_inv = K.inverse();
    state_ = kInitialization;
  }

  ~Viso() = default;

  void OnNewFrame(Keyframe::Ptr keyframe);

  inline const std::vector<V3d> &GetMap() { return map_; }

private:
  // |points1| and |points2| are observations on the image plane,
  // meaning that the inverse intrinsic matrix has already been applied to
  // the respective pixel coordinates.
  void PoseEstimation2d2d(
    std::vector<V3d> points1,
    std::vector<V3d> points2,
    M3d &R, V3d &T, std::vector<bool> &inliers, int &nr_inliers);

  // Same as above.
  void Reconstruct3DPoints(const M3d &R, const V3d &T,
                           const std::vector<V3d> &points1, const std::vector<V3d> &points2,
                           std::vector<V3d> &points3d, const std::vector<bool> &inliers, const int &nr_inliers);

  // Triangulation, see paper "Triangulation", Section 5.1, by Richard I. Hartley, Peter Sturm
  void Triangulate(const M34d &Pi1, const M34d &Pi2, const V3d &x1, const V3d &x2, V3d &P);

  void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse);

  void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse);
};

#endif