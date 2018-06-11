

#ifndef VISO_KEYFRAME_H
#define VISO_KEYFRAME_H

#include "types.h"
#include <memory>
#include <opencv2/opencv.hpp>

class Keyframe {
private:
    static long next_id_;
    cv::Mat mat_;
    long id_;
    std::vector<cv::KeyPoint> keypoints_;
    M3d R_;
    V3d T_;
    M3d K_;

    const int nr_pyramids_ = 4;
    const double pyramid_scale_ = 0.5;
    const double scales_[4] = { 1.0, 0.5, 0.25, 0.125 };
    std::vector<cv::Mat> pyramids_;

public:
    using Ptr = std::shared_ptr<Keyframe>;

    Keyframe(cv::Mat mat)
        : mat_(mat)
    {
        id_ = next_id_;
        next_id_++;

        R_ = M3d::Identity();
        T_ = V3d::Zero();

        // Pyramids
        pyramids_.push_back(mat_);

        for (int i = 1; i < nr_pyramids_; i++) {
            cv::Mat pyr;
            cv::pyrDown(pyramids_[i - 1], pyr,
                cv::Size(pyramids_[i - 1].cols * pyramid_scale_, pyramids_[i - 1].rows * pyramid_scale_));
            pyramids_.push_back(pyr);
        }
    }

    ~Keyframe() = default;

    inline double GetPixelValue(const double& x, const double& y, int level = 0)
    {
        U8* data = &pyramids_[level].data[int(y) * pyramids_[level].step + int(x)];
        double xx = x - floor(x);
        double yy = y - floor(y);
        return double(
            (1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[pyramids_[level].step] + xx * yy * data[pyramids_[level].step + 1]);
    }

    inline V2d GetGradient(const double& u, const double& v, int level = 0)
    {
        double dx = 0.5 * (GetPixelValue(u + 1, v, level) - GetPixelValue(u - 1, v, level));
        double dy = 0.5 * (GetPixelValue(u, v + 1, level) - GetPixelValue(u, v - 1, level));
        return V2d(dx, dy);
    }

    inline long GetId()
    {
        return id_;
    }

    inline bool IsInside(const V3d& point, int level = 0)
    {
        V2d uv = Project(point, level);
        return IsInside(uv.x(), uv.y(), level);
    }

    inline bool IsInside(const double& u, const double& v, int level = 0)
    {
        return u >= 0 && u < pyramids_[level].cols && v >= 0 && v < pyramids_[level].rows;
    }

    inline V2d Project(const V3d& point, int level)
    {
        V3d uv1 = R_ * point + T_;
        uv1 /= uv1.z();
        double u = scales_[level] * (uv1.x() * K_(0, 0) + K_(0, 2));
        double v = scales_[level] * (uv1.y() * K_(1, 1) + K_(1, 2));
        return V2d{ u, v };
    }

    // Calculate the viewing angle, i.e. the angle between the vector from the
    // origin of the camera to the point and the camera's Z-axis.
    inline double ViewingAngle(const V3d& Pw)
    {
        V3d Pc = R_ * Pw + T_;
        Pc.normalize();
        return std::acos(Pc.z());
    }

    inline std::vector<cv::KeyPoint>& Keypoints() { return keypoints_; }

    inline const cv::Mat& Mat() { return mat_; }
    inline M3d GetR() { return R_; }
    inline V3d GetT() { return T_; }
    inline void SetT(V3d T) { T_ = T; }
    inline void SetR(M3d R) { R_ = R; }
    inline M3d GetK() { return K_; }
    inline void SetK(M3d K) { K_ = K; }

    inline double GetScale(int level) { return scales_[level]; }

    inline const std::vector<cv::Mat>& Pyramids() { return pyramids_; }

    static long GetNextId()
    {
        return next_id_;
    }

    inline int AddKeypoint(cv::KeyPoint kp) {
      keypoints_.push_back(kp);
      return keypoints_.size();
    }
};

#endif
