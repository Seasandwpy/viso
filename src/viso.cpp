#include "viso.h"
#include "common.h"
#include "timer.h"
#include <opencv2/core/eigen.hpp>

void Viso::OnNewFrame(Keyframe::Ptr current_frame)
{
    static int max_inliers = 0;
    static int frame_cnt = 0;

    frame_cnt++;

    switch (state_)
    {
    case kInitialization:
        if (current_frame->GetId() != 0)
        {
            std::vector<bool> success;
          current_frame->Keypoints() = last_frame->Keypoints();
            OpticalFlowMultiLevel(ref_frame->Mat(),
                                  current_frame->Mat(), ref_frame->Keypoints(), current_frame->Keypoints(), success, true);

            std::vector<V3d> p1;
            std::vector<V3d> p2;

          std::vector<int> kp_indices;

            for (int i = 0; i < current_frame->Keypoints().size(); i++)
            {
                if (success[i])
                {
                    p1.emplace_back(K_inv * V3d{ref_frame->Keypoints()[i].pt.x, ref_frame->Keypoints()[i].pt.y, 1});
                    p2.emplace_back(
                        K_inv * V3d{current_frame->Keypoints()[i].pt.x, current_frame->Keypoints()[i].pt.y, 1});
                  kp_indices.push_back(i);
                }
            }

            std::vector<bool> inliers;
            int nr_inliers = 0;
            PoseEstimation2d2d(p1, p2, current_frame->R(), current_frame->T(), inliers, nr_inliers);
            Reconstruct(p1, p2, current_frame->R(), current_frame->T(), inliers, nr_inliers, map_);

            if (nr_inliers > max_inliers)
            {
                max_inliers = nr_inliers;
            }

          cv::Mat img;
          cv::cvtColor(current_frame->Mat(), img, CV_GRAY2BGR);

          for (int i = 0; i < p2.size(); i++) {
            //if (inliers[i])
            //{
            cv::Scalar color(255, 0, 0);
            if (nr_inliers > 0 && inliers[i]) {
              color = cv::Scalar(0, 255, 0);
            }
            cv::Point2f kp1 = ref_frame->Keypoints()[kp_indices[i]].pt;
            V3d kp2 = K * p2[i];
            cv::line(img, kp1, {(int) kp2.x(), (int) kp2.y()}, color);
            //}
          }

          cv::imshow("Optical flow", img);
            std::cout << "Tracked points: " << p1.size() << ", Inliers: " << nr_inliers << "(" << max_inliers << ")\n";

            cv::waitKey(10);
          const double thresh = 0.9;
          if (p1.size() > 50 && nr_inliers > 0 && (nr_inliers / (double) p1.size()) > thresh)
            {
                std::cout << "Initialized!\n";
                state_ = kRunning;
            }

            if (frame_cnt > reinitialize_after)
            {
              cv::FAST(current_frame->Mat(), current_frame->Keypoints(), 60);
                ref_frame = current_frame;
                frame_cnt = 0;
            }
        }
        else
        {
          cv::FAST(current_frame->Mat(), current_frame->Keypoints(), fast_thresh);
            ref_frame = current_frame;
        }

        break;

    default:
        break;
    }

    last_frame = current_frame;
}

// 2D-2D pose estimation functions
void Viso::PoseEstimation2d2d(
    std::vector<V3d> kp1,
    std::vector<V3d> kp2,
    M3d &R, V3d &T, std::vector<bool> &inliers, int &nr_inliers)
{
    std::vector<cv::Point2f> kp1_;
    std::vector<cv::Point2f> kp2_;

    for (int i = 0; i < kp1.size(); ++i)
    {
        kp1_.push_back({(float)kp1[i].x(), (float)kp1[i].y()});
        kp2_.push_back({(float)kp2[i].x(), (float)kp2[i].y()});
    }

    cv::Mat outlier_mask;

  double thresh = projection_error_thresh / std::sqrt(K(0, 0) * K(0, 0) + K(1, 1) * K(1, 1));
  cv::Mat essential = cv::findEssentialMat(kp1_, kp2_, 1.0, {0.0, 0.0}, CV_FM_RANSAC, 0.99, thresh, outlier_mask);

  if (essential.data == NULL) {
    nr_inliers = 0;
    return;
  }

    cv::Mat Rmat, tmat;

    // This method does the depth check. Only users points which are not masked out by
    // the outlier mask.
    recoverPose(essential, kp1_, kp2_, Rmat, tmat, 1.0, {}, outlier_mask);

    cv::cv2eigen(Rmat, R);
    cv::cv2eigen(tmat, T);

    inliers = std::vector<bool>(kp1.size());
    for (int i = 0; i < inliers.size(); ++i)
    {
        inliers[i] = outlier_mask.at<bool>(i) > 0;
        nr_inliers += inliers[i];
    }
}

void Viso::OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse)
{

    // parameters
    int half_patch_size = 4;
  int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++)
        {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {

                    double error = 0;
                    V2d J; // Jacobian
                    if (!inverse)
                    {
                        // Forward Jacobian
                        J = -GetImageGradient(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    }
                    else
                    {
                        // Inverse Jacobian
                        J = -GetImageGradient(img1, kp.pt.x + x, kp.pt.y + y);
                    }

                    error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                            GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    // compute H, b and set cost;
                    H += J * J.transpose();
                    b += -J * error;
                    cost += error * error;
                }

            Eigen::Vector2d update = H.inverse() * b;

            if (std::isnan(update[0]))
            {
                // sometimes occurred when we have a black or white patch and H is irreversible
                std::cout << "update is nan" << std::endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost)
            {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial)
        {
            kp2[i].pt = kp.pt + cv::Point2f((float)dx, (float)dy);
        }
        else
        {
            cv::KeyPoint tracked = kp;
            tracked.pt += cv::Point2f((float)dx, (float)dy);
            kp2.push_back(tracked);
        }
    }
}

void Viso::OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse)
{
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
  std::vector<cv::Mat> pyr1, pyr2; // image pyramids

    for (int i = 0; i < pyramids; i++)
    {
        {
            if (i == 0)
            {
                pyr1.push_back(img1);
                pyr2.push_back(img2);
            }
            else
            {
                {
                  cv::Mat down;
                  cv::pyrDown(pyr1[i - 1], down,
                              cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
                  pyr1.push_back(down);
                }
                {
                  cv::Mat down;
                  cv::pyrDown(pyr2[i - 1], down,
                              cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
                  pyr2.push_back(down);
                }
            }
        }
    }

  // Scale the initial guess for kp2.
  for (int j = 0; j < kp2.size(); ++j) {
    kp2[j].pt *= scales[pyramids - 1];
    kp2[j].size *= scales[pyramids - 1];
  }

    for (int i = pyramids - 1; i >= 0; --i)
    {
      std::vector<cv::KeyPoint> kp1_ = kp1;

        for (int j = 0; j < kp1_.size(); ++j)
        {
            kp1_[j].pt *= scales[i];
            kp1_[j].size *= scales[i];
        }

      std::vector<bool> success_single;
        OpticalFlowSingleLevel(pyr1[i], pyr2[i], kp1_, kp2, success_single, true);
        success = success_single;

        if (i != 0)
        {
            for (int j = 0; j < kp1_.size(); ++j)
            {
                kp2[j].pt /= pyramid_scale;
                kp2[j].size /= pyramid_scale;
            }
        }
    }
}

// We want to find the 4d-coorindates of point P = (P1, P2, P3, 1)^T.
// lambda1 * x1 = Pi1 * P
// lambda2 * x2 = Pi2 * P
//
// I:   lambda1 * x11 = Pi11 * P
// II:  lambda1 * x12 = Pi12 * P
// III: lambda1       = Pi13 * P
//
// III in I and II
// I':   Pi13 * P * x11 = Pi11 * P
// II':  Pi13 * P * x12 = Pi12 * P
//
// I'':   (x11 * Pi13  - Pi11) * P = 0
// II'':  (x12 * Pi13  - Pi12) * p = 0
//
// We get another set of two equations from lambda2 * x2 = Pi2 * P.
// Finally we can construct a 4x4 matrix A such that A * P = 0
// A = [[x11 * Pi13  - Pi11];
//      [x12 * Pi13  - Pi12];
//      [x21 * Pi23  - Pi21];
//      [x22 * Pi13  - Pi22]];
void Viso::Triangulate(const M34d &Pi1, const M34d &Pi2, const V3d &x1, const V3d &x2, V3d &P)
{
    M4d A = M4d::Zero();
    A.row(0) = x1.x() * Pi1.row(2) - Pi1.row(0);
    A.row(1) = x1.y() * Pi1.row(2) - Pi1.row(1);
    A.row(2) = x2.x() * Pi2.row(2) - Pi2.row(0);
    A.row(3) = x2.y() * Pi2.row(2) - Pi2.row(1);

    Eigen::JacobiSVD<M4d> svd(A, Eigen::ComputeFullV);
    M4d V = svd.matrixV();

    // The solution is the last column V. This gives us a homogeneous
    // point, so we need to normalize.
    P = V.col(3).block<3, 1>(0, 0) / V(3, 3);
}

void Viso::Reconstruct(const std::vector<V3d> &p1,
                       const std::vector<V3d> &p2,
                       const M3d &R, const V3d &T, std::vector<bool> &inliers, int &nr_inliers, std::vector<V3d> &points3d)
{
  if (nr_inliers == 0) {
    return;
  }

    M34d Pi1 = MakePI0();
    M34d Pi2 = MakePI0() * MakeSE3(R, T);

#if 0
  V3d O1 = V3d::Zero();
  V3d O2 = -R*T;
#endif

    points3d.clear();

    int j = 0;
    for (int i = 0; i < p1.size(); ++i)
    {
        if (inliers[i])
        {
            V3d P1;
            Triangulate(Pi1, Pi2, p1[i], p2[i], P1);

#if 0
          // parallax
          V3d n1 = P1 - O1;
          V3d n2 = P1 - O2;
          double d1 = n1.norm();
          double d2 = n2.norm();

          double parallax = (n1.transpose () * n2);
          parallax /=  (d1 * d2);
          parallax = acos(parallax)*180/CV_PI;
          if (parallax > parallax_thresh)
          {
              inliers[i] = false;
              nr_inliers--;
              continue;
          }
#endif
          // projection error
            V3d P1_proj = P1 / P1.z();
            double dx = (P1_proj.x() - p1[i].x()) * K(0, 0);
            double dy = (P1_proj.y() - p1[i].y()) * K(1, 1);
          double projection_error1 = std::sqrt(dx * dx + dy * dy);

          if (projection_error1 > projection_error_thresh)
            {
                inliers[i] = false;
                nr_inliers--;
                continue;
            }

            V3d P2 = R * P1 + T;
            V3d P2_proj = P2 / P2.z();
            dx = (P2_proj.x() - p2[i].x()) * K(0, 0);
            dy = (P2_proj.y() - p2[i].y()) * K(1, 1);
          double projection_error2 = std::sqrt(dx * dx + dy * dy);

          if (projection_error2 > projection_error_thresh)
            {
                inliers[i] = false;
                nr_inliers--;
                continue;
            }

            points3d.push_back(P1);
        }
    }
}
