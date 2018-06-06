#include "viso.h"
#include "common.h"
#include <opencv2/core/eigen.hpp>


void Viso::OnNewFrame(Keyframe::Ptr current_frame) {
  const int nr_features = 1000;

  switch (state_) {
    case kInitialization:
      if (current_frame->GetId() != 0) {
        std::vector<bool> success;
        OpticalFlowMultiLevel(last_frame->Mat(),
                              current_frame->Mat(), last_frame->Keypoints(), current_frame->Keypoints(), success, true);

        std::vector<V3d> p1_success;
        std::vector<V3d> p2_success;

        for (int i = 0; i < current_frame->Keypoints().size(); i++) {
          if (success[i]) {
            p1_success.emplace_back(K_inv * V3d{last_frame->Keypoints()[i].pt.x, last_frame->Keypoints()[i].pt.y, 1});
            p2_success.emplace_back(
              K_inv * V3d{current_frame->Keypoints()[i].pt.x, current_frame->Keypoints()[i].pt.y, 1});
          }
        }

        PoseEstimation2d2d(p1_success, p2_success, current_frame->R(), current_frame->T());
        Reconstruct3DPoints(current_frame->R(), current_frame->T(), p1_success, p2_success, map_);

        state_ = kRunning;
      } else {
        cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(nr_features, 0.01, 20);
        detector->detect(current_frame->Mat(), current_frame->Keypoints());
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
    M3d &R, V3d &T)
{
    std::vector<cv::Point2f> kp1_;
    std::vector<cv::Point2f> kp2_;

  for (int i = 0; i < kp1.size(); ++i)
    {
      kp1_.emplace_back({(float) kp1[i].x(), (float) kp1[i].y()});
      kp2_.emplace_back({(float) kp2[i].x(), (float) kp2[i].y()});
    }

    // TODO: Remove this and implement own.
    cv::Mat essential = cv::findFundamentalMat(kp1_, kp2_);
    cv::Mat Rmat, tmat;
  recoverPose(essential, kp1_, kp2_, Rmat, tmat, 1, {0, 0});

    cv::cv2eigen(Rmat, R);
    cv::cv2eigen(tmat, T);
}

void Viso::Reconstruct3DPoints(const M3d &R, const V3d &T,
                               const std::vector<V3d> &points1, const std::vector<V3d> &points2,
                               std::vector<V3d> &points3d)
{
    int n = points1.size();

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3 * n, n + 1);
    for (int i = 0; i < n; ++i)
    {
        M.block<3, 1>(i * 3, i) = Hat(points2[i]) * R * points1[i];
        M.block<3, 1>(i * 3, n) = Hat(points2[i]) * T;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();

    points3d = points1;
    for (int i = 0; i < n; ++i)
    {
        points3d[i] *= V(i, n);
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
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size)
            { // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {

                    // TODO START YOUR CODE HERE (~8 lines)
                    double error = 0;
                    Eigen::Vector2d J; // Jacobian
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
                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update = H.inverse() * b;
            // TODO END YOUR CODE HERE

          if (std::isnan(update[0]))
            {
                // sometimes occurred when we have a black or white patch and H is irreversible
              std::cout << "update is nan" << std::endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost)
            {
                //cout << "cost increased: " << cost << ", " << lastCost << endl;
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
          kp2[i].pt = kp.pt + cv::Point2f((float) dx, (float) dy);
        }
        else
        {
          cv::KeyPoint tracked = kp;
          tracked.pt += cv::Point2f((float) dx, (float) dy);
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
    // TODO START YOUR CODE HERE (~8 lines)
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
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE

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

    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}