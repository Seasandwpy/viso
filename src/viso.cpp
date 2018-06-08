#include "viso.h"
#include "common.h"
#include "timer.h"
#include <opencv2/core/eigen.hpp>

void Viso::OnNewFrame(Keyframe::Ptr current_frame) {
  const int nr_features = 50;
  static int max_inliers = 0;
  const int reinitialize_after = 5;
  static int frame_cnt = 0;
  
  frame_cnt++;

  switch (state_) {
    case kInitialization:
      if (current_frame->GetId() != 0) {
        std::vector<bool> success;

        Timer timer;
        OpticalFlowMultiLevel(ref_frame->Mat(),
                              current_frame->Mat(), ref_frame->Keypoints(), current_frame->Keypoints(), success, true);

        timer.Reset();

        std::vector<V3d> p1;
        std::vector<V3d> p2;

        cv::Mat img;
        cv::cvtColor(current_frame->Mat(), img, CV_GRAY2BGR);

        for (int i = 0; i < current_frame->Keypoints().size(); i++) {
          if (success[i]) {
            p1.emplace_back(K_inv * V3d{ref_frame->Keypoints()[i].pt.x, ref_frame->Keypoints()[i].pt.y, 1});
            p2.emplace_back(
              K_inv * V3d{current_frame->Keypoints()[i].pt.x, current_frame->Keypoints()[i].pt.y, 1});
            cv::line(img, ref_frame->Keypoints()[i].pt, current_frame->Keypoints()[i].pt, cv::Scalar(255, 0, 0));
          }
        }

        cv::imshow("Optical flow", img);

        std::vector<bool> inliers;
        int nr_inliers = 0;
        PoseEstimation2d2d(p1, p2, current_frame->R(), current_frame->T(), inliers, nr_inliers);


        //std::cout << "PoseEstimation2d2d elapsed: " << timer.GetElapsed() << "\n";
        //timer.Reset();

	      
         Reconstruct(p1, p2, current_frame->R(), current_frame->T(), inliers, nr_inliers, map_);

        if(nr_inliers > max_inliers)
        {
          max_inliers = nr_inliers;
        }

        std::cout << "Tracked points: " << p1.size() << ", Inliers: " << nr_inliers << "(" << max_inliers << ")\n";
        
        cv::waitKey(10);
        const double thresh = 0.95;
        if (p1.size() > 300 && nr_inliers > 0 && (nr_inliers / (double) p1.size()) > thresh) {
          std::cout << "Initialized!\n";

          //Reconstruct3DPoints(current_frame->R(), current_frame->T(), p1, p2, map_, inliers, nr_inliers);
          //std::cout << "Reconstruct3DPoints elapsed: " << timer.GetElapsed() << "\n";
          timer.Reset();
          state_ = kRunning;

          //std::cout << "Map:\n";
          //for(int i = 0; i < map_.size(); ++i)
          //{         
          //  std::cout << map_[i] << "\n";
          //}

        }

        if(frame_cnt > reinitialize_after) {
          cv::FAST(current_frame->Mat(), current_frame->Keypoints(), nr_features);
          std::cout << "Nr keypoints: " << current_frame->Keypoints().size() << "\n";

          ref_frame = current_frame;

          frame_cnt = 0;
        }
      }
      else
      {
        //cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(nr_features);
        //detector->detect(current_frame->Mat(), current_frame->Keypoints());
        cv::FAST(current_frame->Mat(), current_frame->Keypoints(), nr_features);
        std::cout << "Nr keypoints: " << current_frame->Keypoints().size() << "\n";

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
      kp1_.push_back({(float) kp1[i].x(), (float) kp1[i].y()});
      kp2_.push_back({(float) kp2[i].x(), (float) kp2[i].y()});
    }

  Timer timer;

  cv::Mat outlier_mask;

#if 1

  double thresh = 1/std::sqrt(K(0,0)*K(0,0) + K(1,1)*K(1,1));
  cv::Mat essential = cv::findFundamentalMat(kp1_, kp2_, CV_FM_RANSAC, thresh, 0.99, outlier_mask);

  //std::cout << "findFundamentalMat elapsed " << timer.GetElapsed() << "\n";
  timer.Reset();

    cv::Mat Rmat, tmat;

  // This method does the depth check. Only users points which are not masked out by
  // the outlier mask.
  recoverPose(essential, kp1_, kp2_, Rmat, tmat, 1.0, {}, outlier_mask);

  cv::cv2eigen(Rmat, R);
  cv::cv2eigen(tmat, T);
#endif

#if 0
  cv::Mat H = cv::findHomography(kp1_, kp2_, CV_RANSAC, 1.0, outlier_mask, 2000, 0.99);
  RecoverPoseHomography(H, R, T);
#endif

  //std::cout << "recoverPose elapsed " << timer.GetElapsed() << "\n";
  timer.Reset();

  inliers = std::vector<bool>(kp1.size());
  for (int i = 0; i < inliers.size(); ++i) {
    inliers[i] = outlier_mask.at<bool>(i) > 0;
    nr_inliers += inliers[i];
  }

  std::cout << "Essential matrix inliers: " << nr_inliers << "\n";
}

void Viso::Reconstruct3DPoints(const M3d &R, const V3d &T,
                               const std::vector<V3d> &points1, const std::vector<V3d> &points2,
                               std::vector<V3d> &points3d, const std::vector<bool> &inliers, const int &nr_inliers) {
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3 * nr_inliers, nr_inliers + 1);
  int j;
  for (int i = 0, j = 0; i < points1.size(); ++i)
    {
      if (inliers[i]) {
        M.block<3, 1>(j * 3, j) = Hat(points2[i]) * R * points1[i];
        M.block<3, 1>(j * 3, nr_inliers) = Hat(points2[i]) * T;
        ++j;
      }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();

  points3d.clear();

  for (int i = 0, j = 0; i < points1.size(); ++i)
    {
      if (inliers[i]) {
        points3d.push_back(points1[i] / V(j, nr_inliers));
        ++j;
      }
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
  int iterations = 100;
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
  bool inverse) {

#if 1
  std::vector<uchar> status;
  std::vector<float> error;
  std::vector<cv::Point2f> pt1, pt2;
  for (auto &kp : kp1)
      pt1.push_back(kp.pt);

  cv::calcOpticalFlowPyrLK(img1, img2,
                           pt1, pt2,
                           status, error,
                           cv::Size2i(8, 8));
  for (int i = 0; i < status.size(); ++i)
  {
      success.push_back(status[i] > 0);
  }

  for (auto &p : pt2)
      kp2.push_back(cv::KeyPoint(p, 0));
#endif

#if 0
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

        Timer timer;

      std::vector<bool> success_single;
        OpticalFlowSingleLevel(pyr1[i], pyr2[i], kp1_, kp2, success_single, true);

      //std::cout << "OpticalFlowSingleLevel elapsed " << timer.GetElapsed() << "\n";
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
#endif
    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
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
void Viso::Triangulate(const M34d &Pi1, const M34d &Pi2, const V3d &x1, const V3d &x2, V3d &P) {
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

void Viso::Reconstruct(const std::vector<V3d>& p1,
  const std::vector<V3d>& p2,
  const M3d &R, const V3d &T, std::vector<bool> &inliers,   int &nr_inliers, std::vector<V3d>& points3d){

  M34d Pi1 = MakePI0();
  M34d Pi2 = MakePI0() * MakeSE3(R, T);

  points3d.clear();

  int j = 0;
  for(int i = 0; i < p1.size(); ++i)
  {         
    if(inliers[i]) 
    {
      V3d P1;
      Triangulate(Pi1, Pi2, p1[i], p2[i], P1);
      V3d P1_proj = P1 / P1.z();
      double dx = (P1_proj.x() - p1[i].x()) * K(0,0);
      double dy = (P1_proj.y() - p1[i].y()) * K(1,1);
      double projection_error = std::sqrt(dx * dx + dy * dy);

      if(projection_error > 1.) {
        inliers[i] = false;
        nr_inliers--;
        continue;
      }            

//      std::cout << "-----------------------------------------------\n";
//      std::cout << P1 << "\n";
//      std::cout << "projection error: " << projection_error << "\n";

      V3d P2 = R*P1 + T;
      V3d P2_proj = P2 / P2.z();
      dx = (P2_proj.x() - p2[i].x()) * K(0,0);
      dy = (P2_proj.y() - p2[i].y()) * K(1,1);
      projection_error = std::sqrt(dx * dx + dy * dy);            
//      std::cout << P2 << "\n";
//      std::cout << "projection error: " << projection_error << "\n";

      if(projection_error > 1.) {
        inliers[i] = false;
        nr_inliers--;
        continue;
      }          

      points3d.push_back(P1);      
    }       
  }
}

void Viso::RecoverPoseHomography(const cv::Mat& H, M3d& R, V3d& T)
{
    cv::Mat pose = cv::Mat::eye(3, 4, CV_64FC1); //3x4 matrix
    float norm1 = (float)norm(H.col(0)); 
    float norm2 = (float)norm(H.col(1));
    float tnorm = (norm1 + norm2) / 2.0f;

    cv::Mat v1 = H.col(0);
    cv::Mat v2 = pose.col(0);

    cv::normalize(v1, v2); // Normalize the rotation

    v1 = H.col(1);
    v2 = pose.col(1);

    cv::normalize(v1, v2);

    v1 = pose.col(0);
    v2 = pose.col(1);

    cv::Mat v3 = v1.cross(v2);  //Computes the cross-product of v1 and v2
    cv::Mat c2 = pose.col(2);
    v3.copyTo(c2);      

    pose.col(3) = H.col(2) / tnorm; //vector t [R|t]

    cv::cv2eigen(pose.col(3), T);
    cv::cv2eigen(pose(cv::Rect(0, 0, 3, 3)), R);
}




