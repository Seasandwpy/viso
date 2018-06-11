#include "viso.h"
#include "common.h"
#include "timer.h"

#include <opencv2/core/eigen.hpp>

void Viso::OnNewFrame(Keyframe::Ptr current_frame)
{
    static int max_inliers = 0;
    static int frame_cnt = 0;

    frame_cnt++;
    // TODO: Clean this up.
    current_frame->K() = K;

    switch (state_) {
    case kInitialization:
        if (current_frame->GetId() != 0) {
            std::vector<bool> success;
            current_frame->Keypoints() = last_frame->Keypoints();
            OpticalFlowMultiLevel(ref_frame, current_frame,
                ref_frame->Keypoints(), current_frame->Keypoints(),
                success, true);

            std::vector<V3d> p1;
            std::vector<V3d> p2;

            std::vector<int> kp_indices;

            for (int i = 0; i < current_frame->Keypoints().size(); i++) {
                if (success[i]) {
                    p1.emplace_back(K_inv * V3d{ ref_frame->Keypoints()[i].pt.x,
                                                ref_frame->Keypoints()[i].pt.y, 1 });
                    p2.emplace_back(K_inv * V3d{ current_frame->Keypoints()[i].pt.x,
                                                current_frame->Keypoints()[i].pt.y, 1 });
                    kp_indices.push_back(i);
                }
            }

            std::vector<bool> inliers;
            int nr_inliers = 0;
            std::vector<V3d> points3d;
            PoseEstimation2d2d(p1, p2, current_frame->R(), current_frame->T(),
                inliers, nr_inliers);
            Reconstruct(p1, p2, current_frame->R(), current_frame->T(), inliers,
                nr_inliers, points3d);

            if (nr_inliers > max_inliers) {
                max_inliers = nr_inliers;
            }

            cv::Mat img;
            cv::cvtColor(current_frame->Mat(), img, CV_GRAY2BGR);

            for (int i = 0; i < p2.size(); i++) {
                // if (inliers[i])
                //{
                cv::Scalar color(255, 0, 0);
                if (nr_inliers > 0 && inliers[i]) {
                    color = cv::Scalar(0, 255, 0);
                }
                cv::Point2f kp1 = ref_frame->Keypoints()[kp_indices[i]].pt;
                V3d kp2 = K * p2[i];
                cv::line(img, kp1, { (int)kp2.x(), (int)kp2.y() }, color);
                //}
            }

            cv::imshow("Optical flow", img);
            std::cout << "Tracked points: " << p1.size()
                      << ", Inliers: " << nr_inliers << "(" << max_inliers << ")\n";

            cv::waitKey(10);
            const double thresh = 0.9;
            if (p1.size() > 50 && nr_inliers > 0 && (nr_inliers / (double)p1.size()) > thresh) {
                std::cout << "Initialized!\n";
                map_.AddKeyframe(ref_frame);
                last_frames_.Push(ref_frame);
                last_frames_.Push(current_frame);

                int cnt = 0;
                for (int i = 0; i < p1.size(); ++i) {
                    if (inliers[i]) {
                        cv::Point2f kp1 = ref_frame->Keypoints()[kp_indices[i]].pt;
                        map_.AddPoint(MapPoint(ref_frame, { kp1.x, kp1.y }, points3d[cnt]));
                        ++cnt;
                    }
                }
                state_ = kRunning;
                break;
            }

            if (frame_cnt > reinitialize_after) {
                cv::FAST(current_frame->Mat(), current_frame->Keypoints(), fast_thresh);
                ref_frame = current_frame;
                frame_cnt = 0;
            }
        } else {
            cv::FAST(current_frame->Mat(), current_frame->Keypoints(), fast_thresh);
            ref_frame = current_frame;
        }
        break;

    case kRunning: {
        M3d R = last_frame->R();
        V3d T = last_frame->T();

        Sophus::SE3d X = Sophus::SE3d(R, T);
        DirectPoseEstimationMultiLayer(current_frame, X);

        current_frame->R() = X.rotationMatrix();
        current_frame->T() = X.translation();

        std::vector<V2d> kp_before, kp_after;
        LKAlignment(current_frame, kp_before, kp_after);

        cv::Mat display;
        cv::cvtColor(current_frame->Mat(), display, CV_GRAY2BGR);

        for (int i = 0; i < kp_after.size(); ++i) {
            cv::rectangle(display, cv::Point2f(kp_before[i].x() - 4, kp_before[i].y() - 4), cv::Point2f(kp_before[i].x() + 4, kp_before[i].y() + 2),
                cv::Scalar(0, 0, 255));

            cv::rectangle(display, cv::Point2f(kp_after[i].x() - 4, kp_after[i].y() - 4), cv::Point2f(kp_after[i].x() + 4   , kp_after[i].y() + 2),
                cv::Scalar(0, 250, 0));
        }

        cv::imshow("Tracked", display);
        cv::waitKey(0);

        poses.push_back(X);

        last_frames_.Push(current_frame);
    } break;

    default:
        break;
    }

    last_frame = current_frame;
}

// TODO: Move this to a separate class.
void Viso::PoseEstimation2d2d(std::vector<V3d> kp1, std::vector<V3d> kp2,
    M3d& R, V3d& T, std::vector<bool>& inliers,
    int& nr_inliers)
{
    std::vector<cv::Point2f> kp1_;
    std::vector<cv::Point2f> kp2_;

    for (int i = 0; i < kp1.size(); ++i) {
        kp1_.push_back({ (float)kp1[i].x(), (float)kp1[i].y() });
        kp2_.push_back({ (float)kp2[i].x(), (float)kp2[i].y() });
    }

    cv::Mat outlier_mask;

    double thresh = projection_error_thresh / std::sqrt(K(0, 0) * K(0, 0) + K(1, 1) * K(1, 1));
    cv::Mat essential = cv::findEssentialMat(
        kp1_, kp2_, 1.0, { 0.0, 0.0 }, CV_FM_RANSAC, 0.99, thresh, outlier_mask);

    if (essential.data == NULL) {
        nr_inliers = 0;
        return;
    }

    cv::Mat Rmat, tmat;

    // This method does the depth check. Only users points which are not masked
    // out by
    // the outlier mask.
    recoverPose(essential, kp1_, kp2_, Rmat, tmat, 1.0, {}, outlier_mask);

    cv::cv2eigen(Rmat, R);
    cv::cv2eigen(tmat, T);

    inliers = std::vector<bool>(kp1.size());
    for (int i = 0; i < inliers.size(); ++i) {
        inliers[i] = outlier_mask.at<bool>(i) > 0;
        nr_inliers += inliers[i];
    }
}

// TODO: Move this to a separate class.
void Viso::OpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, bool inverse)
{

    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size || kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    double error = 0;
                    V2d J; // Jacobian
                    if (!inverse) {
                        // Forward Jacobian
                        J = -GetImageGradient(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    } else {
                        // Inverse Jacobian
                        J = -GetImageGradient(img1, kp.pt.x + x, kp.pt.y + y);
                    }

                    error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);

                    // compute H, b and set cost;
                    H += J * J.transpose();
                    b += -J * error;
                    cost += error * error;
                }

            Eigen::Vector2d update = H.inverse() * b;

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is
                // irreversible
                std::cout << "update is nan" << std::endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
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
        if (have_initial) {
            kp2[i].pt = kp.pt + cv::Point2f((float)dx, (float)dy);
        } else {
            cv::KeyPoint tracked = kp;
            tracked.pt += cv::Point2f((float)dx, (float)dy);
            kp2.push_back(tracked);
        }
    }
}

// TODO: Move this to a separate class.
void Viso::OpticalFlowMultiLevel(
    const Keyframe::Ptr ref_frame,
    const Keyframe::Ptr cur_frame,
    const std::vector<cv::KeyPoint>& kp1,
    std::vector<cv::KeyPoint>& kp2,
    std::vector<bool>& success, bool inverse)
{
    // parameters
    // TODO: Move these parameters to a common place.
    const int nr_pyramids = 4;
    const double pyramid_scale = 0.5;
    const double scales[] = { 1.0, 0.5, 0.25, 0.125 };

    // Scale the initial guess for kp2.
    for (int j = 0; j < kp2.size(); ++j) {
        kp2[j].pt *= scales[nr_pyramids - 1];
        kp2[j].size *= scales[nr_pyramids - 1];
    }

    for (int i = nr_pyramids - 1; i >= 0; --i) {
        std::vector<cv::KeyPoint> kp1_ = kp1;

        for (int j = 0; j < kp1_.size(); ++j) {
            kp1_[j].pt *= scales[i];
            kp1_[j].size *= scales[i];
        }

        std::vector<bool> success_single;
        OpticalFlowSingleLevel(ref_frame->Pyramids()[i], cur_frame->Pyramids()[i], kp1_, kp2, success_single, true);
        success = success_single;

        if (i != 0) {
            for (int j = 0; j < kp1_.size(); ++j) {
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

// TODO: Move this to a separate class.
void Viso::Triangulate(const M34d& Pi1, const M34d& Pi2, const V3d& x1,
    const V3d& x2, V3d& P)
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

// TODO: Move this to a separate class.
void Viso::Reconstruct(const std::vector<V3d>& p1, const std::vector<V3d>& p2,
    const M3d& R, const V3d& T, std::vector<bool>& inliers,
    int& nr_inliers, std::vector<V3d>& points3d)
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
    for (int i = 0; i < p1.size(); ++i) {
        if (inliers[i]) {
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

            if (projection_error1 > projection_error_thresh) {
                inliers[i] = false;
                nr_inliers--;
                continue;
            }

            V3d P2 = R * P1 + T;
            V3d P2_proj = P2 / P2.z();
            dx = (P2_proj.x() - p2[i].x()) * K(0, 0);
            dy = (P2_proj.y() - p2[i].y()) * K(1, 1);
            double projection_error2 = std::sqrt(dx * dx + dy * dy);

            if (projection_error2 > projection_error_thresh) {
                inliers[i] = false;
                nr_inliers--;
                continue;
            }

            points3d.push_back(P1);
        }
    }
}

M26d dPixeldXi(const M3d& K, const M3d& R, const V3d& T, const V3d& P,
    const double& scale)
{
    V3d Pc = R * P + T;
    double x = Pc.x();
    double y = Pc.y();
    double z = Pc.z();
    double fx = K(0, 0) * scale;
    double fy = K(1, 1) * scale;
    double zz = z * z;
    double xy = x * y;

    M26d result;
    result << fx / z, 0, -fx * x / zz, -fx * xy / zz, fx + fx * x * x / zz,
        -fx * y / z, 0, fy / z, -fy * y / zz, -fy - fy * y * y / zz, fy * xy / zz,
        fy * x / z;

    return result;
}

// TODO: Move this to a separate class.
void Viso::DirectPoseEstimationSingleLayer(int level,
    Keyframe::Ptr current_frame,
    Sophus::SE3d& T21)
{

    const double scales[] = { 1, 0.5, 0.25, 0.125 };
    const double scale = scales[level];
    const double delta_thresh = 0.005;

    // parameters
    int half_patch_size = 4;
    int iterations = 100;

    double cost = 0, lastCost = 0;
    int nGood = 0; // good projections

    Sophus::SE3d best_T21 = T21;

    for (int iter = 0; iter < iterations; iter++) {
        nGood = 0;

        // Define Hessian and bias
        M6d H = M6d::Zero(); // 6x6 Hessian
        V6d b = V6d::Zero(); // 6x1 bias

        current_frame->R() = T21.rotationMatrix();
        current_frame->T() = T21.translation();

        for (size_t i = 0; i < map_.GetPoints().size(); i++) {

            V3d P1 = map_.GetPoints()[i].world_pos;

            Keyframe::Ptr frame = last_frame;
            V2d uv_ref = frame->Project(P1, level);
            double u_ref = uv_ref.x();
            double v_ref = uv_ref.y();

            V2d uv_cur = current_frame->Project(P1, level);
            double u_cur = uv_cur.x();
            double v_cur = uv_cur.y();

            bool hasNaN = uv_cur.array().hasNaN() || uv_ref.array().hasNaN();
            assert(!hasNaN);

            bool good = frame->IsInside(u_ref - half_patch_size,
                            v_ref - half_patch_size, level)
                && frame->IsInside(u_ref + half_patch_size,
                       v_ref + half_patch_size, level)
                && current_frame->IsInside(u_cur - half_patch_size,
                       v_cur - half_patch_size, level)
                && current_frame->IsInside(u_cur + half_patch_size,
                       v_cur + half_patch_size, level);

            if (!good) {
                continue;
            }

            nGood++;

            M26d J_pixel_xi = dPixeldXi(K, T21.rotationMatrix(), T21.translation(),
                P1, scale); // pixel to \xi in Lie algebra

            for (int x = -half_patch_size; x < half_patch_size; x++) {
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = frame->GetPixelValue(u_ref + x, v_ref + y, level) - current_frame->GetPixelValue(u_cur + x, v_cur + y, level);
                    V2d J_img_pixel = current_frame->GetGradient(u_cur + x, v_cur + y, level);
                    V6d J = -J_img_pixel.transpose() * J_pixel_xi;
                    H += J * J.transpose();
                    b += -error * J;  
                    cost += error * error;
                }
            }
        }

        // solve update and put it into estimation
        V6d update = H.inverse() * b;

        T21 = Sophus::SE3d::exp(update) * T21;

        cost /= nGood;

        if (std::isnan(update[0])) {
            T21 = best_T21;
            break;
        }

        if (iter > 0 && cost > lastCost) {
            T21 = best_T21;
            break;
        }

        if ((1 - cost / (double)lastCost) < delta_thresh) {
            break;
        }

        best_T21 = T21;
        lastCost = cost;
    }
}

void Viso::DirectPoseEstimationMultiLayer(Keyframe::Ptr current_frame,
    Sophus::SE3d& T21)
{
    for (int level = 3; level >= 0; level--) {
        DirectPoseEstimationSingleLayer(level, current_frame, T21);
    }
}

void Viso::LKAlignment(Keyframe::Ptr current_frame, std::vector<V2d>& kp_before, std::vector<V2d>& kp_after)
{
    std::vector<AlignmentPair> alignment_pairs;

    const double max_angle = 180.0; // 180 means basically no restriction on the angle (for now)

    for (size_t i = 0; i < map_.GetPoints().size(); i++) {

        V3d Pw = map_.GetPoints()[i].world_pos;

        if (!current_frame->IsInside(Pw, /*level=*/0)) {
            continue;
        }

        // Find frame with best viewing angle.
        double best_angle = 180.0;
        int best_frame_idx = -1;
        V2d best_uv_ref;

        for (int j = 0; j < last_frames_.GetSize(); ++j) {
            Keyframe::Ptr frame = last_frames_[j];
            V2d uv_ref = frame->Project(Pw, /*level=*/0);
            double u_ref = uv_ref.x();
            double v_ref = uv_ref.y();

            if (!frame->IsInside(u_ref, v_ref)) {
                continue;
            }

            double angle = std::abs(frame->ViewingAngle(Pw) / CV_PI * 180);
            if (angle > max_angle || angle > best_angle) {
                continue;
            }

            best_angle = angle;
            best_frame_idx = j;
            best_uv_ref = uv_ref;
        }

        if (best_frame_idx == -1) {
            continue;
        }

        AlignmentPair pair;
        pair.ref_frame = last_frames_[best_frame_idx];
        pair.cur_frame = current_frame;
        pair.uv_ref = best_uv_ref;
        pair.uv_cur = current_frame->Project(Pw, /*level=*/0);

        alignment_pairs.push_back(pair);
    }

    std::vector<bool> success;

    const int nr_pyramids = 4;
    
    for(int i = 0; i < alignment_pairs.size(); ++i) {
      kp_before.push_back(alignment_pairs[i].uv_cur);
    }

    for (int level = nr_pyramids - 1; level >= 0; --level) {
        LKAlignmentSingle(alignment_pairs, success, kp_after, level);
    }

    assert(success.size() == kp_before.size());

    int i = 0;
    for(auto iter = kp_before.begin(); iter != kp_before.end(); ++i) {      
      if(!success[i]) {
        iter = kp_before.erase(iter);
      } else {
        ++iter;      
      }
    }
}

void Viso::LKAlignmentSingle(std::vector<AlignmentPair>& pairs, std::vector<bool>& success, std::vector<V2d>& kp, int level)
{
    // parameters
    const bool inverse = true;
    const int half_patch_size = 4;
    int iterations = 100;
    const double scales[4] = { 1.0, 0.5, 0.25, 0.125 };
    const double cost_thresh = (half_patch_size * 2) * (half_patch_size * 2) * 255.0;  

    success.clear();
    kp.clear();

    for (size_t i = 0; i < pairs.size(); i++) {
        AlignmentPair& pair = pairs[i];

        double dx = 0, dy = 0; // dx,dy need to be estimated

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (
                !pair.ref_frame->IsInside(pair.uv_ref.x() * scales[level] + dx - half_patch_size, pair.uv_ref.y() * scales[level] + dy - half_patch_size, level) || !pair.ref_frame->IsInside(pair.uv_ref.x() * scales[level] + dx + half_patch_size, pair.uv_ref.y() * scales[level] + dy + half_patch_size, level)) {
                succ = false;
                break;
            }

            double error = 0;
            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++) {
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    V2d J;
                    if(!inverse) {
                    J = -pair.cur_frame->GetGradient(pair.uv_cur.x() * scales[level] + x + dx, pair.uv_cur.y() * scales[level] + y + dy, level);
                    } else {
                    J = -pair.ref_frame->GetGradient(pair.uv_ref.x() * scales[level] + x, pair.uv_ref.y() * scales[level] + y, level);
                    }                    
error = pair.ref_frame->GetPixelValue(pair.uv_ref.x() * scales[level] + x, pair.uv_ref.y() * scales[level] + y, level) - pair.cur_frame->GetPixelValue(pair.uv_cur.x() * scales[level] + x + dx, pair.uv_cur.y() * scales[level] + y + dy, level);

                    // compute H, b and set cost;
                    H += J * J.transpose();
                    b += -J * error;
                    cost += error * error;
                }
            }

            V2d update = H.inverse() * b;

            if (std::isnan(update[0])) {
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        if(lastCost > cost_thresh) {
          succ = false;        
        }

        success.push_back(succ);
        pair.uv_cur += V2d{ dx / scales[level], dy / scales[level] } ;
    }

    for (int i = 0; i < pairs.size(); ++i) {
        if (success[i]) {
            kp.push_back(pairs[i].uv_cur);
        }
    }
}
