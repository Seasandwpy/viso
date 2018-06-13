#include "viso.h"
#include "common.h"
#include "timer.h"


#include <opencv2/core/eigen.hpp>

void Viso::OnNewFrame(Keyframe::Ptr cur_frame)
{
    // TODO: Clean this up.
    cur_frame->SetK(K);

    switch (state_) {
    case kInitialization:
        if (init_.frame_cnt > 0 && init_.frame_cnt <= reinitialize_after) {

            OpticalFlowMultiLevel(init_.ref_frame,
                cur_frame,
                init_.kp1,
                init_.kp2,
                init_.success,
                true);

            {
                auto iter1 = init_.kp1.begin();
                auto iter2 = init_.kp2.begin();
                auto iter3 = init_.success.begin();

                while (iter1 != init_.kp1.end()) {
                    if (!(*iter3)) {
                        iter1 = init_.kp1.erase(iter1);
                        iter2 = init_.kp2.erase(iter2);
                    } else {
                        ++iter1;
                        ++iter2;
                    }
                    ++iter3;
                }

                init_.success.clear();
            }

            std::vector<V3d> p1;
            std::vector<V3d> p2;

            for (int i = 0; i < init_.kp1.size(); ++i) {
                p1.emplace_back(K_inv * V3d{ init_.kp1[i].pt.x, init_.kp1[i].pt.y, 1 });
                p2.emplace_back(K_inv * V3d{ init_.kp2[i].pt.x, init_.kp2[i].pt.y, 1 });
            }

            int nr_inliers = 0;
            std::vector<V3d> points3d;
            PoseEstimation2d2d(p1, p2, init_.R, init_.T, init_.success, nr_inliers);
            Reconstruct(p1, p2, init_.R, init_.T, init_.success, nr_inliers, points3d);

            // Visualization
            cv::Mat img;
            cv::cvtColor(cur_frame->Mat(), img, CV_GRAY2BGR);

            for (int i = 0; i < p2.size(); i++) {
                cv::Scalar color(255, 0, 0);
                if (nr_inliers > 0 && init_.success[i]) {
                    color = cv::Scalar(0, 255, 0);
                }
                cv::Point2f kp1 = init_.kp1[i].pt;
                V3d kp2 = K * p2[i];
                cv::line(img, kp1, { (int)kp2.x(), (int)kp2.y() }, color);
            }

            cv::imshow("Optical flow", img);
            //

            std::cout << "Tracked points: " << p1.size()
                      << ", Inliers: " << nr_inliers << "\n";

            cv::waitKey(10);
            const double thresh = 0.9;
            if (p1.size() > 50 && nr_inliers > 0 && (nr_inliers / (double)p1.size()) > thresh) {
                std::cout << "Initialized!\n";
                map_.AddKeyframe(init_.ref_frame);
                map_.AddKeyframe(cur_frame);

                cur_frame->SetR(init_.R);
                cur_frame->SetT(init_.T);
                int cnt = 0;
                for (int i = 0; i < p1.size(); ++i) {
                    if (init_.success[i]) {
                        init_.ref_frame->AddKeypoint(init_.kp1[i]);
                        cur_frame->AddKeypoint(init_.kp2[i]);
                        MapPoint::Ptr map_point = std::make_shared<MapPoint>(points3d[cnt]);
                        map_point->AddObservation(init_.ref_frame, i);
                        map_point->AddObservation(cur_frame, i);
                        map_.AddPoint(map_point);
                        ++cnt;
                    }
                }
                Sophus::SE3d X = Sophus::SE3d(cur_frame->GetR(), cur_frame->GetT());
                poses.push_back(X);
                BA();
                state_ = kRunning;
                std::cout << "start tracking" << std::endl;
                break;
            }
            else{
            	poses.push_back(Sophus::SE3d(M3d::Identity(),V3d::Zero()));
            }
        } else {
            init_.kp1.clear();
            init_.kp2.clear();
            poses.clear();
            init_.success.clear();
            cv::FAST(cur_frame->Mat(), init_.kp1, fast_thresh);
            init_.kp2 = init_.kp1;
            init_.ref_frame = cur_frame;
            init_.frame_cnt = 0;
            poses.push_back(Sophus::SE3d(M3d::Identity(),V3d::Zero()));
        }

        ++init_.frame_cnt;
        break;

    case kRunning: {
        Sophus::SE3d X = Sophus::SE3d(last_frame->GetR(), last_frame->GetT());
        DirectPoseEstimationMultiLayer(cur_frame, X);

        cur_frame->SetR(X.rotationMatrix());
        cur_frame->SetT(X.translation());

        std::vector<V2d> kp_before, kp_after;
        LKAlignment(cur_frame, kp_before, kp_after);

        //BA(map_->)

        cv::Mat display;
        cv::cvtColor(cur_frame->Mat(), display, CV_GRAY2BGR);

        for (int i = 0; i < kp_after.size(); ++i) {
            cv::rectangle(display, cv::Point2f(kp_before[i].x() - 4, kp_before[i].y() - 4), cv::Point2f(kp_before[i].x() + 4, kp_before[i].y() + 4),
                cv::Scalar(0, 0, 255));

            cv::rectangle(display, cv::Point2f(kp_after[i].x() - 4, kp_after[i].y() - 4), cv::Point2f(kp_after[i].x() + 4, kp_after[i].y() + 4),
                cv::Scalar(0, 250, 0));
        }

        cv::imshow("Tracked", display);
        cv::waitKey(0);

        poses.push_back(X);
    } break;

    default:
        break;
    }

    last_frame = cur_frame;
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
        OpticalFlowSingleLevel(ref_frame->Pyramids()[i], cur_frame->Pyramids()[i], kp1_, kp2, success_single, inverse);
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

        current_frame->SetR(T21.rotationMatrix());
        current_frame->SetT(T21.translation());

        for (size_t i = 0; i < map_.GetPoints().size(); i++) {

            V3d P1 = map_.GetPoints()[i]->GetWorldPos();

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

        V3d Pw = map_.GetPoints()[i]->GetWorldPos();

        if (!current_frame->IsInside(Pw, /*level=*/0)) {
            continue;
        }

        // Find frame with best viewing angle.
        double best_angle = 180.0;
        int best_frame_idx = -1;
        V2d best_uv_ref;

        auto keyframes = map_.Keyframes();
        for (int j = 0; j < keyframes.size(); ++j) {
            Keyframe::Ptr frame = keyframes[j];
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
        pair.ref_frame = keyframes[best_frame_idx];
        pair.cur_frame = current_frame;
        pair.uv_ref = best_uv_ref;
        pair.uv_cur = current_frame->Project(Pw, /*level=*/0);

        alignment_pairs.push_back(pair);
    }

    std::vector<bool> success;

    const int nr_pyramids = 4;

    for (int i = 0; i < alignment_pairs.size(); ++i) {
        kp_before.push_back(alignment_pairs[i].uv_cur);
    }

    for (int level = nr_pyramids - 1; level >= 0; --level) {
        LKAlignmentSingle(alignment_pairs, success, kp_after, level);
    }

    assert(success.size() == kp_before.size());

    int i = 0;
    for (auto iter = kp_before.begin(); iter != kp_before.end(); ++i) {
        if (!success[i]) {
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
    const double cost_thresh = (half_patch_size * 2) * (half_patch_size * 2) * 100.0;

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
                    if (!inverse) {
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

        if (lastCost > cost_thresh) {
            succ = false;
        }

        success.push_back(succ);
        pair.uv_cur += V2d{ dx / scales[level], dy / scales[level] };
    }

    for (int i = 0; i < pairs.size(); ++i) {
        if (success[i]) {
            kp.push_back(pairs[i].uv_cur);
        }
    }
}

void Viso::BA(){
	std::cout << "start BA" << std::endl;
	// build optimization problem
  	// setup g2o
  	typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;  // pose is 6x1, landmark is 3x1
  	std::unique_ptr<Block::LinearSolverType> linearSolver(
      	new g2o::LinearSolverDense<Block::PoseMatrixType>()); // linear solver

  	// use levernberg-marquardt here (or you can choose gauss-newton)
  	g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
      	g2o::make_unique<Block>(std::move(linearSolver)));
  	g2o::SparseOptimizer optimizer;     // graph optimizer
  	optimizer.setAlgorithm(solver);   // solver
  	optimizer.setVerbose(true);       // open the output
  
  	
  	std::vector<VertexSBAPointXYZ*> points_v;
  	std::vector<VertexSophus*> cameras_v;
  	int id=0;

	for (size_t i = 0; i < map_.GetPoints().size(); i++, id++) { //for each mappoint
		MapPoint::Ptr mp = map_.GetPoints()[i];
        //V3d mpxyz = mp->GetWorldPos();
        VertexSBAPointXYZ* p = new VertexSBAPointXYZ;
      	p->setId(id);
      	p->setMarginalized(true);
      	p->setEstimate(mp->GetWorldPos());
      	std::cout << mp->GetWorldPos() << std::endl;
      	optimizer.addVertex(p);
      	points_v.push_back(p);
        /*for(size_t j = 0; j < mp->GetObservations().size(); j++; id++){ //for each observation
        	std::vector<std::pair<Keyframe::Ptr, int> > obs = mp->GetObservations()[j];

    	}*/
    }

    //std::cout << "pose size: " << poses.size() << std::endl;
    for (size_t i = 0; i < poses.size(); i++, id++) {
    	VertexSophus* cam = new VertexSophus;
    	Sophus::SE3d pose = poses[i];
    	//std::cout << pose.matrix() << std::endl;
    	cam->setId(id);
    	if(i == 0) cam->setFixed( true ); //fix the pose of the first frame
      	cam->setEstimate(pose);
      	optimizer.addVertex(cam);
      	cameras_v.push_back(cam);
    }
    //id=0;
    //std::cout << "add edge"<< std::endl;
    for (size_t i = 0; i < map_.GetPoints().size(); i++) {
    	MapPoint::Ptr mp = map_.GetPoints()[i];
    	for(size_t j = 0; j < mp->GetObservations().size(); j++, id++){ //for each observation
        	std::pair<Keyframe::Ptr, int>  obs = mp->GetObservations()[j];
        	EdgeObservation* e = new EdgeObservation(K);
        	e ->setVertex(0,points_v[i]);
        	e ->setVertex(1,cameras_v[obs.first->GetId()]);
        	e->setInformation(Eigen::Matrix2d::Identity()); //intensity is a scale?
        	int idx = obs.second;
        	V2d xy (obs.first->Keypoints()[idx].pt.x,obs.first->Keypoints()[idx].pt.y);
        	//std::cout << xy.transpose() << std::endl;
        	e->setMeasurement(xy);
        	e->setId( id );
        	optimizer.addEdge(e);
    	}
	}


	// perform optimization
  	std::cout << "optimize!"<< std::endl;
  	optimizer.initializeOptimization(0);
  	optimizer.optimize(BA_iteration);
	std::cout << "end!"<< std::endl;
  	
  	for ( int i=0; i < poses.size(); i++ )
  	{
     	VertexSophus* pose = dynamic_cast<VertexSophus*> (optimizer.vertex(map_.GetPoints().size()+i));
     	Sophus::SE3d p_opt = pose->estimate();
     	poses_opt.push_back(p_opt);
  	}
  	/*for ( int i=0; i < map_.GetPoints().size(); i++ )
  	{
     	VertexPoint* point = dynamic_cast<VertexPoint*> (optimizer.vertex(i));
     	V3d point_opt = g->estimate();
     	map_.GetPoints()[i]->GetWorldPos()=point_opt;
  	}*/
}
