//
// Created by Xiang on 2017/12/19.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "types.h"
#include "common.h"
#include <opencv2/core/eigen.hpp>
#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;

string file_1 = "./LK1.png"; // first image
string file_2 = "./LK2.png"; // second image

// TODO implement this funciton
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false);

// TODO implement this funciton
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false);

// 2D-2D pose estimation functions
void poseEstimation2d2d(
    std::vector<cv::KeyPoint> keypoints_1,
    std::vector<cv::KeyPoint> keypoints_2,
    Eigen::Matrix3d &R21, Eigen::Vector3d &t21,
    double fx, double fy, double cx, double cy);

void Draw(const vector<V3d> &points);

void reconstructPoints(double &scale,
                       M3d R, V3d t,
                       const std::vector<V3d> &points1, const std::vector<V3d> &points2,
                       std::vector<V3d> &points3d);

class MyFrameSequence : public FrameSequence
{
  public:
    MyFrameSequence() : FrameSequence("", 1, 200.0, 200.0, 240.0, 240.0)
    {
        last_R = M3d::Identity();
        last_T = V3d::Zero();
        scale = 0;
    }
    virtual void OnNewFrame(cv::Mat frame){

    };

    int main(int argc, char **argv)
    {
        MyFrameSequence sequence;

        sequence.RunOne();

        Draw(sequence.all_points3d);
        return 1;
    }

