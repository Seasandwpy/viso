//
// Created by sage on 04.06.18.
//

#ifndef VISO_TYPES_H
#define VISO_TYPES_H

using U8 = unsigned char;

#include <Eigen/Core>
#include <Eigen/Dense>

using M4d = Eigen::Matrix4d;
using M4f = Eigen::Matrix4f;
using M4i = Eigen::Matrix4i;

using M6d = Eigen::Matrix<double, 6, 6>;

using M34d = Eigen::Matrix<double, 3, 4>;
using M16d = Eigen::Matrix<double, 16, 16>;
using M26d = Eigen::Matrix<double, 2, 6>;

using M2d = Eigen::Matrix2d;
using M3d = Eigen::Matrix3d;
using M3f = Eigen::Matrix3f;
using M3i = Eigen::Matrix3i;

using V4d = Eigen::Vector4d;
using V4f = Eigen::Vector4f;
using V4i = Eigen::Vector4i;

using V3d = Eigen::Vector3d;
using V3f = Eigen::Vector3f;
using V3i = Eigen::Vector3i;

using V2d = Eigen::Vector2d;
using V2f = Eigen::Vector2f;
using V2i = Eigen::Vector2i;

using V16d = Eigen::Matrix<double, 16, 1>;
using V6d = Eigen::Matrix<double, 6, 1>;

#include <opencv2/opencv.hpp>

#endif
