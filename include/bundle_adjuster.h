//
// Created by sage on 09.06.18.
//

#ifndef VISO_BUNDLE_ADJUSTER_H
#define VISO_BUNDLE_ADJUSTER_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <sophus/se3.hpp>

#include "keyframe.h"
#include "types.h"

class BundleAdjuster {
  class VertexPoint : public g2o::BaseVertex<3, V3d> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
      // reset to zero
      _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update) override {
      // update
      _estimate += V3d(update);
    }
  };

  // g2o vertex that use sophus::SE3 as pose
  class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() {
      _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
      Eigen::Map<const Eigen::Matrix<double, 6, 1> > update(update_);
      setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
  };

  class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, V16d, VertexPoint, VertexPose> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(Keyframe::Ptr srcFrame, Keyframe::Ptr targetFrame) {
      this->srcFrame = srcFrame;
      this->targetFrame = targetFrame;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
      const VertexPoint *p = static_cast<const VertexPoint *>(vertex(0));
      const VertexPose *T = static_cast<const VertexPose *>(vertex(1));
      M3d R = T->estimate().rotationMatrix();
      V3d t = T->estimate().translation();

      V3d target_uv1 = targetFrame->K() * (R * p->estimate() + t);
      target_uv1 /= target_uv1.z();
      V2d src_uv = srcFrame->Project(p->estimate());

      int idx = 0;
      bool abort = false;
      for (int j = -2; (j <= 1) && !abort; ++j) {
        for (int i = -2; i <= 1; ++i) {

          float u1 = (float) (target_uv1.x() + (double) i);
          float v1 = (float) (target_uv1.y() + (double) j);
          float u2 = (float) (src_uv.x() + (double) i);
          float v2 = (float) (src_uv.y() + (double) j);

          if (!targetFrame->IsInside(u1, v1) || !srcFrame->IsInside(u2, v2)) {
            setLevel(1);
            abort = true;
            break;
          }

          _error[idx] = srcFrame->GetPixelValue(u2, v2) - targetFrame->GetPixelValue(u1, v1);
          ++idx;
        }
      }
    }

  private:
    Keyframe::Ptr targetFrame;
    Keyframe::Ptr srcFrame;
  };

  using Block = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> >;

public:
  BundleAdjuster() {
    std::unique_ptr<Block::LinearSolverType> linearSolver(
      new g2o::LinearSolverDense<Block::PoseMatrixType>());

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<Block>(std::move(linearSolver)));

    optimizer_.setAlgorithm(solver); // solver
    optimizer_.setVerbose(true); // open the output

    int id = 0;

    points_.reserve(max_points);
    for (int i = 0; i < max_points; ++i) {
      VertexPoint *p = new VertexPoint();
      p->setId(id);
      p->setMarginalized(true);
      points_.push_back(p);

      //pv->setEstimate(points[i]);

      if (!optimizer_.addVertex(p)) {
        std::cerr << "error adding vertex" << std::endl;
      }

      id++;
    }

    poses_.reserve(max_poses);
    for (int i = 0; i < max_poses; ++i) {
      VertexPose *p = new VertexPose();
      p->setId(id);
      poses_.push_back(p);

      //p->setEstimate(poses[j]);

      if (!optimizer_.addVertex(p)) {
        std::cerr << "error adding vertex" << std::endl;
      }

      id++;
    }

    edges_.reserve(max_edges);
    for (int i = 0; i < max_edges; ++i) {
      EdgeDirectProjection *e = new EdgeDirectProjection(color[i], images[j]);
      e->setVertex(0, point_vertices[i]); // first vertex is the point
      e->setVertex(1, pose_vertices[j]); // second vertex is pose
      e->setInformation(M16d::Identity());

      if (!optimizer_.addEdge(e)) {
        std::cerr << "error adding edge" << std::endl;
        exit(0);
      }
    }
  }

  inline int AddPoint(V3d p) {}

  inline int AddPose(Sophus::SE3d p) {}

  inline void AddEdge(int point, int pose) {}

private:
  std::vector<VertexPoint *> points_;
  std::vector<VertexPose *> poses_;
  std::vector<EdgeDirectProjection *> edges_;

  g2o::SparseOptimizer optimizer_;

  const int max_points = 1000;
  const int max_poses = 1000;
  const int max_edges = 1000;

  int id = 0;
};

#endif //VISO_BUNDLE_ADJUSTER_H
