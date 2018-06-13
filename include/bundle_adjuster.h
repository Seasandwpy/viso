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
using namespace g2o;
    /*class VertexPoint : public g2o::BaseVertex<3, V3d> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexPoint()
        {
        }

        virtual bool read(std::istream& )
        {
            //cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        virtual bool write(std::ostream& ) const
        {
            //cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }
        virtual void setToOriginImpl() override
        {
            // reset to zero
            //_estimate << 0, 0, 0;
        }

        virtual void oplusImpl(const double* update) override
        {
            // update
            V3d::ConstMapType v(update);
            _estimate += v;
        }
    };*/

    // g2o vertex that use sophus::SE3 as pose
    // g2o vertex that use sophus::SE3 as pose
    class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d>
    {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        VertexSophus() {}

        ~VertexSophus() {}

        bool read(std::istream &is) {}

        bool write(std::ostream &os) const {}

        virtual void setToOriginImpl()
        {
            _estimate = Sophus::SE3d();
        }

        virtual void oplusImpl(const double *update_)
        {
            Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
            setEstimate(Sophus::SE3d::exp(update) * estimate());
        }
    };


    class EdgeObservation : public g2o::BaseBinaryEdge<2, V2d, g2o::VertexSBAPointXYZ, VertexSophus> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeObservation(M3d K)
        {
            this->K_ = K;
        }

        ~EdgeObservation() {}

        virtual void computeError() override
        {
            //double fx, cx, fy, cy;
            //fx = 517.3; fy = 516.5; cx= 325.1; cy=249.7;
            const VertexSBAPointXYZ* p = static_cast<const VertexSBAPointXYZ*> ( _vertices[0] );
            const VertexSophus* c = static_cast<const VertexSophus*> ( _vertices[1] );
            V3d global = p->estimate();
            V3d local = c->estimate()*global;
            double u = local[0]*K_(0,0)/local[2] + K_(0,2);
            double v = local[1]*K_(1,1)/local[2] + K_(1,2);
            _error(0,0) = u-_measurement[0];
            _error(1,0) = v-_measurement[1];

        }
        virtual bool read(std::istream& /*is*/)
        {
            //cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }

        virtual bool write(std::ostream& /*os*/) const
        {
            //cerr << __PRETTY_FUNCTION__ << " not implemented yet" << endl;
            return false;
        }
    private:
        M3d K_;  // the source image
    };
 
 

#endif //VISO_BUNDLE_ADJUSTER_H
