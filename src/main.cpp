
#include "common.h"
#include "viso.h"
#include <iostream>
#include <pangolin/pangolin.h>

struct PangoState {
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
};

void DrawMap(PangoState* pango, std::vector<V3d> points, const std::vector<Sophus::SE3d>& poses);

const double fx = 517.3;
const double fy = 516.5;
const double cx = 325.1;
const double cy = 249.7;
    
int main(int argc, char const* argv[])
{
    //
    // Initialize pangolin.
    //
    pangolin::CreateWindowAndBind("Map", 1024, 768);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    PangoState pango_state;
    pango_state.s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 10000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pango_state.d_cam = pangolin::CreateDisplay()
                            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                            .SetHandler(new pangolin::Handler3D(pango_state.s_cam));

    //
    // Run main loop.
    //

    //    Viso viso(200.0, 200.0, 240.0, 240.0);
    //    FrameSequence sequence("", &viso);
    Viso viso(fx, fy, cx, cy);
    FrameSequence sequence("rgb/", &viso);

    while (!pangolin::ShouldQuit()) {
        sequence.RunOnce();
        DrawMap(&pango_state, viso.GetPoints(), viso.poses);
    }

    return 0;
}

void DrawMap(PangoState* pango, std::vector<V3d> points, const std::vector<Sophus::SE3d>& poses)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pango->d_cam.Activate(pango->s_cam);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // points
    glPointSize(2);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < points.size(); i++) {
        glColor3f(1.0, 1.0, 1.0);
        glVertex3d(points[i].x(), points[i].y(), points[i].z());
    }
    glEnd();

    // draw poses
    float sz = 0.1;
    int width = 640, height = 480;
        
    for (auto &Tcw: poses)
    {
      glPushMatrix();
      Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
      glMultMatrixf((GLfloat *) m.data());
      glColor3f(0, 0, 1);
      glLineWidth(2);
      glBegin(GL_LINES);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(0, 0, 0);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
      glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
      glEnd();
      glPopMatrix();
    }
    pangolin::FinishFrame();
}
