
#include "common.h"
#include "viso.h"
#include <iostream>
#include <pangolin/pangolin.h>

struct PangoState {
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
};

void DrawMap(PangoState *pango, std::vector<V3d> points, const std::vector<Sophus::SE3d> &poses);

int main(int argc, char const *argv[])
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
    Viso viso(517.3, 516.5, 325.1, 249.7);
    FrameSequence sequence("rgb/", &viso);

    while (!pangolin::ShouldQuit()) {
        sequence.RunOnce();
        DrawMap(&pango_state, viso.GetPoints(), viso.poses);
    }

    return 0;
}

void DrawMap(PangoState *pango, std::vector<V3d> points, const std::vector<Sophus::SE3d> &poses)
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
    float sz = 0.5;
    int width = 640, height = 480;
    for (auto pose : poses) {
        glPushMatrix();

        double f = 500;

        Sophus::Matrix4f m = pose.inverse().matrix().cast<float>();
        glMultMatrixf((GLfloat *) m.data());
        glColor3f(1, 0, 0);
        glLineWidth(2);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - 0) / f, sz * (0 - 0) / f, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - 0) / f, sz * (height - 1 - 0) / f, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - 0) / f, sz * (height - 1 - 0) / f, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - 0) / f, sz * (0 - 0) / f, sz);
        glVertex3f(sz * (width - 1 - 0) / f, sz * (0 - 0) / f, sz);
        glVertex3f(sz * (width - 1 - 0) / f, sz * (height - 1 - 0) / f, sz);
        glVertex3f(sz * (width - 1 - 0) / f, sz * (height - 1 - 0) / f, sz);
        glVertex3f(sz * (0 - 0) / f, sz * (height - 1 - 0) / f, sz);
        glVertex3f(sz * (0 - 0) / f, sz * (height - 1 - 0) / f, sz);
        glVertex3f(sz * (0 - 0) / f, sz * (0 - 0) / f, sz);
        glVertex3f(sz * (0 - 0) / f, sz * (0 - 0) / f, sz);
        glVertex3f(sz * (width - 1 - 0) / f, sz * (0 - 0) / f, sz);
        glEnd();
        glPopMatrix();
    }
    pangolin::FinishFrame();
}
