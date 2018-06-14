
#ifndef VISO_FRAME_SEQUENCE_H
#define VISO_FRAME_SEQUENCE_H

#include <memory>
#include <string>

#include "keyframe.h"
#include "types.h"
#include <vector>

class FrameSequence {
public:
    class FrameHandler {
    public:
        virtual void OnNewFrame(Keyframe::Ptr keyframe) = 0;
    };

    FrameSequence(std::string location,
        FrameHandler* handler, std::vector<std::string> files,
        std::vector<std::string> times)
        : location_(location)
        , handler_(handler), files_(files), times_(times)
    {
    }

    void RunOnce()
    {
        bool success = true;
        //std::string file = location_ + std::to_string(Keyframe::GetNextId() + 1) + ".png";

        cv::Mat frame = cv::imread(files_[Keyframe::GetNextId()], 0);
        success = (frame.data != NULL);

        if (success) {
            handler_->OnNewFrame(std::make_shared<Keyframe>(frame, times_[Keyframe::GetNextId()]));
        } else {
            std::cerr << "Cannot open file " << files_[Keyframe::GetNextId()] << "\n";
        }
    };

private:
    std::string location_;
    FrameHandler* handler_;
    std::vector<std::string> files_;
    std::vector<std::string> times_;
};

#endif
