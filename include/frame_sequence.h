
#ifndef VISO_FRAME_SEQUENCE_H
#define VISO_FRAME_SEQUENCE_H

#include <string>
#include <memory>

#include "types.h"
#include "keyframe.h"

class FrameSequence
{
  public:
    class FrameHandler
    {
    public:
        virtual void OnNewFrame(Keyframe::Ptr keyframe) = 0;
    };

    FrameSequence(std::string location,
                  FrameHandler *handler) : location_(location),
                                           handler_(handler)
    {
    }

    void RunOnce()
    {
        bool success = true;
        std::string file = "000000";

        {
            std::string tmp = std::to_string(Keyframe::GetNextId());
            for (int i = 0; i < tmp.size(); ++i)
            {
                file[file.size() - tmp.size() + i] = tmp[i];
            }
        }

        file = file + ".png";

        cv::Mat frame = cv::imread(file, 0);
        success = (frame.data != NULL);

        if (success)
        {
          handler_->OnNewFrame(std::make_shared<Keyframe>(frame));
        }
        else
        {
          std::cerr << "Cannot open file " << file << "\n";
        }
    };

  private:
    std::string location_;
    FrameHandler *handler_;
};

#endif