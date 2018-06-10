//
// Created by sage on 09.06.18.
//

#ifndef VISO_MAP_POINT_H
#define VISO_MAP_POINT_H

#include "keyframe.h"
#include "types.h"

struct MapPoint {
    MapPoint(Keyframe::Ptr keyframe, V2d frame_pos, V3d world_pos)
        : keyframe(keyframe)
        , frame_pos(frame_pos)
        , world_pos(world_pos)
    {
    }

    V3d world_pos;
    Keyframe::Ptr keyframe;
    V2d frame_pos;
};

#endif //VISO_MAP_POINT_H
