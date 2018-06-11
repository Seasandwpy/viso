
#include "keyframe.h"
#include "map_point.h"
#include "types.h"

class Map {
private:
    std::vector<Keyframe::Ptr> keyframes_;
    std::vector<MapPoint::Ptr> points_;

public:
    Map() = default;
    ~Map() = default;

    inline void AddKeyframe(Keyframe::Ptr keyframe) { keyframes_.push_back(keyframe); }
    inline void AddPoint(MapPoint::Ptr map_point) { points_.push_back(map_point); }

    inline std::vector<Keyframe::Ptr> Keyframes() { return keyframes_; }
    inline std::vector<MapPoint::Ptr> GetPoints() { return points_; }
};
