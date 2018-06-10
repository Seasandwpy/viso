
#include "keyframe.h"
#include "map_point.h"
#include "types.h"

class Map {
private:
  std::vector<Keyframe::Ptr> keyframes_;
  std::vector<MapPoint> points_;

public:
  Map() = default;

  ~Map() = default;

  inline void AddKeyframe(Keyframe::Ptr keyframe) { keyframes_.push_back(keyframe); }

  inline void AddPoint(MapPoint p) { points_.push_back(p); }

  inline std::vector<Keyframe::Ptr> &Keyframes() { return keyframes_; }

  inline std::vector<MapPoint> GetPoints() { return points_; }
};
