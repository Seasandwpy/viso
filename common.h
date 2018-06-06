//
// Created by sage on 04.06.18.
//

#ifndef FINAL_COMMON_H
#define FINAL_COMMON_H

#include "types.h"

inline M4d MakeSE3(M3d R, V3d T)
{
  M4d result = M4d::Zero();
  result.block<3, 3>(0, 0) = R;
  result.block<3, 1>(0, 3) = T;
  result(3, 3) = 1;
  return result;
}

inline M3d Hat(V3d v)
{
  M3d result;
  result << 0.0, -v(2), v(1),
      v(2), 0.0, -v(0),
      -v(1), v(0), 0.0;
  return result;
}

#endif //FINAL_COMMON_H
