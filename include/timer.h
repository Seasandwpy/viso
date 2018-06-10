//
// Created by sage on 07.06.18.
//

#ifndef VISO_TIMER_H
#define VISO_TIMER_H

#include <chrono>
#include <iostream>

class Timer {
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;
    using Units = std::chrono::milliseconds;

    Timer()
    {
        start_ = Clock::now();
    }

    long GetElapsed()
    {
        return std::chrono::duration_cast<Units>(Clock::now() - start_).count();
    }

    void Reset()
    {
        start_ = Clock::now();
    }

private:
    Time start_;
};

#endif //VISO_TIMER_H
