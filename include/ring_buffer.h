//
// Created by sage on 09.06.18.
//

#ifndef VISO_RING_BUFFER_H
#define VISO_RING_BUFFER_H

#include <vector>

template <int N, typename T>
class RingBuffer {
public:
    RingBuffer()
        : end_(0)
        , size_(0)
    {
        ring_.resize(N);
    }

    inline void Push(T t)
    {
        ring_[end_] = t;
        end_ = (end_ + 1) % N;
        size_ = std::min(size_ + 1, N);
    }

    inline int GetSize() { return size_; }

    inline T operator[](int index)
    {
        int start = end_ - size_;
        if (start < 0) {
            start += N;
        }
        return ring_[(start + index) % N];
    }

private:
    std::vector<T> ring_;
    int end_;
    int size_;
};

#endif //VISO_RING_BUFFER_H
