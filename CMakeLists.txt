cmake_minimum_required(VERSION 2.8)
project(viso)
set(CMAKE_CXX_FLAGS "-std=c++11")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# opencv
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3")  # change this if your eigen is not installed by apt-get

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories(include)

add_executable(viso src/keyframe.cpp include/viso.h src/viso.cpp src/main.cpp )
target_link_libraries(viso ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

