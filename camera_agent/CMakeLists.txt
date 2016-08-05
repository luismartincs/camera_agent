cmake_minimum_required(VERSION 2.8.12)

project(camera_agent)
add_definitions(-std=c++11)
file(GLOB_RECURSE SRC_FILES pattern.h patterndetector.h cameraparams.h pattern.cpp patterndetector.cpp camera_agent.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(camera_agent ${SRC_FILES})
target_link_libraries(camera_agent ${OpenCV_LIBS})
