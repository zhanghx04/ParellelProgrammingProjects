cmake_minimum_required(VERSION 2.8.12)
project( homework3 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( SLICSuperpixels )

find_package(Boost REQUIRED COMPONENTS thread)
include_directories(${Boost_INCLUDE_DIRS})


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
add_executable( homework3 main.cpp )

add_library(distanceThreader DistanceThreader.cpp)
target_link_libraries( homework3 ${OpenCV_LIBS} ${Boost_LIBRARIES} distanceThreader)