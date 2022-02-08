cmake_minimum_required(VERSION 2.8.12)
project( homework4 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( SLICSuperpixels )

find_package(MPI REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable( homework4 main.cpp )

# add_library(distanceThreader DistanceThreader.cpp)
target_link_libraries( homework4 PUBLIC MPI::MPI_CXX ${OpenCV_LIBS} )