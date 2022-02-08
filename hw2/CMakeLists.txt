cmake_minimum_required(VERSION 2.8.12)
project( homework2 )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( SLICSuperpixels )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable( homework2 main.cpp )
target_link_libraries( homework2 ${OpenCV_LIBS} )