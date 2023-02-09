cmake_minimum_required(VERSION 2.8.3)
project(ransac_lshape_fitting)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  tf
  obsdet_msgs
)

find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")


catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS pcl_ros obsdet_msgs
)

link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set(SOURCE_FILES
  src/ransac_Lshape_fitting.cpp
  src/line_estimator.cc
)

add_executable(ransac_lshape_fitting ${SOURCE_FILES})

add_dependencies(ransac_lshape_fitting
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(ransac_lshape_fitting
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)