cmake_minimum_required(VERSION 2.8.3)
project(ransac_lshape_fitting)

find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  pcl_ros
  roscpp
  tf
)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  CATKIN_DEPENDS
    pcl_ros
    autoware_msgs
)


link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE_FILES
  nodes/ransac_lshape_fitting/ransac_Lshape_fitting_node.cpp
  nodes/ransac_lshape_fitting/ransac_Lshape_fitting.cpp
)

add_executable(ransac_lshape_fitting ${SOURCE_FILES})

add_dependencies(ransac_lshape_fitting
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(ransac_lshape_fitting
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)