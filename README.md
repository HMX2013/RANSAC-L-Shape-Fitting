## RANSAC L-shape Fitting for 3D LiDAR Point Clouds
An ROS implementation for RANSAC based L-shape fitting for 3D LiDAR point clouds

![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-informational?style=flat&logo=ubuntu&logoColor=white&color=2bbc8a)
![ROS](https://img.shields.io/badge/Tools-ROS-informational?style=flat&logo=ROS&logoColor=white&color=2bbc8a)
![C++](https://img.shields.io/badge/Code-C++-informational?style=flat&logo=c%2B%2B&logoColor=white&color=2bbc8a)

![demo_1](media/demo_01.png)


## Reference
* L-Shape Fitting-Based Vehicle Pose Estimation and Tracking Using 3D-LiDAR. IEEE TRANSACTIONS ON INTELLIGENT VEHICLES. 2020

## Features
* very fast comparing to the optimization-based L-shape fitting algorithm

**TODOs**
* complete ransac L shape fitting
* imporove the stability

**Known Issues**
* the fitting may be invalid if there are very few point clouds.

## Dependencies
* the segementation part to output topic /segmentation/detected_objects
* autoware-msgs
* pcl

## How to use
    # clone the repo
    mkdir -p catkin_ws/src
    cd catkin_ws/src
    git clone ...
    cd ../
    catkin_make 
    roslaunch ransac_lshape_fitting ransac_lshape_fitting.launch

## Contribution
You are welcome contributing to the package by opening a pull-request

We are following: 
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), 
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#main), 
and [ROS C++ Style Guide](http://wiki.ros.org/CppStyleGuide)

## License
MIT License
