
#ifndef OBJECT_TRACKING_BOX_FITTING_H
#define OBJECT_TRACKING_BOX_FITTING_H

#include <ros/ros.h>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#define EIGEN_MPL2_ONLY

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "obsdet_msgs/DetectedObject.h"
#include "obsdet_msgs/DetectedObjectArray.h"

#include "obsdet_msgs/CloudCluster.h"
#include "obsdet_msgs/CloudClusterArray.h"


#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <chrono>
#include <std_msgs/Float32.h>

#define __APP_NAME__ "RANSAC L-shape Fitting"

static ros::Publisher time_ransacLshape_pub;

static ros::Publisher pub_jskrviz_time_;
static std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
static std_msgs::Float32 time_spent;
static double exe_time = 0.0;

// Kitti
const int HORZ_SCAN = 4500;
const int VERT_SCAN = 64;
const float MAX_VERT_ANGLE = 2.0;
const float MIN_VERT_ANGLE = -24.8;

double bias_total = 0.0;

class LShapeFilter
{
private:
  tf2_ros::TransformListener tf2_listener;
  tf2_ros::Buffer tf2_buffer;

  std::vector<float> vert_angles_;
  std::vector<double> bias_vector;
  std::vector<int> runtime_vector;
  bool first_appear;

  std::string input_cluster_topic_;
  std::string output_bbox_topic_;
  std::string corner_point_topic_;

  std::string bbox_source_frame_;
  std::string bbox_target_frame_;

  ros::NodeHandle node_handle_;
  ros::Subscriber sub_object_array_;
  ros::Publisher pub_object_array_;
  ros::Publisher pub_autoware_bboxs_array_;
  ros::Publisher pub_jsk_bboxs_array_;
  ros::Publisher pub_jsk_bboxs_array_gt_;
  ros::Publisher pub_corner_point_;

  ros::Publisher pub_left_side_point_;
  ros::Publisher pub_right_side_point_;
  ros::Publisher pub_ransac_line_left_;
  ros::Publisher pub_ransac_line_right_;
  ros::Publisher pub_rec_corner_points_;
  ros::Publisher pub_local_obstacle_info_;

  void MainLoop(const obsdet_msgs::CloudClusterArray& in_cluster_array);

  int getRowIdx(pcl::PointXYZ pt);
  int getColIdx(pcl::PointXYZ pt);
  void eval_running_time(int running_time);
  void eval_performance(double &theta_kitti, double &theta_optim, const uint32_t &index, const uint32_t &index_seq);
  bool ransac_Lshape_core(pcl::PointCloud<pcl::PointXYZ> &cluster, double &theta_star,
              pcl::PointCloud<pcl::PointXYZI>::Ptr &left_side_pcl, pcl::PointCloud<pcl::PointXYZI>::Ptr &right_side_pcl,
              pcl::PointXYZI &corner_point, pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_left, 
              pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_right, std::vector<cv::Point2f>& rec_corner_points);
  jsk_recognition_msgs::BoundingBox jsk_bbox_transform(const obsdet_msgs::DetectedObject &autoware_bbox, const std_msgs::Header& header);
  void calculateDimPos(double &theta_star, obsdet_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, pcl::PointCloud<pcl::PointXYZ> &cluster);
  void theta_trans(double &theta_atan);
  
public:
  LShapeFilter();
};

#endif  // OBJECT_TRACKING_BOX_FITTING_H
