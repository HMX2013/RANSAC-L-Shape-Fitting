
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

#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#define __APP_NAME__ "RANSAC L-shape Fitting"

class LShapeFilter
{
private:

  tf2_ros::TransformListener tf2_listener;
  tf2_ros::Buffer tf2_buffer;

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
  ros::Publisher pub_corner_point_;

  ros::Publisher pub_left_side_point_;
  ros::Publisher pub_right_side_point_;
  ros::Publisher pub_ransac_line_left_;
  ros::Publisher pub_ransac_line_right_;
  ros::Publisher pub_rec_corner_points_;

  void MainLoop(const autoware_msgs::DetectedObjectArray& in_cluster_array);
  void updateCpFromPoints(const std::vector<cv::Point2f>& pointcloud_frame_points,
                          autoware_msgs::DetectedObject& output);
  void toRightAngleBBox(std::vector<cv::Point2f>& pointcloud_frame_points);
  void updateDimentionAndEstimatedAngle(const std::vector<cv::Point2f>& pcPoints,
                                        autoware_msgs::DetectedObject& object);
  void getPointsInPointcloudFrame(cv::Point2f rect_points[], std::vector<cv::Point2f>& pointcloud_frame_points,
                                  const cv::Point& offset_point);
  bool ransac_Lshape(pcl::PointCloud<pcl::PointXYZ> &cluster, autoware_msgs::DetectedObject &output_object, 
              pcl::PointCloud<pcl::PointXYZI>::Ptr &left_side_pcl, pcl::PointCloud<pcl::PointXYZI>::Ptr &right_side_pcl,
              pcl::PointXYZI &corner_point, pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_left, 
              pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_right, std::vector<cv::Point2f>& rec_corner_points);
  jsk_recognition_msgs::BoundingBox jsk_bbox_transform(const autoware_msgs::DetectedObject &autoware_bbox, 
          const std_msgs::Header& header);
  void calculateDimPos(double &theta_star, autoware_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, pcl::PointCloud<pcl::PointXYZ> &cluster);
  double calcu_coeffic2angle(const pcl::ModelCoefficients::Ptr &coefficients);
  void pca_fitting(const pcl::PointCloud<pcl::PointXYZ>& cluster,  autoware_msgs::DetectedObject& output);
  
public:
  LShapeFilter();
};

#endif  // OBJECT_TRACKING_BOX_FITTING_H
