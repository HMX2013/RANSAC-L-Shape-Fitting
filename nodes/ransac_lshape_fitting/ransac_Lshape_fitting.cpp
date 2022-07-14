#include <random>

#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_datatypes.h>
#include <chrono>
#include "ransac_Lshape_fitting.h"

LShapeFilter::LShapeFilter() : tf2_listener(tf2_buffer)
{
  // L-shape fitting params
  ros::NodeHandle private_nh_("~");

  node_handle_.param<std::string>("input_cluster_topic", input_cluster_topic_, "/segmentation/detected_objects");
  ROS_INFO("Input_cluster_topic: %s", input_cluster_topic_.c_str());

  node_handle_.param<std::string>("output_bbox_topic_", output_bbox_topic_, "/l_shape_fitting/bbox_visual_jsk");
  ROS_INFO("output_bbox_topic_: %s", output_bbox_topic_.c_str());

  private_nh_.param<std::string>("bbox_target_frame", bbox_target_frame_, "velodyne_1");
  ROS_INFO("[%s] bounding box's target frame is: %s", __APP_NAME__, bbox_target_frame_);

  node_handle_.param<std::string>("corner_point_topic_", corner_point_topic_, "/l_shape_fitting/closest_point_cloud");
  ROS_INFO("corner_point_topic_: %s", corner_point_topic_.c_str());

  sub_object_array_ = node_handle_.subscribe("/segmentation/detected_objects", 1, &LShapeFilter::MainLoop, this);
  pub_corner_point_ = node_handle_.advertise<sensor_msgs::PointCloud2>(corner_point_topic_,1);

  pub_left_side_point_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/l_shape_fitting/left_side_point", 1);
  pub_right_side_point_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/l_shape_fitting/right_side_point", 1);
  pub_ransac_line_left_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/l_shape_fitting/ransac_line_left", 1);
  pub_ransac_line_right_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/l_shape_fitting/ransac_line_right", 1);

  pub_autoware_bboxs_array_ = node_handle_.advertise<autoware_msgs::DetectedObjectArray>("/l_shape_fitting/autoware_bboxs_array/objects", 1);
  pub_jsk_bboxs_array_ = node_handle_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/l_shape_fitting/jsk_bboxs_array",1);

  pub_rec_corner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/l_shape_fitting/rec_corner_points", 1);
}


bool LShapeFilter::ransac_Lshape(pcl::PointCloud<pcl::PointXYZ> &cluster, autoware_msgs::DetectedObject &output_object, 
                                pcl::PointCloud<pcl::PointXYZI>::Ptr &left_side_pcl, pcl::PointCloud<pcl::PointXYZI>::Ptr &right_side_pcl,
                                pcl::PointXYZI &corner_point, pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_left, 
                                pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_right, std::vector<cv::Point2f>& rec_corner_points)
{
  int num_points = cluster.size();
  double theta_star;
 
  // Searching the corner point
  float min_dist = std::numeric_limits<float>::max();
  
  int corner_index;
  
  pcl::PointXYZI side_point;

  for (int i_point = 0; i_point < num_points; i_point++)
  {
    const float p_x = cluster[i_point].x;
    const float p_y = cluster[i_point].y;
    
    float distance_o = p_x * p_x + p_y * p_y;

    if (distance_o < min_dist)
    {
      min_dist = distance_o;
      corner_index = i_point;
    }

    if (min_dist == std::numeric_limits<float>::max())
    {
      continue;
    }
  }

  corner_point.x = cluster[corner_index].x;
  corner_point.y = cluster[corner_index].y;
  corner_point.z = cluster[corner_index].z;
  corner_point.intensity = min_dist;


  // divide these point into sets A and B by column
  float verticalAngle, horizonAngle;
  size_t columnIdn, column_corner;
  const int N_SCAN = 16;
  const int Horizon_SCAN = 1800;
  const float ang_res_x = 0.2;

  horizonAngle = atan2(cluster[corner_index].x, cluster[corner_index].y) * 180 / M_PI;

  column_corner = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;

  if (column_corner >= Horizon_SCAN)
    column_corner -= Horizon_SCAN;

  if (column_corner < 0 || column_corner >= Horizon_SCAN)
    return false;

  for (int index = 0; index < num_points; index++)
  {
    if (index == corner_index)
      continue;

    horizonAngle = atan2(cluster[index].x, cluster[index].y) * 180 / M_PI;

    columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
    if (columnIdn >= Horizon_SCAN)
      columnIdn -= Horizon_SCAN;

    if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
      continue;

    if (columnIdn < column_corner)
    {
      side_point.x = cluster[index].x;
      side_point.y = cluster[index].y;
      side_point.z = 0;
      side_point.intensity = 1;
      left_side_pcl->points.push_back(side_point);
    }
    else
    {
      side_point.x = cluster[index].x;
      side_point.y = cluster[index].y;
      side_point.z = 0;
      side_point.intensity = 2;
      right_side_pcl->points.push_back(side_point);
    }
  }

  if (left_side_pcl->size() <= 10 && left_side_pcl->size() <= 10)
    return false;

  // using ransac to fitting two line
  pcl::ModelCoefficients::Ptr coefficients_left(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_left(new pcl::PointIndices);  //inliers表示误差能容忍的点 记录的是点云的序号
  pcl::ModelCoefficients::Ptr coefficients_right(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_right(new pcl::PointIndices);

  bool perform_ransac_left = false;
  bool perform_ransac_right = false;

  pcl::SACSegmentation<pcl::PointXYZI> ransac_;
  ransac_.setOptimizeCoefficients(true);    // Optional，这个设置可以选定结果平面展示的点是分割掉的点还是分割剩下的点。
  ransac_.setModelType(pcl::SACMODEL_LINE); // Mandatory-设置目标几何形状
  ransac_.setMethodType(pcl::SAC_RANSAC);   // method
  ransac_.setMaxIterations(2000);           // maximum iterations
  ransac_.setDistanceThreshold(0.2);       //设置误差容忍范围，也就是阈值 这东西不能太小，自己多试试

  // ROS_INFO("left_side_pcl=%d", left_side_pcl->size());
  // ROS_INFO("right_side_pcle%d", right_side_pcl->size());

  if (left_side_pcl->size() >= 10)
  {
    ransac_.setInputCloud(left_side_pcl);
    ransac_.segment(*inliers_left, *coefficients_left);
    perform_ransac_left = true;
  }

  if (right_side_pcl->size() >= 10)
  {
    ransac_.setInputCloud(right_side_pcl);
    ransac_.segment(*inliers_right, *coefficients_right);
    perform_ransac_right = true;
  }

  std::cout << "perform_ransac_left =" << perform_ransac_left << std::endl;
  std::cout << "perform_ransac_right =" << perform_ransac_right << std::endl;

  // fitting zero line
  if (perform_ransac_left == false && perform_ransac_right == false)
    return false;

  if (perform_ransac_left){
    for (int i = 0; i < inliers_left->indices.size(); ++i){
      c_line_left->points.push_back(left_side_pcl->points.at(inliers_left->indices[i]));
    }
  }
  if (perform_ransac_right){
    for (int i = 0; i < inliers_right->indices.size(); ++i){
      c_line_right->points.push_back(right_side_pcl->points.at(inliers_right->indices[i]));
    }
  }

  // ROS_INFO("c_line left size=%d", c_line_left->size());
  // ROS_INFO("c_line right size=%d", c_line_right->size());

  bool has_calcu_theta = false;
  pcl::ModelCoefficients::Ptr coefficients_final(new pcl::ModelCoefficients);

  if (perform_ransac_left == false || perform_ransac_right == false)
  {
    if (perform_ransac_left == true)
      coefficients_final = coefficients_left;
    else
      coefficients_final = coefficients_right;
    
    theta_star = calcu_coeffic2angle(coefficients_final);

    has_calcu_theta = true;
  }

  // fitting two line
  if (has_calcu_theta == false)
  {
    double k1 = coefficients_left->values[4] / coefficients_left->values[3];
    double k2 = coefficients_right->values[4] / coefficients_right->values[3];
    double inters_angle = atan(abs(k1 - k2) / abs(1 + k1 * k2)) * 180 / M_PI;

    // ROS_INFO("k1=%f", k1);
    // ROS_INFO("k2=%f", k2);
    // ROS_INFO("inters_angle=%f", inters_angle);

    if (inters_angle < 10)
    {
      ransac_.setDistanceThreshold(0.2);
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_ptr(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::PointIndices::Ptr inliers_full(new pcl::PointIndices);

      pcl::copyPointCloud(cluster, *cluster_ptr);

      ransac_.setInputCloud(cluster_ptr);
      ransac_.segment(*inliers_full, *coefficients_final);
    }
    else
    {
      if (c_line_left->size() > c_line_right->size())
      {
        coefficients_final = coefficients_left;
      }
      else
      {
        coefficients_final = coefficients_right;
      }
    }
    theta_star = calcu_coeffic2angle(coefficients_final);
    has_calcu_theta = true;
  }

  // if (!has_calcu_theta)
  //   theta_star = 0;

  calculateDimPos(theta_star, output_object, rec_corner_points, cluster);

  if (std::max(output_object.dimensions.x, output_object.dimensions.y) > 5)
    return false;

  return true;
}


double LShapeFilter::calcu_coeffic2angle(const pcl::ModelCoefficients::Ptr &coefficients)
{
  double theta_star;
  theta_star = atan(coefficients->values[4] / coefficients->values[3]);
  if (theta_star < 0)
    theta_star = theta_star + M_PI;
  return theta_star;
}


void LShapeFilter::calculateDimPos(double &theta_star, autoware_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, pcl::PointCloud<pcl::PointXYZ> &cluster)
{  
  // calc centroid point for cylinder height(z)
  pcl::PointXYZ centroid;
  centroid.x = 0;
  centroid.y = 0;
  centroid.z = 0;
  for (const auto& pcl_point : cluster)
  {
    centroid.x += pcl_point.x;
    centroid.y += pcl_point.y;
    centroid.z += pcl_point.z;
  }
  centroid.x = centroid.x / (double)cluster.size();
  centroid.y = centroid.y / (double)cluster.size();
  centroid.z = centroid.z / (double)cluster.size();

  // calc min and max z for cylinder length
  double min_z = 0;
  double max_z = 0;
  for (size_t i = 0; i < cluster.size(); ++i)
  {
    if (cluster.at(i).z < min_z || i == 0)
      min_z = cluster.at(i).z;
    if (max_z < cluster.at(i).z || i == 0)
      max_z = cluster.at(i).z;
  }

  Eigen::Vector2d e_1_star;  // col.11, Algo.2
  Eigen::Vector2d e_2_star;
  e_1_star << std::cos(theta_star), std::sin(theta_star);
  e_2_star << -std::sin(theta_star), std::cos(theta_star);
  std::vector<double> C_1_star;  // col.11, Algo.2
  std::vector<double> C_2_star;  // col.11, Algo.2
  for (const auto& point : cluster)
  {
    C_1_star.push_back(point.x * e_1_star.x() + point.y * e_1_star.y());
    C_2_star.push_back(point.x * e_2_star.x() + point.y * e_2_star.y());
  }

  // col.12, Algo.2
  const double min_C_1_star = *std::min_element(C_1_star.begin(), C_1_star.end());
  const double max_C_1_star = *std::max_element(C_1_star.begin(), C_1_star.end());
  const double min_C_2_star = *std::min_element(C_2_star.begin(), C_2_star.end());
  const double max_C_2_star = *std::max_element(C_2_star.begin(), C_2_star.end());

  const double a_1 = std::cos(theta_star);
  const double b_1 = std::sin(theta_star);
  const double c_1 = min_C_1_star;
  const double a_2 = -1.0 * std::sin(theta_star);
  const double b_2 = std::cos(theta_star);
  const double c_2 = min_C_2_star;
  const double a_3 = std::cos(theta_star);
  const double b_3 = std::sin(theta_star);
  const double c_3 = max_C_1_star;
  const double a_4 = -1.0 * std::sin(theta_star);
  const double b_4 = std::cos(theta_star);
  const double c_4 = max_C_2_star;

  // calc center of bounding box
  double intersection_x_1 = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
  double intersection_y_1 = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);
  double intersection_x_2 = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
  double intersection_y_2 = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

  cv::Point2f rec_corner_p1, rec_corner_p2, rec_corner_p3, rec_corner_p4;

  rec_corner_p1.x = intersection_x_1;
  rec_corner_p1.y = intersection_y_1;

  rec_corner_p2.x = (b_2 * c_3 - b_3 * c_2) / (a_3 * b_2 - a_2 * b_3);
  rec_corner_p2.y = (a_2 * c_3 - a_3 * c_2) / (a_2 * b_3 - a_3 * b_2);

  rec_corner_p3.x = intersection_x_2;
  rec_corner_p3.y = intersection_y_2;

  rec_corner_p4.x = (b_1 * c_4 - b_4 * c_1) / (a_4 * b_1 - a_1 * b_4);
  rec_corner_p4.y = (a_1 * c_4 - a_4 * c_1) / (a_1 * b_4 - a_4 * b_1);

  rec_corner_points[0] = rec_corner_p1;
  rec_corner_points[1] = rec_corner_p2;
  rec_corner_points[2] = rec_corner_p3;
  rec_corner_points[3] = rec_corner_p4;

  // calc dimention of bounding box
  Eigen::Vector2d e_x;
  Eigen::Vector2d e_y;
  e_x << a_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1)), b_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1));
  e_y << a_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2)), b_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2));
  Eigen::Vector2d diagonal_vec;
  diagonal_vec << intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2;

  // calc yaw
  tf2::Quaternion quat;
  quat.setEuler(/* roll */ 0, /* pitch */ 0, /* yaw */ std::atan2(e_1_star.y(), e_1_star.x()));

  output.pose.position.x = (intersection_x_1 + intersection_x_2) / 2.0;
  output.pose.position.y = (intersection_y_1 + intersection_y_2) / 2.0;
  output.pose.position.z = centroid.z;
  output.pose.orientation = tf2::toMsg(quat);
  constexpr double ep = 0.001;
  output.dimensions.x = std::fabs(e_x.dot(diagonal_vec));
  output.dimensions.y = std::fabs(e_y.dot(diagonal_vec));
  output.dimensions.z = std::max((max_z - min_z), ep);
  output.pose_reliable = true;

  // check wrong output
  // if (output.dimensions.x < ep && output.dimensions.y < ep)
  //   return false;
  output.dimensions.x = std::max(output.dimensions.x, ep);
  output.dimensions.y = std::max(output.dimensions.y, ep);
  // return true;
}


void LShapeFilter::pca_fitting(const pcl::PointCloud<pcl::PointXYZ>& cluster,  autoware_msgs::DetectedObject& output)
{
  // Compute the bounding box height (to be used later for recreating the box)
  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(cluster, min_pt, max_pt);
  const float box_height = max_pt.z - min_pt.z;
  const float box_z = (max_pt.z + min_pt.z)/2;

  // Compute the cluster centroid 
  Eigen::Vector4f pca_centroid;
  pcl::compute3DCentroid(cluster, pca_centroid);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_project(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cluster, *cluster_project);

  // Squash the cluster to x-y plane with z = centroid z 
  for (size_t i = 0; i < cluster_project->size(); ++i)
  {
    cluster_project->points[i].z= pca_centroid(2);
  }

  // Compute principal directions & Transform the original cloud to PCA coordinates
  pcl::PointCloud<pcl::PointXYZ>::Ptr pca_projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(cluster_project);
  pca.project(*cluster_project, *pca_projected_cloud);

  const auto eigen_vectors = pca.getEigenVectors();

  // Get the minimum and maximum points of the transformed cloud.
  pcl::getMinMax3D(*pca_projected_cloud, min_pt, max_pt);
  const Eigen::Vector3f meanDiagonal = 0.5f * (max_pt.getVector3fMap() + min_pt.getVector3fMap());

  // Final transform
  const Eigen::Quaternionf quaternion(eigen_vectors); // Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
  const Eigen::Vector3f position = eigen_vectors * meanDiagonal + pca_centroid.head<3>();
  const Eigen::Vector3f dimension((max_pt.x - min_pt.x), (max_pt.y - min_pt.y), box_height);

  output.pose.position.x = position(0);
  output.pose.position.y = position(1);
  output.pose.position.z = position(2);

  output.pose.orientation.x = quaternion.x();
  output.pose.orientation.y = quaternion.y();
  output.pose.orientation.z = quaternion.z();
  output.pose.orientation.w = quaternion.w();

  output.dimensions.x=dimension(0);
  output.dimensions.y=dimension(1);
  output.dimensions.z=dimension(2);

  output.pose_reliable = true;
}

jsk_recognition_msgs::BoundingBox LShapeFilter::jsk_bbox_transform(const autoware_msgs::DetectedObject &autoware_bbox, 
          const std_msgs::Header& header)
{
  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_bbox.header = header;
  jsk_bbox.pose = autoware_bbox.pose;
  jsk_bbox.dimensions = autoware_bbox.dimensions;
  jsk_bbox.label = autoware_bbox.id;
  jsk_bbox.value = 1.0f;

  return std::move(jsk_bbox);
}

void LShapeFilter::MainLoop(const autoware_msgs::DetectedObjectArray& in_cluster_array)
{
  const auto start_time = std::chrono::steady_clock::now();

  autoware_msgs::DetectedObjectArray out_object_array;

  out_object_array.header = in_cluster_array.header;

  pcl::PointCloud<pcl::PointXYZI>::Ptr corner_pcl(new pcl::PointCloud<pcl::PointXYZI>());
  corner_pcl->header.frame_id = in_cluster_array.header.frame_id;

  pcl::PointCloud<pcl::PointXYZI>::Ptr left_side_pcl(new pcl::PointCloud<pcl::PointXYZI>());
  left_side_pcl->header.frame_id = in_cluster_array.header.frame_id;

  pcl::PointCloud<pcl::PointXYZI>::Ptr right_side_pcl(new pcl::PointCloud<pcl::PointXYZI>());
  right_side_pcl->header.frame_id = in_cluster_array.header.frame_id;

  int intensity_mark = 1;
  pcl::PointCloud<pcl::PointXYZI>::Ptr corner_points_visual(new pcl::PointCloud<pcl::PointXYZI>());

  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_recognition_msgs::BoundingBoxArray jsk_bbox_array;

/*----------------------------------transform the bounding box to target frame.-------------------------------------------*/
  geometry_msgs::TransformStamped transform_stamped;
  geometry_msgs::Pose pose, pose_transformed;
  auto bbox_header = in_cluster_array.header;
  bbox_source_frame_ = bbox_header.frame_id;
  bbox_header.frame_id = bbox_target_frame_;
  jsk_bbox_array.header = bbox_header;

  try
  {
    transform_stamped = tf2_buffer.lookupTransform(bbox_target_frame_, bbox_source_frame_, ros::Time());
    // ROS_INFO("target_frame is %s",bbox_target_frame_.c_str());
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    ROS_WARN("Frame Transform Given Up! Outputing obstacles in the original LiDAR frame %s instead...", bbox_source_frame_.c_str());
    bbox_header.frame_id = bbox_source_frame_;
    try
    {
      transform_stamped = tf2_buffer.lookupTransform(bbox_source_frame_, bbox_source_frame_, ros::Time(0));
    }
    catch (tf2::TransformException& ex2)
    {
      ROS_ERROR("%s", ex2.what());
      return;
    }
  }
/*-----------------------------------------------------------------------------------------------------------------------*/

  for (const auto& in_object : in_cluster_array.objects)
  {
    pcl::PointCloud<pcl::PointXYZ> cluster;
    pcl::fromROSMsg(in_object.pointcloud, cluster);

    autoware_msgs::DetectedObject output_object;

    // double theta_star;
    pcl::PointXYZI corner_point;

    pcl::PointCloud<pcl::PointXYZI>::Ptr c_line_left(new pcl::PointCloud<pcl::PointXYZI>);
    c_line_left->header.frame_id = in_cluster_array.header.frame_id;

    pcl::PointCloud<pcl::PointXYZI>::Ptr c_line_right(new pcl::PointCloud<pcl::PointXYZI>);
    c_line_right->header.frame_id = in_cluster_array.header.frame_id;

    std::vector<cv::Point2f> rec_corner_points(4);

    /*-----------------------------------------------------------------------------------------*/
    left_side_pcl->clear();
    right_side_pcl->clear();

    ROS_INFO("left_side_pcl size=%d", left_side_pcl->size());
    ROS_INFO("right_side_pcl size=%d", right_side_pcl->size());

    ROS_INFO("c_line left size=%d", c_line_left->size());
    ROS_INFO("c_line right size=%d", c_line_right->size());

    bool fitting_success = ransac_Lshape(cluster, output_object, left_side_pcl, right_side_pcl, corner_point, c_line_left, c_line_right, rec_corner_points);

    if (!fitting_success)
      continue;

    /*-----------------------------------------------------------------------------------------*/
    pub_ransac_line_left_.publish(c_line_left);
    pub_ransac_line_right_.publish(c_line_right);

    corner_pcl->points.push_back(corner_point);

    // transform the bounding box
    pose.position = output_object.pose.position;
    pose.orientation = output_object.pose.orientation;

    tf2::doTransform(pose, pose_transformed, transform_stamped);

    output_object.header = bbox_header;
    output_object.pose = pose_transformed;

    //copy the autoware box to jsk box
    jsk_bbox = jsk_bbox_transform(output_object, bbox_header);
    jsk_bbox_array.boxes.push_back(jsk_bbox);

    //push the autoware bounding box in the array
    out_object_array.objects.push_back(output_object);

    //visulization the rectangle four corner points
    pcl::PointXYZI rec_corner_pt;
    for (int i = 0; i < 4; i++)
    {
      rec_corner_pt.x = rec_corner_points[i].x;
      rec_corner_pt.y = rec_corner_points[i].y;
      rec_corner_pt.z = output_object.pose.position.z + 0.5 * output_object.dimensions.z;
      rec_corner_pt.intensity = intensity_mark;
      corner_points_visual->push_back(rec_corner_pt);
    }
    intensity_mark++;
  }

  out_object_array.header = bbox_header;
  pub_autoware_bboxs_array_.publish(out_object_array);

  jsk_bbox_array.header = bbox_header;
  pub_jsk_bboxs_array_.publish(jsk_bbox_array);
  
  ROS_INFO("out_object_array.header.frame_id is %s", out_object_array.header.frame_id.c_str());

  //visulization the detail of ransac
  pub_corner_point_.publish(corner_pcl);
  pub_left_side_point_.publish(left_side_pcl);
  pub_right_side_point_.publish(right_side_pcl);

  // rectangle corner points
  sensor_msgs::PointCloud2 corner_points_visual_ros;
  pcl::toROSMsg(*corner_points_visual, corner_points_visual_ros);
  corner_points_visual_ros.header = in_cluster_array.header;
  pub_rec_corner_points_.publish(corner_points_visual_ros);


  // Time the whole process
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "\033[1;36m [RANSAC L-shape fitting] took " << elapsed_time.count() << " milliseconds" << "\033[0m" << std::endl;
}