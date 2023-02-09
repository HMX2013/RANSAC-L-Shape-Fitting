#include <random>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_datatypes.h>
#include <chrono>
#include "ransac_Lshape_fitting.h"
#include "ransac_lib/ransac.h"
#include "ransac_lib/line_estimator.h"

LShapeFilter::LShapeFilter() : tf2_listener(tf2_buffer)
{
  // L-shape fitting params
  ros::NodeHandle private_nh_("~");

  private_nh_.param<std::string>("input_cluster_topic", input_cluster_topic_, "/kitti3d/cluster_array");
  ROS_INFO("Input_cluster_topic: %s", input_cluster_topic_.c_str());

  node_handle_.param<std::string>("output_bbox_topic_", output_bbox_topic_, "/ransacLshape_fitting/visual_jsk_bboxs");
  ROS_INFO("output_bbox_topic_: %s", output_bbox_topic_.c_str());

  private_nh_.param<std::string>("bbox_target_frame", bbox_target_frame_, "map");
  ROS_INFO("[%s] bounding box's target frame is: %s", __APP_NAME__, bbox_target_frame_);

  node_handle_.param<std::string>("corner_point_topic_", corner_point_topic_, "/ransacLshape_fitting/closest_point_cloud");
  ROS_INFO("corner_point_topic_: %s", corner_point_topic_.c_str());

  float resolution = (float)(MAX_VERT_ANGLE - MIN_VERT_ANGLE) / (float)(VERT_SCAN - 1);
  for (int i = 0; i < VERT_SCAN; i++)
    vert_angles_.push_back(MIN_VERT_ANGLE + i * resolution);

  bias_vector.clear();
  runtime_vector.clear();
  first_appear = true;

  sub_object_array_ = node_handle_.subscribe(input_cluster_topic_, 1, &LShapeFilter::MainLoop, this);
  pub_corner_point_ = node_handle_.advertise<sensor_msgs::PointCloud2>(corner_point_topic_,1);

  pub_left_side_point_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/ransacLshape_fitting/left_side_point", 1);
  pub_right_side_point_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/ransacLshape_fitting/right_side_point", 1);
  pub_ransac_line_left_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/ransacLshape_fitting/ransac_line_left", 1);
  pub_ransac_line_right_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/ransacLshape_fitting/ransac_line_right", 1);

  pub_autoware_bboxs_array_ = node_handle_.advertise<obsdet_msgs::DetectedObjectArray>("/ransacLshape_fitting/autoware_bboxs_array", 1);
  pub_jsk_bboxs_array_ = node_handle_.advertise<jsk_recognition_msgs::BoundingBoxArray>("/ransacLshape_fitting/jsk_bboxs_array",1);

  pub_rec_corner_points_ = node_handle_.advertise<sensor_msgs::PointCloud2>("/ransacLshape_fitting/rec_corner_points", 1);
  
  pub_jskrviz_time_ = node_handle_.advertise<std_msgs::Float32>("/time_spent", 1);
}

bool LShapeFilter::ransac_Lshape_core(pcl::PointCloud<pcl::PointXYZ> &cluster, double &theta_star,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr &left_side_pcl, pcl::PointCloud<pcl::PointXYZI>::Ptr &right_side_pcl,
                                      pcl::PointXYZI &corner_point, pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_left,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr &c_line_right, std::vector<cv::Point2f> &rec_corner_points)
{
  left_side_pcl->clear();
  right_side_pcl->clear();

  int num_points = cluster.size();

  // Searching the corner point
  float min_dist = std::numeric_limits<float>::max();
  
  int corner_index;
  
  pcl::PointXYZI side_point;

  for (int i_point = 0; i_point < num_points; i_point++)
  {
    const float p_x = cluster[i_point].x;
    const float p_y = cluster[i_point].y;
    
    float distance_o = p_x * p_x + p_y * p_y;

    if (distance_o < min_dist){
      min_dist = distance_o;
      corner_index = i_point;
    }

    if (min_dist == std::numeric_limits<float>::max()){
      continue;
    }
  }

  corner_point.x = cluster[corner_index].x;
  corner_point.y = cluster[corner_index].y;
  corner_point.z = cluster[corner_index].z;
  corner_point.intensity = min_dist;

  // divide these point into sets A and B by column
  size_t columnIdn, column_corner;

  column_corner = getColIdx(cluster.points[corner_index]);

  if (column_corner < 0 || column_corner >= HORZ_SCAN)
    return false;

  for (int index = 0; index < num_points; index++)
  {
    if (index == corner_index)
      continue;

    columnIdn = getColIdx(cluster.points[index]);

    if (columnIdn < 0 || columnIdn >= HORZ_SCAN)
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

  // using ransac lib to fitting
  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 20u;
  options.max_num_iterations_ = 2000u;
  options.squared_inlier_threshold_ = 0.2 * 0.2;
  options.num_lo_steps_ = 10u;

  std::random_device rand_dev;
  options.random_seed_ = rand_dev();

  ransac_lib::LocallyOptimizedMSAC<Eigen::Vector3d, std::vector<Eigen::Vector3d>, ransac_lib::LineEstimator> lomsac;

  ransac_lib::RansacStatistics ransac_stats;
  Eigen::Vector3d best_model_l, best_model_r, best_model_final;
  Eigen::Vector3d best_model_Lshape;
  Eigen::Matrix2Xd data;

  bool perform_ransac_left = false;
  bool perform_ransac_right = false;

  int num_ransac_inliers_l = 0;
  int num_ransac_inliers_r = 0;

  if (left_side_pcl->size() >= 10)
  {
    data.resize(2, left_side_pcl->size());
    for (int i = 0; i < left_side_pcl->size(); i++)
    {
      data.col(i)[0]=left_side_pcl->points[i].x;
      data.col(i)[1]=left_side_pcl->points[i].y;
    }
    
    ransac_lib::LineEstimator solver(data);

    num_ransac_inliers_l = lomsac.EstimateModel(options, solver, &best_model_l, &ransac_stats);

    // std::cout << "   ...left LOMSAC found " << num_ransac_inliers_l << " inliers in "
    //           << ransac_stats.num_iterations << " iterations with an inlier "
    //           << "ratio of " << ransac_stats.inlier_ratio << std::endl;

    perform_ransac_left = true;
  }

  if (right_side_pcl->size() >= 10)
  {
    data.resize(2, right_side_pcl->size());
    for (int i = 0; i < right_side_pcl->size(); i++)
    {
      data.col(i)[0]=right_side_pcl->points[i].x;
      data.col(i)[1]=right_side_pcl->points[i].y;
    }
    ransac_lib::LineEstimator solver(data);

    num_ransac_inliers_r = lomsac.EstimateModel(options, solver, &best_model_r, &ransac_stats);
    // std::cout << "   ...right LOMSAC found " << num_ransac_inliers_r << " inliers in "
    //           << ransac_stats.num_iterations << " iterations with an inlier "
    //           << "ratio of " << ransac_stats.inlier_ratio << std::endl;

    perform_ransac_right = true;
  }

  // std::cout << "perform_ransac_left =" << perform_ransac_left << std::endl;
  // std::cout << "perform_ransac_right =" << perform_ransac_right << std::endl;

  bool has_calcu_theta = false;

  // fitting zero line
  if (perform_ransac_left == false && perform_ransac_right == false)
    return false;

  // fitting one line
  if (perform_ransac_left == false || perform_ransac_right == false)
  {
    if (perform_ransac_left == true)
      best_model_final = best_model_l;
    else
      best_model_final = best_model_r;

    theta_star = atan(-best_model_final[0] / best_model_final[1]);

    theta_trans(theta_star);

    has_calcu_theta = true;
  }

  // fitting two line
  if (has_calcu_theta == false)
  {
    double k1 = -best_model_l[0] / best_model_l[1];
    double k2 = -best_model_r[0] / best_model_r[1];
    double inters_angle = atan(abs(k1 - k2) / abs(1 + k1 * k2)) * 180 / M_PI;

    if (inters_angle < 20)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_ptr(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::copyPointCloud(cluster, *cluster_ptr);

      data.resize(2, cluster_ptr->size());
      for (int i = 0; i < cluster_ptr->size(); i++)
      {
        data.col(i)[0] = cluster_ptr->points[i].x;
        data.col(i)[1] = cluster_ptr->points[i].y;
      }
      ransac_lib::LineEstimator solver(data);

      int num_ransac_inliers_final = lomsac.EstimateModel(options, solver, &best_model_final, &ransac_stats);
    }
    else
    {
      if (num_ransac_inliers_l > num_ransac_inliers_r)
      {
        best_model_final = best_model_l;
      }
      else
      {
        best_model_final = best_model_r;
      }
    }

    theta_star = atan(-best_model_final[0] / best_model_final[1]);

    theta_trans(theta_star);

    has_calcu_theta = true;
  }

  if (!has_calcu_theta)
    theta_star = 0;

  return true;
}

void LShapeFilter::theta_trans(double &theta_atan)
{
  if (theta_atan < 0)
    theta_atan = theta_atan + M_PI / 2;
}

void LShapeFilter::calculateDimPos(double &theta_star, obsdet_msgs::DetectedObject &output, std::vector<cv::Point2f>& rec_corner_points, pcl::PointCloud<pcl::PointXYZ> &cluster)
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

  output.dimensions.x = std::max(output.dimensions.x, ep);
  output.dimensions.y = std::max(output.dimensions.y, ep);
}

jsk_recognition_msgs::BoundingBox LShapeFilter::jsk_bbox_transform(const obsdet_msgs::DetectedObject &autoware_bbox, 
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

int LShapeFilter::getRowIdx(pcl::PointXYZ pt)
{
  float angle = atan2(pt.z, sqrt(pt.x * pt.x + pt.y * pt.y)) * 180 / M_PI;
  auto iter_geq = std::lower_bound(vert_angles_.begin(), vert_angles_.end(), angle);
  int row_idx;

  if (iter_geq == vert_angles_.begin())
  {
    row_idx = 0;
  }
  else
  {
    float a = *(iter_geq - 1);
    float b = *(iter_geq);
    if (fabs(angle - a) < fabs(angle - b))
    {
      row_idx = iter_geq - vert_angles_.begin() - 1;
    }
    else
    {
      row_idx = iter_geq - vert_angles_.begin();
    }
  }
  return row_idx;
}

int LShapeFilter::getColIdx(pcl::PointXYZ pt)
{
  float horizonAngle = atan2(pt.x, pt.y) * 180 / M_PI;
  static float ang_res_x = 360.0 / float(HORZ_SCAN);
  int col_idx = -round((horizonAngle - 90.0) / ang_res_x) + HORZ_SCAN / 2;
  if (col_idx >= HORZ_SCAN)
    col_idx -= HORZ_SCAN;
  return col_idx;
}

void LShapeFilter::eval_running_time(int running_time)
{
  double runtime_std;
  double runtime_sqr_sum = 0.0;
  double runtime_aver;

  runtime_vector.push_back(running_time);

  double runtime_total_v = 0.0;

  for (size_t i = 0; i < runtime_vector.size(); i++)
  {
    runtime_total_v += runtime_vector[i];
  }

  runtime_aver = runtime_total_v / runtime_vector.size();

  for (size_t i = 0; i < runtime_vector.size(); i++)
  {
    runtime_sqr_sum += (runtime_vector[i] - runtime_aver) * (runtime_vector[i] - runtime_aver);
  }

  runtime_std = sqrt(runtime_sqr_sum / runtime_vector.size());

  std::cout << "runtime_vector.size() is = " << runtime_vector.size() << std::endl;
  std::cout << "running_time is = " << running_time / 1000.0 << std::endl;
  std::cout << "runtime_aver is = " << runtime_aver / 1000.0 << std::endl;
  std::cout << "runtime_std is = " << runtime_std / 1000.0 << std::endl;
  std::cout << "---------------------------------" << std::endl;
}

void LShapeFilter::eval_performance(double &theta_kitti, double &theta_optim, const uint32_t &index, const uint32_t &index_seq)
{
  double bias_org = abs(theta_kitti - theta_optim);

  double bias = std::min(bias_org, M_PI / 2 - bias_org);

  double bias_std;
  double bias_sqr_sum = 0.0;
  double aver_accu;

  bias_vector.push_back(bias);

  double bias_total_v = 0.0;

  for (size_t i = 0; i < bias_vector.size(); i++)
  {
    bias_total_v += bias_vector[i];
  }
  aver_accu = bias_total_v / bias_vector.size();

  for (size_t i = 0; i < bias_vector.size(); i++)
  {
    bias_sqr_sum += (bias_vector[i] - aver_accu) * (bias_vector[i] - aver_accu);
  }

  bias_std = sqrt(bias_sqr_sum / bias_vector.size());

  std::cout << "index is = " << index << std::endl;
  std::cout << "index_seq is = " << index_seq << std::endl;
  std::cout << "theta_kitti is = " << theta_kitti << std::endl;
  std::cout << "theta_optim is = " << theta_optim << std::endl;

  std::cout << "bias_total_v is = " << bias_total_v * 180 / M_PI << std::endl;
  std::cout << "bias_vector[] is = " << bias_vector[bias_vector.size() - 1] << std::endl;
  std::cout << "bias_vector size is = " << bias_vector.size() << std::endl;
  std::cout << "bias_std is = " << bias_std * 180 / M_PI << std::endl;
  std::cout << "aver_accy is = " << aver_accu * 180 / M_PI << std::endl;
  std::cout << "bias is = " << bias * 180 / M_PI << std::endl;
  std::cout << "---------------------------------" << std::endl;

  std::string filename = "/home/dnn/paper_pose_est/ransac_Lshape_fitting/src/ransac_lshape_fitting/evaluation/bias.txt";
  if (boost::filesystem::exists(filename) && first_appear)
  {
    boost::filesystem::remove(filename);
    first_appear = false;
  }

  std::ofstream out_txt(filename, std::ios::app);
  if (bias * 180 / M_PI > 16.0)
    out_txt << index << " " << index_seq << " " << bias * 180 / M_PI << std::endl;
  out_txt.close();
}

void LShapeFilter::MainLoop(const obsdet_msgs::CloudClusterArray& in_cluster_array)
{
  start_time = std::chrono::system_clock::now();
  double theta_star;
  obsdet_msgs::DetectedObjectArray out_object_array;

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

  /*-------------------------------------------------------------------------------------------*/
  for (const auto &in_cluster : in_cluster_array.clusters)
  {
    pcl::PointCloud<pcl::PointXYZ> cluster;
    pcl::fromROSMsg(in_cluster.cloud, cluster);

    obsdet_msgs::DetectedObject output_object;
    std::vector<cv::Point2f> rec_corner_points(4);

    // double theta_star;
    pcl::PointXYZI corner_point;

    pcl::PointCloud<pcl::PointXYZI>::Ptr c_line_left(new pcl::PointCloud<pcl::PointXYZI>);
    c_line_left->header.frame_id = in_cluster_array.header.frame_id;

    pcl::PointCloud<pcl::PointXYZI>::Ptr c_line_right(new pcl::PointCloud<pcl::PointXYZI>);
    c_line_right->header.frame_id = in_cluster_array.header.frame_id;

    if (cluster.size() < 10)
      return;

    /*-----------------------------------------------------------------------------------------*/
    const auto eval_start_time = std::chrono::system_clock::now();

    bool fitting_success = ransac_Lshape_core(cluster, theta_star, left_side_pcl, right_side_pcl, corner_point, c_line_left, c_line_right, rec_corner_points);

    if (!fitting_success)
      return;

    const auto eval_end_time = std::chrono::system_clock::now();
    const auto eval_exe_time = std::chrono::duration_cast<std::chrono::microseconds>(eval_end_time - eval_start_time);

    eval_running_time(eval_exe_time.count());


    double theta_kitti = in_cluster.orientation;

    eval_performance(theta_kitti, theta_star, in_cluster.index, in_cluster.index_seq);
    /*-----------------------------------------------------------------------------------------*/
    corner_pcl->points.push_back(corner_point);

    calculateDimPos(theta_star, output_object, rec_corner_points, cluster);

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
  /*---------------------------------------------------------------------------------------------------*/

  out_object_array.header = bbox_header;
  pub_autoware_bboxs_array_.publish(out_object_array);

  jsk_bbox_array.header = bbox_header;
  pub_jsk_bboxs_array_.publish(jsk_bbox_array);

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
  // const auto end_time = std::chrono::steady_clock::now();
  // const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "\033[1;36m [RANSAC L-shape fitting] took " << elapsed_time.count() << " milliseconds" << "\033[0m" << std::endl;

  end_time = std::chrono::system_clock::now();
  exe_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
  time_spent.data = exe_time;
  pub_jskrviz_time_.publish(time_spent);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ransac_lshape_fitting_node");
  LShapeFilter app;
  ros::spin();

  return 0;
}