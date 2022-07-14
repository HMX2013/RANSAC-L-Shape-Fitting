#include "ransac_Lshape_fitting.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ransac_lshape_fitting_node");
  LShapeFilter app;
  ros::spin();

  return 0;
}