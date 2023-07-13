#ifndef GROUND_PROCESSOR_H
#define GROUND_PROCESSOR_H

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <stdlib.h>
#include <time.h> 
#include <thread>
#include <mutex>  
#include <ros/ros.h>
#include <ros/console.h>
//PCL
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>//自定义点云类型时要加
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/extract_indices.h>
//Eigen
#include <Eigen/Dense>
//ROS
#include <ros/ros.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>
//src
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "ground_remove2.h"

#define Pi 3.1415926

using namespace std;
using namespace Eigen;

enum Filter {None, SG, Gaussian};

class CloudProcessor
{
private:
	pcl::PointCloud<pcl::PointR> cloud_input, cloud_ng;
	int input_cloud_size, col_num, row_num;
	double vertical_deg, horizontal_deg;
	double range_min = 0.15;
	double range_max = 50;
	vector<float> sines_vec, cosines_vec, horizontal_angles, vertical_angles;
	Filter filter_type{SG};
	int window_size{5};
	double start_angle_thresh{45.0 / 180.0 * Pi};
	int step_row{2}, step_col{1};
	double ground_angle_thresh{5.0 / 180.0 * Pi};

public:
	cv::Mat range_mat, angle_mat, smoothed_mat, no_ground_image, label_mat, dilated;
	pcl::PointCloud<pcl::PointR> cloud_output;

public:
	void setCloudInput(pcl::PointCloud<pcl::PointR> cloud_input_);
	void reset();
	void toRangeImage();
	void processCloud();
	void toAngleImage();
	void getSinCosVec();
	void ApplySavitskyGolaySmoothing() ;
	cv::Mat GetSavitskyGolayKernel() const;
	void LabelPixel(uint16_t label, int row_id, int col_id);
	void GetNeighbors(queue<pair<int, int>>& labeling_queue, const int& cur_row, const int& cur_col);
	cv::Mat GetUniformKernel(int window_size, int type) const;
	void ZeroOutGroundCeillingBFS();
	void toCloud();
	void UnprojectPoint(const cv::Mat image, int row, int col, pcl::PointR& pt);

};




#endif
