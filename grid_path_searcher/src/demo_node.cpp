#include <iostream>
#include <fstream>
#include <math.h>
#include <unordered_map>
#include <queue>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/io.h>
#include <pcl/PolygonMesh.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <random>
#include <tuple>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <tf/transform_broadcaster.h>
#include "ground_remove2.h"
#include "cloud_processor.h"


#define logit(x) (log((x) / (1 - (x))))

/*typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImageOdom;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImageOdom;
typedef message_filters::Synchronizer<SyncPolicyImageOdom>* SynchronizerImageOdom;*/

using namespace std;
using namespace Eigen;
const double PI = 3.1415926;
// pointcloud segmentation: wall and ground detection
struct Range {
	double range_xy;
	double range_zxy;
	int ring_i;
	int frame_j;
	int count_num;
};

CloudProcessor cloud_processor;
string frame_id_;
double height_thresh, range_min, range_max;
ros::Publisher ngc_pub, g_pub, c_pub;
ros::Subscriber odom_sub_, depth_sub_, cloud_sub_;
ros::Timer vis_timer_;
//pcl::PointCloud<pcl::PointXYZI> cloud_msg;
pcl::PointCloud<pcl::PointR> cloud_msg;
pcl::PointCloud<pcl::PointXYZITR> cloud_g, cloud_nc, cloud_c, cloud_ngc;
unordered_map<int, Range> range_image;
ros::Time time_begin; 
int idx_begin = 0;
int total_frame = 0;
vector<int> cluster_index;
bool have_pcl = false;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("pcd")); //PCLVisualizer 可视化类

// camera position and pose data
Eigen::Vector3d camera_pos_, last_camera_pos_;
Eigen::Quaterniond camera_q_, last_camera_q_;
Eigen::Vector3d d435_pos_;

// depth image data
cv::Mat depth_image_, last_depth_image_;
int image_cnt_;

//function declaration
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& point_msg);
void depthOdomCallback(const sensor_msgs::ImageConstPtr& img,
                               const geometry_msgs::PoseStampedConstPtr& odom);
void visCallback(const ros::TimerEvent& /*event*/);
                                

double Polar_angle_cal(double x, double y) {
	double temp_tangle = 0;
	if (x == 0 && y == 0) {
		temp_tangle = 0;
	} else if (y >= 0) {
		temp_tangle = (double) atan2(y, x);
	} else if (y <= 0) {
		temp_tangle = (double) atan2(y, x) + 2 * PI;
	}
	return temp_tangle;
}
//HSV转rgb
vector<double> hsv2rgb(vector<double>& hsv) {
	vector<double> rgb(3);
	double R, G, B, H, S, V;
	H = hsv[0];
	S = hsv[1];
	V = hsv[2];
	if (S == 0) {
		rgb[0] = rgb[1] = rgb[2] = V;
	} else {

		int i = int(H * 6);
		double f = (H * 6) - i;
		double a = V * (1 - S);
		double b = V * (1 - S * f);
		double c = V * (1 - S * (1 - f));
		i = i % 6;
		switch (i) {
		case 0: {
			rgb[0] = V;
			rgb[1] = c;
			rgb[2] = a;
			break;
		}
		case 1: {
			rgb[0] = b;
			rgb[1] = V;
			rgb[2] = a;
			break;
		}
		case 2: {
			rgb[0] = a;
			rgb[1] = V;
			rgb[2] = c;
			break;
		}
		case 3: {
			rgb[0] = a;
			rgb[1] = b;
			rgb[2] = V;
			break;
		}
		case 4: {
			rgb[0] = c;
			rgb[1] = a;
			rgb[2] = V;
			break;
		}
		case 5: {
			rgb[0] = V;
			rgb[1] = a;
			rgb[2] = b;
			break;
		}
		}
	}

	return rgb;
}

//可视化

template<typename T> string toString(const T& t) {
	ostringstream oss;
	oss << t;
	return oss.str();
}



template<typename PointInT>
double CalculateRangeXY(const PointInT pointIn) {

	return sqrt(pointIn.x * pointIn.x + pointIn.y * pointIn.y);
}

template<typename PointInT>
double CalculateRangeZXY(const PointInT pointIn) {

	return sqrt(
			pointIn.x * pointIn.x + pointIn.y * pointIn.y
					+ (pointIn.z) * (pointIn.z));
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "demo_node");
    ros::NodeHandle nh("~");
    time_begin = ros::Time::now();

    message_filters::Subscriber<sensor_msgs::Image> depth_sub_(nh, "/camera/depth/image_rect_raw", 50);
    cloud_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, pointCloudCallback);
    //message_filters::Subscriber<nav_msgs::Odometry> odom_sub_(nh, "/t265/odom/sample", 100);
    /*message_filters::Subscriber<geometry_msgs::PoseStamped> odom_sub_(nh, "/mavros/vision_pose/pose", 100);
    message_filters::Synchronizer<SyncPolicyImageOdom> sync_image_odom_(SyncPolicyImageOdom(100), depth_sub_, odom_sub_);
    sync_image_odom_.registerCallback(boost::bind(depthOdomCallback, _1, _2));*/

    vis_timer_ = nh.createTimer(ros::Duration(0.05), visCallback); 

    ngc_pub = nh.advertise<sensor_msgs::PointCloud2>("/trolley/lidar/no_ground_ceilling", 10);
    g_pub = nh.advertise<sensor_msgs::PointCloud2>("/trolley/lidar/ground", 10);
    c_pub = nh.advertise<sensor_msgs::PointCloud2>("/trolley/lidar/ceilling", 10);

    nh.param("layout/height_thresh", height_thresh, 0.2);
    nh.param("layout/frame_id", frame_id_, string("world"));
    nh.param("layout/range_min", range_min, 0.1);
    nh.param("layout/range_max", range_max, 200.0);

	
	viewer->setBackgroundColor(0.8, 0.8, 0.8);
	viewer->addCoordinateSystem(1);

    ros::Rate rate(100);
    bool status = ros::ok();



    while (status)
    {    
        ros::spinOnce();
        status = ros::ok();
        rate.sleep();

		if (!viewer->wasStopped())
			viewer->spinOnce();;
    }

    return 0;
}

void CloudFilter(const pcl::PointCloud<pcl::PointXYZITR>& cloudIn,
		pcl::PointCloud<pcl::PointXYZITR>& cloudOut, double x_min, double x_max,
		double y_min, double y_max, double z_min, double z_max) {
  //TODO
  cloudOut = cloudIn;
  return;

	cloudOut.header = cloudIn.header;
	//cloudOut.sensor_orientation_ = cloudIn.sensor_orientation_;
	//cloudOut.sensor_origin_ = cloudIn.sensor_origin_;
	cloudOut.points.clear();
	//1) set parameters for removing cloud reflect on ego vehicle
	double x_limit_min = -1.8, x_limit_max = 1.8, y_limit_forward = 5.0,
			y_limit_backward = -4.5;
	//2 apply the filter
	for (int i = 0; i < cloudIn.size(); ++i) {
		double x = cloudIn.points[i].x;
		double y = cloudIn.points[i].y;
		double z = cloudIn.points[i].z;
		// whether on ego vehicle
		if ((x > x_limit_min && x < x_limit_max && y > y_limit_backward
				&& y < y_limit_forward))
			continue;
		if ((x > x_min && x < x_max && y > y_min && y < y_max && z > z_min
				&& z < z_max)) {

			cloudOut.points.push_back(cloudIn.points[i]);
		}
	}
}

void transform2RangeImage(const pcl::PointCloud<pcl::PointR>& cloudIn,
		pcl::PointCloud<pcl::PointXYZITR>& ngc_cloudOut, pcl::PointCloud<pcl::PointXYZITR>& g_cloudOut, 
      pcl::PointCloud<pcl::PointXYZITR>& c_cloudOut, unordered_map<int, Range> &unordered_map_out) 
{
  total_frame = (int)(cloudIn.points.size() / 16);
	pcl::PointCloud<pcl::PointXYZITR>::Ptr cloud2range(new pcl::PointCloud<pcl::PointXYZITR>);
	pcl::PointCloud<pcl::PointXYZITR>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZITR>);
  cloud2range->points.resize(cloudIn.points.size());

  double max_h{-100.0}, min_h{100.0}; 
  for(const auto pt : cloudIn.points)
  {
    int row_i = pt.ring;
    double h_angle = 180 / PI * std::atan2(pt.y, pt.x) + 180;
    int col_i = ceil(h_angle / 360 * total_frame) - 1;  

    if(row_i < 0 || row_i >= 16 || col_i < 0 || col_i >= total_frame) {
      ROS_WARN("ROW_COL_IDX_ERROR");
      continue;
    }

    cloud2range->points[col_i * 16 + row_i].x = pt.x ;
    cloud2range->points[col_i * 16 + row_i].y = pt.y ;
    cloud2range->points[col_i * 16 + row_i].z = pt.z ;
    cloud2range->points[col_i * 16 + row_i].intensity = pt.intensity;
    cloud2range->points[col_i * 16 + row_i].ring = row_i;
    cloud2range->points[col_i * 16 + row_i].pcaketnum = col_i;

    if(pt.z > max_h) {max_h = pt.z;}
    if(pt.z < min_h) {min_h = pt.z;}
    //cout<<"ring="<<row_i<<" col="<<col_i<<" x="<<pt.x<<" y="<<pt.y<<" z="<<pt.z<<" max_h="<<max_h<<" min_h="<<min_h<<endl;
  }

/*
	for (int j = 0; j < total_frame; ++j) {
		int num_odd = 8;                //基数从8位开始排
		int num_even = 0;                //偶数从头
		for (int i = 0; i < 16; ++i) {
			if (double(i % 2) == 0.0) {
				cloud2range->points[j * 16 + i].x = cloudIn.points[j * 16 + i].x ;
				cloud2range->points[j * 16 + i].y = cloudIn.points[j * 16 + i].y ;
				cloud2range->points[j * 16 + i].z = cloudIn.points[j * 16 + i].z ;
        cloud2range->points[j * 16 + i].intensity =cloudIn.points[j * 16 + i].intensity;
				cloud2range->points[j * 16 + i].ring =num_even;
				cloud2range->points[j * 16 + i].pcaketnum =j;
				num_even++;
			} else {
				cloud2range->points[j * 16 + i].x = cloudIn.points[j * 16 + i].x;
				cloud2range->points[j * 16 + i].y = cloudIn.points[j * 16 + i].y ;
				cloud2range->points[j * 16 + i].z = cloudIn.points[j * 16 + i].z ;
        cloud2range->points[j * 16 + i].intensity =cloudIn.points[j * 16 + i].intensity;
				cloud2range->points[j * 16 + i].ring =num_odd;
				cloud2range->points[j * 16 + i].pcaketnum =j;
				num_odd++;
			}
		} //按索引顺序排列
  }*/

	cloud2range->height = 1;
	cloud2range->width = cloud2range->points.size();


	double xmin = -350, xmax = 350, ymin = -300, ymax = 300, zmin = -100, zmax = 300;
  CloudFilter(*cloud2range, *cloud_filtered, xmin, xmax, ymin, ymax, zmin, zmax);

  //PCA ground & ceilling removal
  //TODO:better fitting
  ros::Time time1 = ros::Time::now(); 
  GroundRemove2 ground_remove(3, 20, 1.0, 0.15);
  ground_remove.RemoveGround2(*cloud_filtered, g_cloudOut, c_cloudOut, ngc_cloudOut);
  //ROS_INFO("%f ms to remove ground and ceilling",(ros::Time::now() - time1).toSec() * 1000);
  

  for(int i = 0; i < ngc_cloudOut.points.size(); ++i){

    double x = ngc_cloudOut.points[i].x;
    double y = ngc_cloudOut.points[i].y;
    double z = ngc_cloudOut.points[i].z;

    double distance = CalculateRangeXY( ngc_cloudOut.points[i]);
    int ringnum = ngc_cloudOut.points[i].ring;
    int image_index = ngc_cloudOut.points[i].pcaketnum * 16 + (15 - ringnum);

    Range r;
    r.ring_i = 15 -ringnum;
    r.frame_j = ngc_cloudOut.points[i].pcaketnum;
    r.count_num = i;
    r.range_xy = ngc_cloudOut.points[i].z;
    r.range_zxy = CalculateRangeZXY(ngc_cloudOut.points[i]);
    unordered_map_out.insert(make_pair(image_index, r));
  }
}

void find_neighbors(int Ix, int Iy, vector<int>& neighborudindex,
		vector<int>& neighborlrindex) {
	//cout<<"silinyu " << Ix<< " "<<Iy<<endl;
	for (int x = Ix - 2; x <= Ix + 2; ++x) {
		//cout<<"DDDDDDw "<<x<<endl;
		if (x == Ix)
			continue;
		int px = x;
		if (x < 0) {
			px = total_frame-1;
                        //cout<<"LL"<<endl;
		}
		if (x > total_frame-1) {
			px = 0;
                       // cout<<"RR"<<endl;
		}
		//cout<<px << " "<<Iy<<endl;
		neighborlrindex.push_back(px * 16 + Iy); 
	}

	for (int y = Iy - 1; y <= Iy + 1; ++y) {
		if (y == Iy)
			continue;
		if (y < 0 || y > 31)
			continue;
		//cout<<Ix << " "<<y<<endl;
		neighborudindex.push_back((Ix * 16 + y)); 
	}
	have_pcl = true;
}

bool compare_index(pair<int, Range> a, pair<int, Range> b) {
	return a.first < b.first;
} //升序

void range_cluster(unordered_map<int, Range> &unordered_map_in, double ringnum, vector<int>& cluster_vec) {
	int current_cluster = 0;
	vector<int> cluster_indices = vector<int>(total_frame * 16, -1);
	double horizon_angle_resolution = 360.0 / double(total_frame) * PI / 180;
	double vertical_angle_resolution = 2.0 * PI / 180.0;

  vector<pair<int, Range>> tr(unordered_map_in.begin(), unordered_map_in.end());
  sort(tr.begin(), tr.end(), compare_index);

	double theta_thresh = 3 * PI / 180;
	double theta_thresh2 = 30 * PI / 180;

	for (int i = 0; i < tr.size(); ++i) {
			unordered_map<int, Range>::iterator it_find;
			it_find = unordered_map_in.find(tr[i].first);

			if (it_find != unordered_map_in.end() && cluster_indices[tr[i].first] == -1) {
				queue<vector<int>> q;
				vector<int> indexxy(2);
				indexxy[0] = it_find->second.frame_j;
				indexxy[1] = it_find->second.ring_i;
				q.push(indexxy);
				while (q.size()>0) {
					if (cluster_indices[q.front()[0] * 16 + q.front()[1]]	!= -1) {
					  q.pop();
						continue;
					}
 
					cluster_indices[q.front()[0] * 16 + q.front()[1]] =	current_cluster;
					vector<int> neighborudid;
					vector<int> neighborlfid;
					unordered_map<int, Range>::iterator it_findo;
					it_findo = unordered_map_in.find(q.front()[0] * 16 + q.front()[1]);
					find_neighbors(q.front()[0], q.front()[1], neighborudid, neighborlfid);

					if (neighborudid.size() > 0) {
						for (int in = 0; in < neighborudid.size(); ++in) {
							unordered_map<int, Range>::iterator it_findn;
							it_findn = unordered_map_in.find(neighborudid[in]);
							if (it_findn != unordered_map_in.end()) {
								double d1 = max(it_findo->second.range_zxy,
										it_findn->second.range_zxy);
								double d2 = min(it_findo->second.range_zxy,
										it_findn->second.range_zxy);


								double angle = fabs((double) atan2(d2* sin(vertical_angle_resolution),d1- d2* cos(vertical_angle_resolution)));
                double dmax = (it_findo->second.range_zxy) * sin(1.33*PI/180)/sin(50*PI/180 -1.33*PI/180) + 3*0.2;
				//cout<<"angle1="<<angle * 180 / PI<<" deltad="<<fabs(d2-d1)<<" dmax1="<<dmax<<endl;

							  if (it_findo->second.range_xy>1.2 && fabs(d2-d1) < dmax) {
									vector<int> indexxy(2);
									indexxy[0] = it_findn->second.frame_j;
									indexxy[1] = it_findn->second.ring_i;
									q.push(indexxy);
								}else if (angle > theta_thresh2) {
                //if (angle > theta_thresh2) {
									vector<int> indexxy(2);
									indexxy[0] = it_findn->second.frame_j;
									indexxy[1] = it_findn->second.ring_i;
									q.push(indexxy);
								}
							}
						}
					}

					if (neighborlfid.size() > 0) {
						for (int in = 0; in < neighborlfid.size(); ++in) {
							unordered_map<int, Range>::iterator it_findn;
							it_findn = unordered_map_in.find(neighborlfid[in]);
							if (it_findn != unordered_map_in.end()) {
								double d1 = max(it_findo->second.range_zxy,
										it_findn->second.range_zxy);
								double d2 = min(it_findo->second.range_zxy,
										it_findn->second.range_zxy);

								double angle = fabs((double) atan2(d2* sin(horizon_angle_resolution),d1- d2* cos(horizon_angle_resolution)));
                                double dmax = (it_findo->second.range_zxy) * sin(360/ringnum*PI/180)/sin(30*PI/180 -360/ringnum*PI/180) + 3*0.2;
								//cout<<"angle2="<<angle * 180 / PI<<" deltad="<<fabs(d2-d1)<<" dmax2="<<dmax<<endl;
								//cout<<"d1="<<d1<<" d2="<<d2<<" horizon_angle_resolution="<<horizon_angle_resolution / PI * 180<<" total_frame="<<total_frame<<endl;


							        //if (fabs(it_findo->second.range_zxy-it_findn->second.range_zxy) < dmax) {
								if (angle > theta_thresh) {
									//cluster_indices[neighborlfid[in]] =
											//current_cluster;
									vector<int> indexxy(2);
									indexxy[0] = it_findn->second.frame_j;
									indexxy[1] = it_findn->second.ring_i;
                                                                        //cout<<"LLL2 "<<indexxy[0]<<" "<<indexxy[1]<<endl;
									q.push(indexxy);
								}
							}
						}
					}
					q.pop();
				}
				current_cluster++;
			}

	}
	cluster_vec = cluster_indices;
}

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& point_msg)
{
	time_begin = ros::Time::now();

	pcl::fromROSMsg(*point_msg, cloud_msg);
	cloud_msg.header.frame_id = "map";
	
	cloud_processor->reset();
	cloud_processor.setCloudInput(cloud_msg);
	cloud_processor.processCloud();






	//remove ground and ceilling use plane fitting & to hash range image
	cloud_ngc.clear(); 
	cloud_g.clear();
	cloud_c.clear();
	range_image.clear();
	transform2RangeImage(cloud_msg, cloud_ngc, cloud_g, cloud_c, range_image);
	//cout<<"total="<<cloud_msg.points.size()<<" ground pcl size="<<cloud_g.points.size()
		//<<" ceilling pcl size="<<cloud_c.points.size()<<" left size="<<cloud_ngc.points.size()<<endl;

	cloud_ngc.header.frame_id = "map";
	cloud_ngc.height = 1;
	cloud_ngc.width = cloud_ngc.points.size();
	cloud_ngc.is_dense = false;
	cloud_g.header.frame_id = "map";
	cloud_g.height = 1;
	cloud_g.width = cloud_g.points.size();
	cloud_g.is_dense = false;
	cloud_c.header.frame_id = "map";
	cloud_c.height = 1;
	cloud_c.width = cloud_c.points.size();
	cloud_c.is_dense = false;

	double ringnum=(cloud_msg.points.size() / 16) - 1;
	cluster_index.clear();
	range_cluster(range_image,ringnum, cluster_index);

	
	//ROS_WARN("%f ms to process 1 frame pcl",(ros::Time::now() - time_begin).toSec() * 1000);
}

bool compare_cluster(pair<int, int> a, pair<int, int> b) {
	return a.second < b.second;
} //升序

bool most_frequent_value(vector<int> values, vector<int> &cluster_idx) {
	unordered_map<int, int> histcounts;
	for (int i = 0; i < values.size(); i++) {
		if (histcounts.find(values[i]) == histcounts.end()) {
			histcounts[values[i]] = 1;
		} else {
			histcounts[values[i]] += 1;
		}
	}

	int max = 0, maxi;
	vector<pair<int, int>> tr(histcounts.begin(), histcounts.end());
	sort(tr.begin(), tr.end(), compare_cluster);
	for (int i = 0; i < tr.size(); ++i) {
		if (tr[i].second > 10) {
			cluster_idx.push_back(tr[i].first);
		}
	}

	return true;
}

void visCVClusters()
{
	ros::Time time2 =  ros::Time::now(); double duration = 0.0;

    cv::Mat bvimage = cv::Mat(16, total_frame, CV_8UC1, cv::Scalar::all(0));
	cv::Mat range_imagec = cv::Mat(16, total_frame, CV_8UC3, cv::Scalar::all(0));

	for (auto it = range_image.begin(); it != range_image.end(); ++it) {
		int index = it->second.count_num;
		bvimage.at<uchar>(it->second.ring_i, it->second.frame_j) =
				it->second.range_zxy / 30 * 256;
	}

	cv::resize(bvimage, bvimage, cv::Size(1000, 100), 0, 0);
    cv::imwrite("/home/sustech1411/depth_img.jpg", bvimage);

    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		//	new pcl::visualization::PCLVisualizer("pcd")); //PCLVisualizer 可视化类
	//viewer->setBackgroundColor(0.8, 0.8, 0.8);
	//viewer->addCoordinateSystem(1);

	vector<int> cluster_id;
	most_frequent_value(cluster_index, cluster_id);
    cv::RNG rng(12345);

cout<<"pcl cluster num="<<cluster_id.size()<<endl;
viewer->removeAllPointClouds();
	for (int k = 0; k < cluster_id.size(); ++k) {
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr Colorcloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
		vector<double> hsv(3);
		hsv[0] = double(k) / double(cluster_id.size());
		hsv[1] = 1;
		hsv[2] = 1;

    int r = rng.uniform(0, 255);
    int g = rng.uniform(0, 255);
    int b = rng.uniform(0, 255);
		vector<double> rgb = hsv2rgb(hsv);
		for (int i = 0; i < total_frame; ++i) {
			for (int j = 0; j < 16; ++j) {
				if (cluster_index[i * 16 + j] == cluster_id[k]
						&& cluster_id[k] != -1) {
					unordered_map<int, Range>::iterator it_find;
					it_find = range_image.find(i * 16 + j);
					if (it_find != range_image.end()) {
						pcl::PointXYZRGB p;
						p.x = cloud_ngc.points[it_find->second.count_num].x;
						p.y = cloud_ngc.points[it_find->second.count_num].y;
						p.z = cloud_ngc.points[it_find->second.count_num].z;
						p.r = rgb[0];
						p.g = rgb[1];
						p.b = rgb[2];
						Colorcloud2->points.push_back(p);
            			range_imagec.at<cv::Vec3b>(it_find->second.ring_i, it_find->second.frame_j) = cv::Vec3b(r,g,b);
					}
				}
			}
		}

		if (Colorcloud2->points.size() > 5) {
			Colorcloud2->height = 1;
			Colorcloud2->width = Colorcloud2->points.size();

			/*if (zmin1 < 0)
				zmin1 = 0; //make sure object is up the ground*/

			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color2(Colorcloud2, (b), (g), (r));
			//viewer->removePointCloud("cloud2" + toString(k)); 
			viewer->addPointCloud(Colorcloud2, color2, "cloud2" + toString(k));
			
		}
	}

	//cout << range_image.size() << endl;

	int count_num = 0;
	for (auto it = range_image.begin(); it != range_image.end(); ++it) {
		count_num++;
	}

  //cv::imshow("bv", range_imagec);
	//cv::waitKey(0);
  
	cv::resize(range_imagec, range_imagec, cv::Size(1000, 100), 0, 0);
    cv::imwrite("/home/sustech1411/color_img.png", range_imagec);

	
	/*while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
	}*/
}

void depthOdomCallback(const sensor_msgs::ImageConstPtr& img,
                               const geometry_msgs::PoseStampedConstPtr& odom) {
  camera_pos_(0) = odom->pose.position.x;
  camera_pos_(1) = odom->pose.position.y;
  camera_pos_(2) = odom->pose.position.z;
  camera_q_ = Eigen::Quaterniond(odom->pose.orientation.w, odom->pose.orientation.x,
                                     odom->pose.orientation.y, odom->pose.orientation.z);
  /* get pose */
  /*camera_pos_(0) = odom->pose.pose.position.x;
  camera_pos_(1) = odom->pose.pose.position.y;
  camera_pos_(2) = odom->pose.pose.position.z;
  camera_q_ = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                                     odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);*/

  /* get depth image */
  /*cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(depth_image_);*/
}

void visCallback(const ros::TimerEvent& /*event*/) {
  if (!have_pcl)
	return;

  sensor_msgs::PointCloud2 pub_cloud;
  //publish ground
  pcl::toROSMsg(cloud_g, pub_cloud);
  g_pub.publish(pub_cloud);
  //publish ceilling
  pcl::toROSMsg(cloud_c, pub_cloud);
  c_pub.publish(pub_cloud);
  //publish no_ground
  pcl::toROSMsg(cloud_ngc, pub_cloud);
  ngc_pub.publish(pub_cloud);

  visCVClusters();
}

