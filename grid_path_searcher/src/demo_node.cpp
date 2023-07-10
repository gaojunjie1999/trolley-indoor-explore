#include <iostream>
#include <fstream>
#include <math.h>
#include <unordered_map>
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
#include <lidar_point.h>


#define logit(x) (log((x) / (1 - (x))))

/*typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImageOdom;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImageOdom;
typedef message_filters::Synchronizer<SyncPolicyImageOdom>* SynchronizerImageOdom;*/

using namespace std;
using namespace Eigen;

// pointcloud segmentation: wall and ground detection
struct Range {
	float range_xy;
	float range_zxy;
	int ring_i;
	int frame_j;
	int count_num;
};

string frame_id_;
double height_thresh, range_min, range_max;
ros::Publisher original_map_pub_, no_floor_pub_, floor_pub_;
ros::Subscriber odom_sub_, depth_sub_, cloud_sub_;
ros::Timer vis_timer_;
pcl::PointCloud<pcl::PointXYZI> cloud_msg;
pcl::PointCloud<pcl::PointXYZITR> cloud_ng, cloud_g;
unordered_map<int, Range> range_image;
ros::Time time_begin; int idx_begin = 0;


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

    //vis_timer_ = nh.createTimer(ros::Duration(0.05), visCallback); 

    original_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/layout/original_map", 10);
    no_floor_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/layout/no_floor", 10);
    floor_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/layout/floor", 10);

    nh.param("layout/height_thresh", height_thresh, 0.2);
    nh.param("layout/frame_id", frame_id_, string("world"));
    nh.param("layout/range_min", range_min, 0.1);
    nh.param("layout/range_max", range_max, 200.0);

    ros::Rate rate(100);
    bool status = ros::ok();
    while (status)
    {    
        ros::spinOnce();
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}

void CloudFilter(const pcl::PointCloud<pcl::PointXYZITR>& cloudIn,
		pcl::PointCloud<pcl::PointXYZITR>& cloudOut, float x_min, float x_max,
		float y_min, float y_max, float z_min, float z_max) {
  //TODO
  return;

	cloudOut.header = cloudIn.header;
	//cloudOut.sensor_orientation_ = cloudIn.sensor_orientation_;
	//cloudOut.sensor_origin_ = cloudIn.sensor_origin_;
	cloudOut.points.clear();
	//1) set parameters for removing cloud reflect on ego vehicle
	float x_limit_min = -1.8, x_limit_max = 1.8, y_limit_forward = 5.0,
			y_limit_backward = -4.5;
	//2 apply the filter
	for (int i = 0; i < cloudIn.size(); ++i) {
		float x = cloudIn.points[i].x;
		float y = cloudIn.points[i].y;
		float z = cloudIn.points[i].z;
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

void transform2RangeImage(const pcl::PointCloud<pcl::PointXYZI>& cloudIn,
		pcl::PointCloud<pcl::PointXYZITR>& ng_cloudOut,
		  pcl::PointCloud<pcl::PointXYZITR>& g_cloudOut, 
        unordered_map<int, Range> &unordered_map_out) 
{
  int total_frame = (int)(cloudIn.points.size() / 16);
	pcl::PointCloud<pcl::PointXYZITR>::Ptr cloud2range(new pcl::PointCloud<pcl::PointXYZITR>);
	pcl::PointCloud<pcl::PointXYZITR>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZITR>);
  cloud2range->points.resize(cloudIn.points.size());

	for (int j = 0; j < total_frame; ++j) {
		int num_odd = 8;                //基数从8位开始排
		int num_even = 0;                //偶数从头

		for (int i = 0; i < 16; ++i) {
			if (float(i % 2) == 0.0) {
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
  }

	cloud2range->height = 1;
	cloud2range->width = cloud2range->points.size();


	float xmin = -350, xmax = 350, ymin = -300, ymax = 300, zmin = -100, zmax = 300;
  CloudFilter(*cloud2range, *cloud_filtered, xmin, xmax, ymin, ymax, zmin, zmax);

  GroundRemove2 ground_remove(3, 20, 1.0, 0.15);
    
  ground_remove.RemoveGround2(*cloud_filtered, g_cloudOut, ng_cloudOut);


	float x_limit_min = -1, x_limit_max = 1, y_limit_forward = 4.0,
			y_limit_backward = -3;

        for(int i=0; i<ng_cloudOut.points.size(); ++i){

            float x = ng_cloudOut.points[i].x;
	    float y = ng_cloudOut.points[i].y;
	    float z = ng_cloudOut.points[i].z;
       	    float distance = CalculateRangeXY( ng_cloudOut.points[i]);
            /*if (distance > 30)
				continue;
			if ((x > x_limit_min && x < x_limit_max && y > y_limit_backward
					&& y < y_limit_forward))
				continue;

			if ((z < -1 || z > 3))
				continue;*/


                        int ringnum = ng_cloudOut.points[i].ring;

			int image_index = ng_cloudOut.points[i].pcaketnum * 16 + (31 - ringnum);
			Range r;
			r.ring_i = 31 -ringnum;
			r.frame_j = ng_cloudOut.points[i].pcaketnum;
			r.count_num = i;
			r.range_xy = ng_cloudOut.points[i].z;
			r.range_zxy = CalculateRangeZXY(ng_cloudOut.points[i]);
			unordered_map_out.insert(make_pair(image_index, r));

    }
}

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& point_msg)
{
    double duration = (ros::Time::now() - time_begin).toSec();
    if (duration < 1) {
      return;
    } else {
      time_begin = ros::Time::now();
    }
    pcl::fromROSMsg(*point_msg, cloud_msg);
    cloud_msg.header.frame_id = "map";

    transform2RangeImage(cloud_msg, cloud_ng, cloud_g, range_image);



/*







 
    //根据雷达型号，创建Mat矩阵，由于在此使用的雷达为128线，每条线上有1281个点，所以创建了一个大小为128*1281尺寸的矩阵，并用0元素初始化。
    int horizon_num = int(cloud_msg->points.size() / 16);
    cv::Mat range_mat = cv::Mat(16, horizon_num, CV_64FC3, cv::Scalar::all(0));
    //cv::Mat range_mat = cv::Mat(16, horizon_num, CV_8UC3, cv::Scalar::all(0));
    double range; //range为该点的深度值
    int row_i, col_i; 
    int scan_num = 16;
    const double PI = 3.1415926;
    int min_idx = 1000000; int max_idx = -1000000;
    //遍历每个点，计算该点对应的Mat矩阵中的索引，并为该索引对应的Mat矩阵元素赋值
    for(const auto pt : cloud_msg->points)
    {
      if (!pcl_isfinite(pt.x) || !pcl_isfinite(pt.y) || !pcl_isfinite(pt.z)) {
        continue;
      }
      
      range = (double)sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);  
      if (range < range_min || range > range_max) {
        continue;
      }

      row_i = pt.ring;
      double h_angle = 180 / PI * std::atan2(pt.y, pt.x) + 180;
      col_i = ceil(h_angle / 360 * horizon_num) - 1;  
      cout<<"pointxyz: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<" row="<<row_i<<" col="<<col_i<<" range="<<range<<endl; 
     
      //忽略索引不合法的点
      if(row_i < 0 || row_i >= scan_num)
          continue;
      if(col_i < 0 || col_i >= horizon_num)
          continue;
      range_mat.at<double>(row_i, col_i) = range;
    }
    
    /*std::string pic_file{"/home/sustech1411/"};
    std::string pic_name = pic_file + std::to_string(idx_begin) + ".png";
    idx_begin++; 
    bool result = cv::imwrite("/home/sustech1411/a.jpg", range_mat);

    cv::namedWindow("map",CV_WINDOW_NORMAL);//AUTOSIZE //创建一个窗口，用于显示深度图
    cv::imshow("map",range_mat); //在这个窗口输出图片。
    cv::waitKey(0); //设置显示时间 */
    

}

void depthOdomCallback(const sensor_msgs::ImageConstPtr& img,
                               const geometry_msgs::PoseStampedConstPtr& odom) {
//cout<<"odom img"<<endl;
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
  //publishMapInflate(true);
}


/*void publishMapInflate(bool all_info) {
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  grid_cloud = cloud;

  Eigen::Vector3i min_cut = local_bound_min_;
  Eigen::Vector3i max_cut = local_bound_max_;

  if (all_info) {
    int lmm = local_map_margin_;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);
  }

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z) {
        if (occupancy_buffer_inflate_[toAddress(x, y, z)] == 0) continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > visualization_truncate_height_) continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_.publish(cloud_msg);

  if (save_pcd && cloud.width > 10) {
    cout<<"saving pcd file"<<endl;
    pcl::io::savePCDFileASCII ("/home/hitcsc/map_ws/src/grid_path_searcher/src/test_pcd.pcd", cloud); //将点云保存到PCD文件中
  }
  // ROS_INFO("pub map");
}*/
