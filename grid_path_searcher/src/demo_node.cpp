#include <iostream>
#include <fstream>
#include <math.h>
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
string frame_id_;
double height_thresh, range_min, range_max;
ros::Publisher original_map_pub_, no_floor_pub_, floor_pub_;
ros::Subscriber odom_sub_, depth_sub_, cloud_sub_;
ros::Timer vis_timer_;
pcl::PointCloud<pcl::PointXYZITR> cloud_msg;
//pcl::PointCloud<pcl::PointXYZI> cloud_msg;

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
    nh.param("layout/range_max", range_max, 20.0);

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

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& point_msg)
{
  //首先将接收到的ROS格式的点云消息sensor_msgs::PointCloud2转化为PCL格式的消息    PointCloud<pcl::PointXYZITR>
    pcl::fromROSMsg(*point_msg, cloud_msg);
    cloud_msg.header.frame_id = "map";
 /*
    //根据雷达型号，创建Mat矩阵，由于在此使用的雷达为128线，每条线上有1281个点，所以创建了一个大小为128*1281尺寸的矩阵，并用0元素初始化。
    int horizon_num = int(cloud_msg.points.size() / 16);
    cv::Mat range_mat = cv::Mat(16, horizon_num, CV_8UC3, cv::Scalar::all(0));
    double range; //range为该点的深度值
    int row_i, col_i; 
    int scan_num = 16;
    const double PI = 3.1415926;
    const double ANG = 57.2957795;

    int min_idx = 1000000; int max_idx = -1000000;
    //遍历每个点，计算该点对应的Mat矩阵中的索引，并为该索引对应的Mat矩阵元素赋值
    for(const auto pt : cloud_msg.points)
    {
      //cout<<"pointxyz: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<" angle="<<h_angle<<" ring="<<pt.ring<<endl;
      if (pcl_isfinite(pt.x) || pcl_isfinite(pt.y) || pcl_isfinite(pt.z)) {
        continue;
      }
      
      range = (double)sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);  
      if (range < range_min || range > range_max) {
        continue;
      }

      row_i = pt.ring;
      double h_angle = 180 / PI * std::atan2(pt.y, pt.x) + 180;
      int col = ceil(h_angle / 360 * horizon_num);
      if (col < min_idx) {min_idx = col;}
      if (col > max_idx) {max_idx = col;}
      cout<<"row="<<pt.ring<<" col="<<col<<" max="<<max_idx<<" min="<<min_idx<<endl;


      col_i = h_angle / 360 * horizon_num;   
      //忽略索引不合法的点
      if(row_i < 0 || row_i >= scan_num)
          continue;
      if(col_i < 0 || col_i >= horizon_num)
          continue;
      range_mat.at<double>(row_i, col_i) = range;
      //如果想转化为彩色的深度图，可以注释上面这一句，改用下面这一句；
      //range_mat.at<cv::Vec3b>(row_i, col_i) = cv::Vec3b(254-int(pt.x *2), 254- int(fabs(pt.y) / 0.5), 254-int(fabs((pt.z + 1.0f) /0.05)));
    }*/
    
    //cv::namedWindow("map",CV_WINDOW_NORMAL);//AUTOSIZE //创建一个窗口，用于显示深度图
    //cv::imshow("map",range_mat); //在这个窗口输出图片。
    //cv::waitKey(10); //设置显示时间
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
