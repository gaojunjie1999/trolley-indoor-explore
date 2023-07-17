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
#include <unordered_set>
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
enum ImageType {NGCW, GC, CONTOUR};

class CloudProcessor
{
private:
	//range image related
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
	ImageType image_type{NGCW};

	//proj image related
	double voxel_dim{0.1}, sensor_ramge{30.0};
	double VOXEL_DIM_INV = 1.0 / voxel_dim;
	int MAT_SIZE{std::ceil(sensor_ramge / voxel_dim)}, CMAT{MAT_SIZE / 2};
	cv::Mat img_mat;
	double resize_ratio{3.0};
	int blur_size{3};
	double cos_thresh = cos(2.0 * Pi / 180.0);
	double dist_thresh2{4.5}, dist_thresh1{4.5};
	std::vector<cv::Vec4i> refined_hierarchy;



public:
	cv::Mat range_mat, angle_mat, smoothed_mat, no_gcw_image, label_mat, dilated, gc_image, contour_image;
	pcl::PointCloud<pcl::PointR> cloud_ngcw, cloud_contour,cloud_gc;

public:
	//range image
	void setCloudInput(pcl::PointCloud<pcl::PointR> cloud_input_);
	void reset();
	void toRangeImage();
	void processCloud();
	void toAngleImage();
	void getSinCosVec();
	void ApplySavitskyGolaySmoothing() ;
	cv::Mat GetSavitskyGolayKernel() const;
	void LabelPixel(uint16_t label, int row_id, int col_id);
	void GetNeighbors(queue<pair<int, int>>& labeling_queue, const int& cur_row, const int& cur_col, const uint16_t label);
	cv::Mat GetUniformKernel(int window_size, int type) const;
	void ZeroOutGroundCeillingBFS();
	void toCloud(const cv::Mat& image_mat);
	void UnprojectPoint(const cv::Mat image, int row, int col, pcl::PointR& pt);

	//proj image
	void AdjecentDistanceFilter(std::vector<std::vector<cv::Point2f>>& contoursInOut);
	void TopoFilterContours(std::vector<vector<cv::Point2f>>& contoursInOut);
	void ExtractContour(const pcl::PointCloud<pcl::PointR>& cur_cloud, vector<vector<Vector2d>>realworld_contour);

	template <typename Point>
	inline void PointToImgSub(const Point& posIn, int& row_idx, int& col_idx) 
	{
		const int c_idx = CMAT;
		row_idx = c_idx + (int)std::round(posIn.x  * VOXEL_DIM_INV);
		col_idx = c_idx + (int)std::round(posIn.y * VOXEL_DIM_INV);
	}

	inline Vector2d ConvertCVPointToPoint2D(const cv::Point2f& cv_p) {
        Vector2d p;
        const int c_idx = CMAT;
        const float ratio = 1.0f;
        p(0) = (cv_p.y - c_idx) * voxel_dim / ratio;
        p(1) = (cv_p.x - c_idx) * voxel_dim / ratio;
        return p;
    }

	inline bool IsIdxesInImg(int& row_idx, int& col_idx) 
	{
        if (row_idx < 0 || row_idx > MAT_SIZE - 1 || col_idx < 0 || col_idx > MAT_SIZE - 1) {
            return false;
        }
        return true;
    }

	inline void InternalContoursIdxs(const std::vector<cv::Vec4i>& hierarchy,
                                     const std::size_t& high_idx,
                                     std::unordered_set<int>& internal_idxs)
    {
        if (hierarchy[high_idx][2] == -1) return;
        SameAndLowLevelIdxs(hierarchy, hierarchy[high_idx][2], internal_idxs);
    }

    inline void SameAndLowLevelIdxs(const std::vector<cv::Vec4i>& hierarchy,
                                    const std::size_t& cur_idx,
                                    std::unordered_set<int>& remove_idxs)
    {
        if (cur_idx == -1) return;
        int next_idx = cur_idx;
        while (next_idx != -1) {
            remove_idxs.insert(next_idx);
            SameAndLowLevelIdxs(hierarchy, hierarchy[next_idx][2], remove_idxs);
            next_idx = hierarchy[next_idx][0];
        }
    }

	inline float PixelDistance(const cv::Point2f& pre_p, const cv::Point2f& cur_p)
	{
		return std::hypotf(pre_p.x - cur_p.x, pre_p.y - cur_p.y);
	}
	
    inline void RemoveWallConnection(const std::vector<cv::Point2f>& contour,
                                     const cv::Point2f& add_p,
                                     std::size_t& refined_idx)
    {
        if (refined_idx < 2) return;
        if (!IsPrevWallVertex(contour[refined_idx-2], contour[refined_idx-1], add_p)) {
            return;
        } else {
            -- refined_idx;
            RemoveWallConnection(contour, add_p, refined_idx);
        }
    }
	    
	inline bool IsPrevWallVertex(const cv::Point2f& first_p,
                                 const cv::Point2f& mid_p,
                                 const cv::Point2f& add_p)
    {
        cv::Point2f diff_p1 = first_p - mid_p;
        cv::Point2f diff_p2 = add_p - mid_p;
        diff_p1 /= std::hypotf(diff_p1.x, diff_p1.y);
        diff_p2 /= std::hypotf(diff_p2.x, diff_p2.y);
        if (abs(diff_p1.dot(diff_p2)) > cos_thresh) 
			return true;
        return false;
    }

	void VisContours(std::vector<std::vector<cv::Point2i>> contours, const cv::Mat& origin_img, const string& pic_name)
	{
		cv::Mat imageContours = cv::Mat::zeros(origin_img.size(), CV_8UC1);  
		//cv::Mat Contours= cv::Mat::zeros(origin_img.size(), CV_8UC1);  
		for(int i = 0;i < contours.size(); i++)  
		{  
			/*for(int j = 0; j < contours[i].size(); j++)   
			{   
				cv::Point P = cv::Point(contours[i][j].x,contours[i][j].y);  
				Contours.at<uchar>(P)=255;  
			}  */
	
			//输出hierarchy向量内容  
			/*char ch[256];  
			sprintf(ch,"%d",i);  
			string str=ch;  
			cout<<"向量hierarchy的第" <<str<<" 个元素内容为："<<endl<<hierarchy[i]<<endl<<endl;  */
	
			//绘制轮廓  
			//cv::drawContours(imageContours,contours,i, cv::Scalar(255), 1, 8, refined_hierarchy);  
		}  
		cv::drawContours(imageContours,contours, -1, cv::Scalar(255), 1, 8, refined_hierarchy);
		string file_name = "/home/sustech1411/";
		const string pic_file = file_name + pic_name;
		cv::imwrite(pic_file, imageContours);
	}

	// extract contours
	int max_iter = 10;



	void EndPointExtraction(const cv::Mat& src, vector<Vector2i>& endpt_vec, vector<Vector2i>& midpt_vec);
	void cvThin(const cv::Mat& src, cv::Mat& dst, int intera);

};




#endif
