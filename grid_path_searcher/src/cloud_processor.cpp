#include "cloud_processor.h"



void CloudProcessor::setCloudInput(pcl::PointCloud<pcl::PointR> cloud_input_)
{
    ROS_WARN("new_image");
	cloud_input = cloud_input_;
	input_cloud_size = cloud_input.points.size();
    //cout<<cloud_input.points.size()<<endl;
	row_num = 16;
	col_num = input_cloud_size / 16;
	vertical_deg = 2.0 * 180.0 / Pi;
	horizontal_deg = 360.0 / double(col_num) * 180.0 / Pi;
	range_mat = cv::Mat::zeros(row_num, col_num, CV_32F);
}

void CloudProcessor::processCloud()
{
	toRangeImage();
	toAngleImage();
}

void CloudProcessor::getSinCosVec()
{
    sines_vec.resize(row_num);
    cosines_vec.resize(row_num);
	float theta0 = 75.0;
	float dtheta = 2.0;
	for (int i = 0; i < row_num; i++) {
		float cur_theta = theta0 + i * dtheta;
		if (cur_theta < 90.0) {
			sines_vec[i] = sin(cur_theta * Pi / 180.0);
			cosines_vec[i] = cos(cur_theta * Pi / 180.0);
		} else {
			sines_vec[i] = sin((cur_theta - 90.0) * Pi / 180.0);
			cosines_vec[i] = cos((cur_theta - 90.0) * Pi / 180.0);
		}	
	}
}

void CloudProcessor::toAngleImage()
{
	angle_mat = cv::Mat::zeros(row_num, col_num, CV_32F);
	cv::Mat x_mat = cv::Mat::zeros(row_num, col_num, CV_32F);
	cv::Mat y_mat = cv::Mat::zeros(row_num, col_num, CV_32F);

	getSinCosVec();
	float dx, dy;
	x_mat.row(0) = range_mat.row(0) * cosines_vec[0];
	y_mat.row(0) = range_mat.row(0) * sines_vec[0];
	for (int r = 1; r < range_mat.rows; ++r) {
		x_mat.row(r) = range_mat.row(r) * cosines_vec[r];
		y_mat.row(r) = range_mat.row(r) * sines_vec[r];
		for (int c = 0; c < range_mat.cols; ++c) {
			dx = fabs(x_mat.at<float>(r, c) - x_mat.at<float>(r - 1, c));
			dy = fabs(y_mat.at<float>(r, c) - y_mat.at<float>(r - 1, c));
			angle_mat.at<float>(r, c) = atan2(dy, dx);
            /*if (r == 1)
                cout<<"angle mat: row="<<r<<" col="<<c<<" val_deg="<<angle_mat.at<float>(r, c) * 180.0 / Pi<<endl;*/
		}
	}

    //angle smoothing
    auto smoothed_image = ApplySavitskyGolaySmoothing(angle_mat, 5);
}

Mat CloudProcessor::ApplySavitskyGolaySmoothing(const Mat& image, int window_size) {
  Mat kernel = GetSavitskyGolayKernel(window_size);

  Mat smoothed_image;  // init an empty smoothed image
  cv::filter2D(image, smoothed_image, SAME_OUTPUT_TYPE, kernel, ANCHOR_CENTER,
               0, cv::BORDER_REFLECT101);
  return smoothed_image;
}

void CloudProcessor::toRangeImage()
{
    for(const auto pt : cloud_input.points)
    {
      double range = (double)sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);  
      if (range < range_min || range > range_max) 
        continue;

      int row_i = pt.ring;
      double h_angle = 180.0 / Pi * std::atan2(pt.y, pt.x) + 180;
      int col_i = ceil(h_angle / 360 * col_num) - 1;  
      //cout<<"pointxyz: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<pt.intensity<<" row="<<row_i<<" col="<<col_i<<" range="<<range<<endl; 
      if(row_i < 0 || row_i >= row_num || col_i < 0 || col_i >= col_num)
          ROS_ERROR("wrong idx num");

      range_mat.at<float>(row_i, col_i) = range;
    }
}

void CloudProcessor::reset()
{

}