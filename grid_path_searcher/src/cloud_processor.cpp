#include "cloud_processor.h"



void CloudProcessor::setCloudInput(pcl::PointCloud<pcl::PointR> cloud_input_)
{
    //ROS_WARN("new_image");
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

    //angle mat smoothing
    switch (filter_type)
    {
    case None:
        smoothed_mat = angle_mat;
        break;
    case SG:
        ApplySavitskyGolaySmoothing();
    default:
        break;
    }

    //angle-based ground removal
    ZeroOutGroundBFS();
}

void CloudProcessor::ZeroOutGroundBFS() const {

    label_mat = cv::Mat::zeros(range_mat.size(), range_mat.type());

    for (int col_id = 0; col_id < range_mat.cols; col_id++) {
        int r = 0;
        while (r < range_mat.rows && range_mat.at<float>(r, c) < 0.001f) {
            ++r;
        }
        
        if (label_mat.at<float>(r, c) > 0) 
            continue;
        if (angle_mat.at<float>(r, c) > start_angle_thresh) 
            continue;
        LabelPixel(1, r, c);
    }

    int new_window_size = std::max(window_size - 2, 3);
    cv::Mat kernel = GetUniformKernel(new_window_size, CV_8U);
    cv::Mat dilated = cv::Mat::zeros(label_mat.size(), label_mat.type());
    cv::dilate(label_mat, dilated, kernel);
    for (int r = 0; r < dilated.rows; ++r) {
        for (int c = 0; c < dilated.cols; ++c) {
            if (dilated.at<uint16_t>(r, c) == 0) {
                // all unlabeled points are non-ground
                no_ground_image.at<float>(r, c) = range_mat.at<float>(r, c);
            }
        }
    }
    return res;
}

cv::Mat CloudProcessor::GetUniformKernel(int new_window_size, int type) const {
  if (new_window_size % 2 == 0) {
    throw std::logic_error("only odd window size allowed");
  }
  cv::Mat kernel = cv::Mat::zeros(new_window_size, 1, type);
  kernel.at<float>(0, 0) = 1;
  kernel.at<float>(new_window_size - 1, 0) = 1;
  kernel /= 2;
  r
}

void CloudProcessor::LabelPixel(int label, const int& row_id, const int& col_id)
{
    queue<pair<int, int>> labeling_queue;
    labeling_queue.push(make_pair(row_id, col_id));
    while (!labeling_queue.empty()) {
        int cur_row = labeling_queue.front().first;
        int cur_col = labeling_queue.front().second;
        labeling_queue.pop();

        int cur_label = label_mat.at<cur_row, cur_col>;
        if (cur_label > 0)
            continue;
        abel_mat.at<cur_row, cur_col> = label;

        auto current_depth = range_mat.at<cur_row, cur_col>;
        if (current_depth < 0.001f) {
            continue;
        
        GetNeighbors(labeling_queue, cur_row, cur_col);
      }
    }
}

void CloudProcessor::GetNeighbors(queue<pair<int, int>>& labeling_queue, const int& cur_row, const int& cur_col)
{
    for (int i = cur_row - step_row, i <= cur_row + step_row; i++) {
        if (i < 0 || i >= row_num)
            continue;

        for (int j = cur_col - step_col, i <= cur_col + step_col; j++) {
            if (i == cur_row && j == cur_col)
                continue;
            if (j < 0 || j >= col_num)
                continue;
            if (label_mat.at<i, j> > 0)
                continue;
            if (i == 0 || smoothed_mat.at<i, j> < ground_angle_thresh) 
                labeling_queue.push(make_pair(i, j));    
        }
    }
}

void CloudProcessor::ApplySavitskyGolaySmoothing() 
{
  cv::Mat kernel = GetSavitskyGolayKernel();
  cv::filter2D(angle_mat, smoothed_mat, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
}

cv::Mat CloudProcessor::GetSavitskyGolayKernel() const 
{
  if (window_size % 2 == 0) {
    throw std::logic_error("only odd window size allowed");
  }
  bool window_size_ok = window_size == 5 || window_size == 7 ||
                        window_size == 9 || window_size == 11;
  if (!window_size_ok) {
    throw std::logic_error("bad window size");
  }
  // below are no magic constants. See Savitsky-golay filter.
  cv::Mat kernel;
  switch (window_size) {
    case 5:
      kernel = cv::Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -3.0f;
      kernel.at<float>(0, 1) = 12.0f;
      kernel.at<float>(0, 2) = 17.0f;
      kernel.at<float>(0, 3) = 12.0f;
      kernel.at<float>(0, 4) = -3.0f;
      kernel /= 35.0f;
      return kernel;
    case 7:
      kernel = cv::Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -2.0f;
      kernel.at<float>(0, 1) = 3.0f;
      kernel.at<float>(0, 2) = 6.0f;
      kernel.at<float>(0, 3) = 7.0f;
      kernel.at<float>(0, 4) = 6.0f;
      kernel.at<float>(0, 5) = 3.0f;
      kernel.at<float>(0, 6) = -2.0f;
      kernel /= 21.0f;
      return kernel;
    case 9:
      kernel = cv::Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -21.0f;
      kernel.at<float>(0, 1) = 14.0f;
      kernel.at<float>(0, 2) = 39.0f;
      kernel.at<float>(0, 3) = 54.0f;
      kernel.at<float>(0, 4) = 59.0f;
      kernel.at<float>(0, 5) = 54.0f;
      kernel.at<float>(0, 6) = 39.0f;
      kernel.at<float>(0, 7) = 14.0f;
      kernel.at<float>(0, 8) = -21.0f;
      kernel /= 231.0f;
      return kernel;
    case 11:
      kernel = cv::Mat::zeros(window_size, 1, CV_32F);
      kernel.at<float>(0, 0) = -36.0f;
      kernel.at<float>(0, 1) = 9.0f;
      kernel.at<float>(0, 2) = 44.0f;
      kernel.at<float>(0, 3) = 69.0f;
      kernel.at<float>(0, 4) = 84.0f;
      kernel.at<float>(0, 5) = 89.0f;
      kernel.at<float>(0, 6) = 84.0f;
      kernel.at<float>(0, 7) = 69.0f;
      kernel.at<float>(0, 8) = 44.0f;
      kernel.at<float>(0, 9) = 9.0f;
      kernel.at<float>(0, 10) = -36.0f;
      kernel /= 429.0f;
      return kernel;
  }
  return kernel;
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

