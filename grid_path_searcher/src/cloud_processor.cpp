#include "cloud_processor.h"



void CloudProcessor::setCloudInput(pcl::PointCloud<pcl::PointR> cloud_input_)
{
    //ROS_WARN("new_image");
    reset();
	cloud_input = cloud_input_;
	input_cloud_size = cloud_input.points.size();
    cout<<"input size="<<cloud_input.points.size()<<endl;
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
    ZeroOutGroundCeillingBFS();

    image_type = NGCW;
    toCloud(no_gcw_image);
    image_type = GC;
    toCloud(gc_image);
    image_type = CONTOUR;
    toCloud(contour_image);
}

void CloudProcessor::getSinCosVec()
{
    sines_vec.resize(row_num);
    cosines_vec.resize(row_num);
	float theta0 = -15.0;
	float dtheta = 2.0;
	for (int i = 0; i < row_num; i++) {
		float cur_theta = theta0 + i * dtheta;
        sines_vec[i] = sin(fabs(cur_theta) * Pi / 180.0);
        cosines_vec[i] = cos(fabs(cur_theta) * Pi / 180.0);
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
}

void CloudProcessor::ZeroOutGroundCeillingBFS()
{
    label_mat = cv::Mat::zeros(row_num, col_num, cv::DataType<uint16_t>::type);
    uint16_t label{1}, label_contour{2};
    /*for (int i = 0; i < col_num; i++) {
        label_mat.at<uint16_t>(0, i) = label;
    }*/
    // ground removal
    for (int c = 0; c < label_mat.cols; c++) {
        for (int r = 0; r <= 0; r++) {
            if (range_mat.at<float>(r, c) < 0.001f) 
                continue;     
            if (label_mat.at<uint16_t>(r, c) > 0) 
                continue;
            if (r == 0 && smoothed_mat.at<float>(1, c) > start_angle_thresh) 
                continue;       
            
            LabelPixel(label, r, c);

        }
    }
    //ceilling removal
    for (int c = 0; c < label_mat.cols; c++) {
        int r = row_num - 1;
        if (range_mat.at<float>(r, c) < 0.001f) 
            continue;     
        if (label_mat.at<uint16_t>(r, c) > 0) 
            continue;
        if (smoothed_mat.at<float>(r, c) > start_angle_thresh) 
            continue;       

        LabelPixel(label, r, c);

    }
    //wall removal
    for (int c = 0; c < label_mat.cols; c++) {
        for (int r = 0; r < label_mat.rows; r++) {
            if (range_mat.at<float>(r, c) < 0.001f) 
                continue;     
            if (label_mat.at<uint16_t>(r, c) > 0) 
                continue;
            //if (r == 0 && smoothed_mat.at<float>(1, c) < start_angle_thresh) 
               // continue;       
            
            //LabelPixel(label_contour, r, c);



            if (smoothed_mat.at<float>(r, c) < 85 * Pi / 180) 
                continue;
            label_mat.at<uint16_t>(r, c) = label_contour;

        }
    }

    /*int new_window_size = std::max(window_size - 2, 3);
    cv::Mat kernel = GetUniformKernel(new_window_size, CV_8U);
    dilated = cv::Mat::zeros(label_mat.size(), label_mat.type());
    cv::dilate(label_mat, dilated, kernel);*/
    dilated = label_mat;
    no_gcw_image = cv::Mat::zeros(range_mat.size(), range_mat.type());
    gc_image = cv::Mat::zeros(range_mat.size(), range_mat.type()); 
    contour_image = cv::Mat::zeros(range_mat.size(), range_mat.type()); 
    for (int r = 0; r < dilated.rows; ++r) {
        for (int c = 0; c < dilated.cols; ++c) { 
            if (dilated.at<uint16_t>(r, c) == 0) {
                // all unlabeled points are non-ground
                //cout<<"r,c="<<r<<" "<<c<<" range="<<range_mat.at<float>(r,c)<<endl;
                no_gcw_image.at<float>(r, c) = range_mat.at<float>(r, c);
            } else if (dilated.at<uint16_t>(r, c) == label){
                gc_image.at<float>(r, c) = range_mat.at<float>(r, c);
            } else if (dilated.at<uint16_t>(r, c) == label_contour){
                contour_image.at<float>(r, c) = range_mat.at<float>(r, c);
            } else {
                ROS_ERROR("wrong label");
            }
        }
    }
}

cv::Mat CloudProcessor::GetUniformKernel(int new_window_size, int type) const {
  if (new_window_size % 2 == 0) {
    throw std::logic_error("only odd window size allowed");
  }
  cv::Mat kernel = cv::Mat::zeros(new_window_size, 1, type);
  kernel.at<float>(0, 0) = 1;
  kernel.at<float>(new_window_size - 1, 0) = 1;
  kernel /= 2;
  return kernel;
}

void CloudProcessor::LabelPixel(uint16_t label, int row_id, int col_id)
{
    queue<pair<int, int>> labeling_queue;
    labeling_queue.push(make_pair(row_id, col_id));
    while (!labeling_queue.empty()) {
        int cur_row = labeling_queue.front().first;
        int cur_col = labeling_queue.front().second;
        labeling_queue.pop();

        if (label_mat.at<uint16_t>(cur_row, cur_col) > 0)
            continue;
        label_mat.at<uint16_t>(cur_row, cur_col) = label;

        auto current_depth = range_mat.at<float>(cur_row, cur_col);;
        if (current_depth < 0.001f) {
            continue;
        }
        GetNeighbors(labeling_queue, cur_row, cur_col, label);
      
    }
}

void CloudProcessor::GetNeighbors(queue<pair<int, int>>& labeling_queue, const int& cur_row, const int& cur_col, const uint16_t label)
{
    //cout<<"cur: r="<<cur_row<<" c="<<cur_col<<" range="<<range_mat.at<float>(cur_row, cur_col)<<" angle="<<180/Pi*smoothed_mat.at<float>(cur_row, cur_col)<<endl;
    for (int i = cur_row - step_row; i <= cur_row + step_row; i++) {
        if (i < 0 || i >= row_num || i == cur_row)
            continue;
        if (label_mat.at<uint16_t>(i, cur_col) > 0)
            continue;


        if (fabs(smoothed_mat.at<float>(i, cur_col) - smoothed_mat.at<float>(cur_row, cur_col)) < ground_angle_thresh) {
                //cout<<"r="<<i<<" c="<<cur_col<<" angle="<<180/Pi*smoothed_mat.at<float>(i, cur_col)<<endl;
            labeling_queue.push(make_pair(i, cur_col)); 
        } else {
            //cout<<"r="<<i<<" c="<<" range="<<range_mat.at<float>(i, cur_col)<<cur_col<<" angle="<<180/Pi*smoothed_mat.at<float>(i, cur_col)<<endl;
        }
    }
               
    for (int j = cur_col - step_col; j <= cur_col + step_col; j++) {
            if (j == cur_col || j < 0 || j >= col_num)
                continue;
            if (label_mat.at<uint16_t>(cur_row, j) > 0)
                continue;


            if (fabs(smoothed_mat.at<float>(cur_row, j) - smoothed_mat.at<float>(cur_row, cur_col)) < ground_angle_thresh) {
                labeling_queue.push(make_pair(cur_row, j)); 
            } else {
                //cout<<"rr="<<cur_row<<" c="<<j<<" range="<<range_mat.at<float>(cur_row, j)<<" angle="<<180/Pi*smoothed_mat.at<float>(cur_row, j)<<endl;
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

void CloudProcessor::ExtractContour(const pcl::PointCloud<pcl::PointR>& cur_cloud, vector<vector<Vector3d>>realworld_contour)
{
    img_mat = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
    //from 3d cloud to 2d image
    int row_idx, col_idx, inf_row, inf_col;
    const std::vector<int> inflate_vec{-1, 0, 1};
    for (const auto& pcl_p : cur_cloud.points) {
        PointToImgSub(pcl_p, row_idx, col_idx);
        if (!IsIdxesInImg(row_idx, col_idx)) 
            continue;
        for (const auto& dr : inflate_vec) {
            for (const auto& dc : inflate_vec) {
                inf_row = row_idx + dr;
                inf_col = col_idx + dc;
                if (IsIdxesInImg(inf_row, inf_col)) {
                    img_mat.at<float>(inf_row, inf_col) += 1.0;
                }
            }
        }
    }
    cv::imwrite("/home/sustech1411/img_before_resize.png", img_mat);

    cv::Mat Rimg;
    //resize & blur
    img_mat.convertTo(Rimg, CV_8UC1, 255);
    cv::resize(Rimg, Rimg, cv::Size(), resize_ratio, resize_ratio, cv::InterpolationFlags::INTER_LINEAR);
    cv::boxFilter(Rimg, Rimg, -1, cv::Size(blur_size, blur_size), cv::Point2i(-1, -1), false);
    cv::imwrite("/home/sustech1411/img_after_resize.png", Rimg);

    //ExtractRefinedContours
    std::vector<std::vector<cv::Point2i>> raw_contours;
    std::vector<vector<cv::Point2f>> refined_contours;
    std::vector<cv::Vec4i> refined_hierarchy;
    //refined_contours.clear(), refined_hierarchy.clear();
    cv::findContours(Rimg, raw_contours, refined_hierarchy, 
                     cv::RetrievalModes::RETR_TREE, 
                     cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1);
                     
    refined_contours.resize(raw_contours.size());
    for (std::size_t i=0; i<raw_contours.size(); i++) {
        // using Ramer–Douglas–Peucker algorithm url: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        cv::approxPolyDP(raw_contours[i], refined_contours[i], dist_thresh1, true);
    }
    TopoFilterContours(refined_contours); 
    AdjecentDistanceFilter(refined_contours);


}


void CloudProcessor::AdjecentDistanceFilter(std::vector<std::vector<cv::Point2f>>& contoursInOut) {
    /* filter out vertices that are overlapped with neighbor */
    std::unordered_set<int> remove_idxs;
    for (std::size_t i=0; i<contoursInOut.size(); i++) { 
        const auto c = contoursInOut[i];
        const std::size_t c_size = c.size();
        std::size_t refined_idx = 0;
        for (std::size_t j=0; j<c_size; j++) {
            cv::Point2f p = c[j]; 
            if (refined_idx < 1 || PixelDistance(contoursInOut[i][refined_idx-1], p) > dist_thresh2) {
                /** Reduce wall nodes */
                RemoveWallConnection(contoursInOut[i], p, refined_idx);
                contoursInOut[i][refined_idx] = p;
                refined_idx ++;
            }
        }
        /** Reduce wall nodes */
        RemoveWallConnection(contoursInOut[i], contoursInOut[i][0], refined_idx);
        contoursInOut[i].resize(refined_idx);
        if (refined_idx > 1 && PixelDistance(contoursInOut[i].front(), contoursInOut[i].back()) < dist_thresh2) {
            contoursInOut[i].pop_back();
        }
        if (contoursInOut[i].size() < 3) remove_idxs.insert(i);
    }
    if (!remove_idxs.empty()) { // clear contour with vertices size less that 3
        std::vector<CVPointStack> temp_contours = contoursInOut;
        contoursInOut.clear();
        for (int i=0; i<temp_contours.size(); i++) {
            if (remove_idxs.find(i) != remove_idxs.end()) continue;
            contoursInOut.push_back(temp_contours[i]);
        }
    }
}


void CloudProcessor::TopoFilterContours(std::vector<vector<cv::Point2f>>& contoursInOut) {
    std::unordered_set<int> remove_idxs;
    for (int i=0; i<contoursInOut.size(); i++) {
        if (remove_idxs.find(i) != remove_idxs.end()) 
            continue;
        const auto poly = contoursInOut[i];
        if (poly.size() < 3) {
            remove_idxs.insert(i);
        } else {
            InternalContoursIdxs(refined_hierarchy_, i, remove_idxs);
        }
    }
    if (!remove_idxs.empty()) {
        std::vector<vector<cv::Point2f>> temp_contours = contoursInOut;
        contoursInOut.clear();
        for (int i=0; i<temp_contours.size(); i++) {
            if (remove_idxs.find(i) != remove_idxs.end()) continue;
            contoursInOut.push_back(temp_contours[i]);
        }
    }
}

void CloudProcessor::toCloud(const cv::Mat& image_mat)
{
    pcl::PointCloud<pcl::PointR> pt_cloud; 
    for (int i = 0; i < row_num; i++) {
        for (int j = 0; j < col_num; j++) {
            if (image_mat.at<float>(i, j) < 0.0001f)
                continue;
            pcl::PointR pt;
            UnprojectPoint(image_mat, i, j, pt);
            pt_cloud.points.push_back(pt);
        }
    }
    switch (image_type)
    {
    case NGCW:
        cloud_ngcw = pt_cloud;
        cout<<"cloud ngcw";
        break;
    case GC:
        cloud_gc = pt_cloud;
        cout<<"cloud gc";
        break;
    case CONTOUR:
        cloud_contour = pt_cloud;
        cout<<"cloud contour";
        break;
    }
    cout<<" size="<<pt_cloud.points.size()<<endl;
    pt_cloud.points.clear();
}

void CloudProcessor::UnprojectPoint(const cv::Mat image, int row, int col, pcl::PointR& pt)
{
    float depth = image.at<float>(row, col);
    double angle_z = (-15.0 + 2.0 * row) / 180.0 * Pi;
    double angle_xy = ((col + 1) * 360.0 / col_num - 180.0) * Pi / 180.0;

    pt.x = depth * cos(angle_z) * cos(angle_xy);
    pt.y = depth * cos(angle_z) * sin(angle_xy);
    pt.z = depth * sin(angle_z);
    pt.ring = row;
    //pt.intensity = 0;
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
    vertical_angles.clear();
    horizontal_angles.clear();
	sines_vec.clear();
	cosines_vec.clear();
    cloud_ngcw.points.clear();
    cloud_contour.points.clear();
    cloud_gc.points.clear();
    
    /*no_gcw_image.release();
    angle_mat.release();
    range_mat.release();
    smoothed_mat.release();
    label_mat.release();*/
}