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

void CloudProcessor::ExtractContour(const pcl::PointCloud<pcl::PointR>& cur_cloud, vector<vector<Vector2d>>realworld_contour)
{
    img_mat = cv::Mat::zeros(MAT_SIZE, MAT_SIZE, CV_32FC1);
    //from 3d cloud to 2d image
    int row_idx, col_idx, inf_row, inf_col;
    //const std::vector<int> inflate_vec{-1, 0, 1};
    const std::vector<int> inflate_vec{0};
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
    //cv::imwrite("/home/sustech1411/img_before_resize.png", img_mat);
 
    cv::Mat Rimg;
    //resize & blur
    img_mat.convertTo(Rimg, CV_8UC1, 255);
    cv::resize(Rimg, Rimg, cv::Size(), resize_ratio, resize_ratio, cv::InterpolationFlags::INTER_LINEAR);
    //cv::imwrite("/home/sustech1411/img_after_resize.png", Rimg);
    
    cv::boxFilter(Rimg, Rimg, -1, cv::Size(blur_size, blur_size), cv::Point2i(-1, -1), false);
    //cv::imwrite("/home/sustech1411/img_after_box_filter.png", Rimg);

    //extract end points & refine
    vector<vector<cv::Point2i>> raw_points, refined_points;
    cv::Mat ske_img;
    cvThin(Rimg, ske_img, max_iter);
    //cv::imwrite("/home/sustech1411/skeleton.png", ske_img);
    EndPointExtraction(ske_img, raw_points);

    //wall seg fitting  
   /* cv::Mat contour_pts = cv::Mat::zeros(Rimg.size(), Rimg.type());
    for (const auto& vec : raw_points)
	{
        for (const auto& pt : vec) {
            contour_pts.at<uchar>(pt.y, pt.x) = 255;
        }
	}*/
    cv::imwrite("/home/sustech1411/skeleton.png", ske_img);

    vector<cv::Vec4i> lines;
    cv::Mat res;  
    cv::HoughLinesP(ske_img, lines, 1, Pi/180, 30, 5, 10);  
    cout<<"size="<<lines.size()<<endl;
    //MergeLines(lines);
    /*for( size_t i = 0; i < lines.size(); i++ )  
    {  
        cv::Vec4i l = lines[i];  
        cv::line(ske_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(155, 0, 0), 2);  
    }  
    cv::imwrite("/home/sustech1411/lines.png",ske_img);*/

    MergeLines(lines);
    cout<<"size="<<lines.size()<<endl;
    for( size_t i = 0; i < lines.size(); i++ )  
    {  
        cv::Vec4i l = lines[i];  
        cv::line(ske_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(155, 0, 0), 2);  
    }  
    cv::imwrite("/home/sustech1411/lines.png",ske_img);
    
    // dynamic space filter & interpolate (rmv obstacle pt cloud: ob pcl will be replaced if a farther surface in the same scan)











/*
    //ExtractRefinedContours
    std::vector<std::vector<cv::Point2i>> raw_contours;
    std::vector<vector<cv::Point2f>> refined_contours;
  
    refined_hierarchy.clear();
    cv::findContours(Rimg, raw_contours, refined_hierarchy, 
                     cv::RetrievalModes::RETR_TREE, 
                     cv::ContourApproximationModes::CHAIN_APPROX_TC89_L1);
    VisContours(raw_contours, Rimg, "find_contours.png");
                  
    refined_contours.resize(raw_contours.size());
    for (std::size_t i=0; i<raw_contours.size(); i++) {
        // using Ramer–Douglas–Peucker algorithm url: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        cv::approxPolyDP(raw_contours[i], refined_contours[i], dist_thresh1, true);
    }
    TopoFilterContours(refined_contours); 
    //VisContoursFloat(refined_contours, Rimg, "after_topo_filter.png");

    AdjecentDistanceFilter(refined_contours);
    //VisContours(refined_contours, Rimg, "after_dist_filter.png");

    //ConvertContoursToRealWorld
    const std::size_t C_N = refined_contours.size();
    realworld_contour.clear(), realworld_contour.resize(C_N);
    for (std::size_t i = 0; i < C_N; i++) {
        const auto cv_contour = refined_contours[i];
        const std::size_t vec_size = cv_contour.size();
        realworld_contour[i].resize(vec_size);
        for (std::size_t j = 0; j < vec_size; j++) {
            cv::Point2f cv_p = cv_contour[j];
            Vector2d p = ConvertCVPointToPoint2D(cv_p);
            realworld_contour[i][j] = p;
        }
    }*/
}

void CloudProcessor::MergeLines(vector<cv::Vec4i>& lines_inout)
{
    //cv::Point center(420, 520);
    vector<cv::Vec4i> lines_out;
    std::map<int, int> idx_map;

    for (int i = 0; i < lines_inout.size(); i++) {
        if (idx_map.find(i) != idx_map.end()) continue;
        //cout<<"i="<<i<<endl;
        auto line = lines_inout[i];
        idx_map[i] = 1e3;

        cv::Point2i start_pt(line[0], line[1]);
        cv::Point2i end_pt(line[2], line[3]);
        double length = PixelDistance(start_pt, end_pt);
        int max_idx = i;
        for (int j = i + 1; j < lines_inout.size(); j++) {
            auto next_line = lines_inout[j];
            if (idx_map.find(j) != idx_map.end()) continue;

            if (CanMergeLine(line, next_line)) {
                //cout<<"   j="<<j<<endl;
                idx_map[j] = 1e3;
                cv::Point2i next_start_pt(next_line[0], next_line[1]);
                cv::Point2i next_end_pt(next_line[2], next_line[3]);
                
                if (PixelDistance(next_start_pt, next_end_pt) > length) {
                    start_pt = next_start_pt;
                    end_pt = next_end_pt;
                    length = PixelDistance(next_start_pt, next_end_pt);
                    max_idx = j;
                }
            }
        }
        lines_out.emplace_back(lines_inout[max_idx]);
    }
    lines_inout = lines_out;
}   

inline bool CloudProcessor::CanMergeLine(const cv::Vec4i& line1, const cv::Vec4i& line2)
{
    cv::Point2i p11(line1[0], line1[1]);
    cv::Point2i p12(line1[2], line1[3]);
    //cout<<"lin1: "<<"("<<p11.x<<","<<p11.y<<")"<<"("<<p12.x<<","<<p12.y<<")"<<endl;
    double A1 = p12.y - p11.y;
    double B1 = p11.x - p12.x;
    double C1 = p12.x * p11.y - p11.x * p12.y;

    cv::Point2i p21(line2[0], line2[1]);
    cv::Point2i p22(line2[2], line2[3]);
    //cout<<"lin2: "<<"("<<p21.x<<","<<p21.y<<")"<<"("<<p22.x<<","<<p22.y<<")"<<endl;
    double d1 = fabs(A1 * p21.x + B1 * p21.y + C1) / sqrt(A1 * A1 + B1 * B1);
    double d2 = fabs(A1 * p22.x + B1 * p22.y + C1) / sqrt(A1 * A1 + B1 * B1);
    //cout<<"d="<<d1<<" "<<d2<<endl;

    if (d1 < merge_thresh && d2 < merge_thresh) {
        return true;
    }
    return false;
}

void CloudProcessor::cvThin(const cv::Mat& src, cv::Mat& dst, int intera)
{
if(src.type()!=CV_8UC1)
    {
    printf("只能处理二值或灰度图像\n");
    return;
    }
//非原地操作时候，copy src到dst
if(dst.data!=src.data)
    {
    src.copyTo(dst);
    }

int i, j, n;
int width, height;
width = src.cols -1;
//之所以减1，是方便处理8邻域，防止越界
height = src.rows -1;
int step = src.step;
int  p2,p3,p4,p5,p6,p7,p8,p9;
uchar* img;
bool ifEnd;
int A1;
cv::Mat tmpimg;
//n表示迭代次数
for(n = 0; n<intera; n++)
    {
    dst.copyTo(tmpimg);
    ifEnd = false;
    img = tmpimg.data;
    for(i = 1; i < height; i++)
        {
        img += step;
        for(j =1; j<width; j++)
            {
            uchar* p = img + j;
            A1 = 0;
            if( p[0] > 0)
                {
                if(p[-step]==0&&p[-step+1]>0) //p2,p3 01模式
                    {
                    A1++;
                    }
                if(p[-step+1]==0&&p[1]>0) //p3,p4 01模式
                    {
                    A1++;
                    }
                if(p[1]==0&&p[step+1]>0) //p4,p5 01模式
                    {
                    A1++;
                    }
                if(p[step+1]==0&&p[step]>0) //p5,p6 01模式
                    {
                    A1++;
                    }
                if(p[step]==0&&p[step-1]>0) //p6,p7 01模式
                    {
                    A1++;
                    }
                if(p[step-1]==0&&p[-1]>0) //p7,p8 01模式
                    {
                    A1++;
                    }
                if(p[-1]==0&&p[-step-1]>0) //p8,p9 01模式
                    {
                    A1++;
                    }
                if(p[-step-1]==0&&p[-step]>0) //p9,p2 01模式
                    {
                    A1++;
                    }
                p2 = p[-step]>0?1:0;
                p3 = p[-step+1]>0?1:0;
                p4 = p[1]>0?1:0;
                p5 = p[step+1]>0?1:0;
                p6 = p[step]>0?1:0;
                p7 = p[step-1]>0?1:0;
                p8 = p[-1]>0?1:0;
                p9 = p[-step-1]>0?1:0;
                if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7  &&  A1==1)
                    {
                    if((p2==0||p4==0||p6==0)&&(p4==0||p6==0||p8==0)) //p2*p4*p6=0 && p4*p6*p8==0
                        {
                        dst.at<uchar>(i,j) = 0; //满足删除条件，设置当前像素为0
                        ifEnd = true;
                        }
                    }
                }
            }
        }
    
    dst.copyTo(tmpimg);
    img = tmpimg.data;
    for(i = 1; i < height; i++)
        {
        img += step;
        for(j =1; j<width; j++)
            {
            A1 = 0;
            uchar* p = img + j;
            if( p[0] > 0)
                {
                if(p[-step]==0&&p[-step+1]>0) //p2,p3 01模式
                    {
                    A1++;
                    }
                if(p[-step+1]==0&&p[1]>0) //p3,p4 01模式
                    {
                    A1++;
                    }
                if(p[1]==0&&p[step+1]>0) //p4,p5 01模式
                    {
                    A1++;
                    }
                if(p[step+1]==0&&p[step]>0) //p5,p6 01模式
                    {
                    A1++;
                    }
                if(p[step]==0&&p[step-1]>0) //p6,p7 01模式
                    {
                    A1++;
                    }
                if(p[step-1]==0&&p[-1]>0) //p7,p8 01模式
                    {
                    A1++;
                    }
                if(p[-1]==0&&p[-step-1]>0) //p8,p9 01模式
                    {
                    A1++;
                    }
                if(p[-step-1]==0&&p[-step]>0) //p9,p2 01模式
                    {
                    A1++;
                    }
                p2 = p[-step]>0?1:0;
                p3 = p[-step+1]>0?1:0;
                p4 = p[1]>0?1:0;
                p5 = p[step+1]>0?1:0;
                p6 = p[step]>0?1:0;
                p7 = p[step-1]>0?1:0;
                p8 = p[-1]>0?1:0;
                p9 = p[-step-1]>0?1:0;
                if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7  &&  A1==1)
                    {
                    if((p2==0||p4==0||p8==0)&&(p2==0||p6==0||p8==0)) //p2*p4*p8=0 && p2*p6*p8==0
                        {
                        dst.at<uchar>(i,j) = 0; //满足删除条件，设置当前像素为0
                        ifEnd = true;
                        }
                    }
                }
            }
        }

    //如果两个子迭代已经没有可以细化的像素了，则退出迭代
    if(!ifEnd) break;
    }

}

void CloudProcessor::EndPointExtraction(const cv::Mat& src, vector<vector<cv::Point2i>>& raw_pts)
{
    int count = 0;
	int col_num = src.cols;
    int row_num = src.rows;
    auto dst = src;
    auto dst2 = src;
    cv::threshold(dst, dst, 100, 255, cv::ThresholdTypes::THRESH_BINARY);
    vector<Vector2i> endpt_vec, midpt_vec;

    for (int i = 0; i < row_num; ++i)
    {
        for (int j = 0; j < col_num;++ j)
        {  
            int p1 = (int)(src.at<uchar>(i, j));

            if (p1 == 0) 
                continue;
            int p2 = (i == 0) ? 0 : (int)(src.at<uchar>(i - 1, j) / 255);
            int p3 = (i == 0 || j == col_num - 1) ? 0 : (int)(src.at<uchar>(i - 1, j + 1) / 255);
            int p4 = (j == col_num - 1) ? 0 : (int)(src.at<uchar>(i, j + 1) / 255);
            int p5 = (i == row_num - 1 || j == col_num - 1) ? 0 : (int)(src.at<uchar>(i + 1, j + 1) / 255);
            int p6 = (i == row_num - 1) ? 0 : (int)(src.at<uchar>(i + 1, j) / 255);
            int p7 = (i == row_num - 1 || j == 1) ? 0 : (int)(src.at<uchar>(i + 1, j - 1) / 255);
            int p8 = (j == 0) ? 0 : (int)(src.at<uchar>(i, j - 1) / 255);
            int p9 = (i == 0 || j == 0) ?0 : (int)(src.at<uchar>(i - 1, j - 1) / 255);
            //cout<<p1<<"    "<<p2<<" "<<p3<<" "<<p4<<" "<<p5<<" "<<p6<<" "<<p7<<" "<<p8<<" "<<p9<<endl;
            if (( p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 1)
            {	
                //cout << "(y,x):" << "(" << i << "," << j << ")"<<endl;  
                Vector2i pt(i, j);
                endpt_vec.emplace_back(pt);
            } else if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) == 2)
            {
                /*if ((p2 == 1 && p4 == 1) || (p6 == 1 && p4 == 1) || (p2 == 1 && p8 == 1) || (p6 == 1 && p8 == 1)) {
                    Vector2i pt(i, j);
                    midpt_vec.emplace_back(pt);
                }
                if (((p3 == p7 == 1)&&(p5 == p9 == 0)) || ((p5 == p9 == 1) && (p3 == p7 == 0))) {
                    Vector2i pt(i, j);
                    midpt_vec.emplace_back(pt);
                }*/
                Vector2i pt(i, j);
                    midpt_vec.emplace_back(pt);
            }
        }
    }

    vector<cv::Point2i> raw_contour;

	for (const auto& pt : endpt_vec)
	{
        cv::Point center;
		center.y = pt(0);
		center.x = pt(1);	
		//cv::circle(dst, center, 6, cv::Scalar(155, 0, 0), -10);
        raw_contour.emplace_back(center);
	}
    
    for (const auto& pt : midpt_vec)
	{
        cv::Point center;
		center.y = pt(0);
		center.x = pt(1);	
		//cv::circle(dst, center, 6, cv::Scalar(155, 0, 0));
        raw_contour.emplace_back(center);
	}
    cv::imwrite("/home/sustech1411/img_endpts.png", dst);
    
    //euclidean clustering using lib-pcl
    vector<vector<cv::Point2i>> raw_contours, refined_contours;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    for (auto cv_pt : raw_contour) {
        pcl::PointXYZ pt;
        pt.x = cv_pt.x;
        pt.y = cv_pt.y;
        pt.z = 0;
        cloud_filtered->points.push_back(pt);
    }
    tree->setInputCloud(cloud_filtered);
 
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(3); 
	ec.setMinClusterSize(1);
	ec.setMaxClusterSize(2500);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);
    
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{   
        vector<cv::Point2i> point_vec;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
			cloud_cluster->points.push_back(cloud_filtered->points[*pit]); 
 
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
 
		/*std::cout << "当前聚类 "<<j<<" 包含的点云数量: " << cloud_cluster->points.size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".pcd";
		j++;*/

        for (auto pt : cloud_cluster->points) {
            cv::Point2i cv_pt;
            cv_pt.x = (int)(pt.x);
            cv_pt.y = (int)(pt.y);
            point_vec.emplace_back(cv_pt);
            //cout<<"pt.x="<<cv_pt.x<<" pt.y="<<cv_pt.y<<endl;
        }
        sort(point_vec.begin(), point_vec.end(), compare_dist);
        raw_contours.emplace_back(point_vec);
	}

    refined_contours.resize(raw_contours.size());
    for (int i = 0; i < raw_contours.size(); i++) {
        //refined_contours[i] = raw_contours[i];
        cv::approxPolyDP(raw_contours[i], refined_contours[i], poly_thresh, true);
    }
    for (const auto& vec : refined_contours)
	{
        for (const auto& pt : vec) {
            //cout<<"pt.x="<<pt.x<<" pt.y="<<pt.y<<endl;
            count++;
            cv::circle(dst2, pt, 6, cv::Scalar(155, 0, 0));
        }
        //cv::circle(dst2, cv::Point(420, 520), 15, cv::Scalar(155, 0, 0));
	}
    
    cv::imwrite("/home/sustech1411/img_endpts_afterpoly.png", dst2);
    cout<<count<<" pts before filter"<<endl;
    //cv::drawContours(dst, round_contours, idx, color, cv::LineTypes::LINE_4);
    raw_pts = refined_contours;
    

}

void CloudProcessor::AdjacentDistanceFilter(std::vector<std::vector<cv::Point2i>>& contoursIn) 
{
    std::vector<std::vector<cv::Point2i>> contoursOut;
    for (std::size_t i = 0; i < contoursIn.size(); i++) { 
        const auto c = contoursIn[i];
        const std::size_t c_size = c.size();
        std::vector<cv::Point2i> out_vec;

        if (c.size() < 2) continue;
        int idx = 0;

        for (std::size_t j = 0; j < c_size; j++) {
            cv::Point2i p = c[j]; 
            if (j > 0 && j < c_size - 1) {
                if (PtDiscard(c, j, out_vec[idx - 1])) continue;           
            } else {
                out_vec.emplace_back(p);
                idx++;
            }
        }
        contoursOut.emplace_back(out_vec);
        
    }
    contoursIn = contoursOut;
}

inline bool CloudProcessor::PtDiscard(const std::vector<cv::Point2i>& pt_vec, std::size_t idx, const cv::Point2i& pt_pre)
{
    auto dist = PixelDistance(pt_vec[idx], pt_pre);

    cv::Point2i diff_p1 = pt_pre - pt_vec[idx];
    cv::Point2i diff_p2 = pt_vec[idx + 1] - pt_vec[idx];
    diff_p1 /= std::hypotf(diff_p1.x, diff_p1.y);
    diff_p2 /= std::hypotf(diff_p2.x, diff_p2.y);

    if ((abs(diff_p1.dot(diff_p2)) < cos_thresh) && (dist < filter_thresh)) {
        return true;
    }
    return false;
}

void CloudProcessor::AdjecentDistanceFilter(std::vector<std::vector<cv::Point2i>>& contoursInOut) {
    /* filter out vertices that are overlapped with neighbor */
    std::unordered_set<int> remove_idxs;
    for (std::size_t i=0; i<contoursInOut.size(); i++) { 
        const auto c = contoursInOut[i];
        const std::size_t c_size = c.size();
        std::size_t refined_idx = 0;
        for (std::size_t j=0; j<c_size; j++) {
            cv::Point2i p = c[j]; 
            /*if (refined_idx < 1 || PixelDistance(contoursInOut[i][refined_idx-1], p) > filter_thresh) {
                
                RemoveWallConnection(contoursInOut[i], p, refined_idx);
                RemoveCloseConnection(contoursInOut[i], p, refined_idx, c_size);
                contoursInOut[i][refined_idx] = p;
                refined_idx ++;
            }*/
        }
        /** Reduce wall nodes */
        /*RemoveWallConnection(contoursInOut[i], contoursInOut[i][0], refined_idx);
        contoursInOut[i].resize(refined_idx);
        if (refined_idx > 1 && PixelDistance(contoursInOut[i].front(), contoursInOut[i].back()) < filter_thresh) {
            contoursInOut[i].pop_back();
        }*/
        if (contoursInOut[i].size() < 2) 
            remove_idxs.insert(i);
    }
    if (!remove_idxs.empty()) { // clear contour with vertices size less that 3
        std::vector<vector<cv::Point2i>> temp_contours = contoursInOut;
        contoursInOut.clear();
        for (int i = 0; i < temp_contours.size(); i++) {
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
            InternalContoursIdxs(refined_hierarchy, i, remove_idxs);
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
      //cout<<"pointxyZ: "<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<pt.intensity<<" row="<<row_i<<" col="<<col_i<<" range="<<range<<endl; 
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