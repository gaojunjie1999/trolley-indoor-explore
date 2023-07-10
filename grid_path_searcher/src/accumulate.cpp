#include "ros/ros.h"
#include <math.h>
#include <iostream>
#include <cmath>
#include "std_msgs/String.h"
#include <sensor_msgs/Imu.h>
#include <sbg_driver/SbgImuData.h>
#include <sbg_driver/SbgEkfQuat.h>
#include <sbg_driver/SbgGpsPos.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include "tf/transform_datatypes.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/visualization/cloud_viewer.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
  


#define PI 3.1415926

//source devel/setup.bash
//rosrun sbg_driver ros_imu_gps
//rosrun sbg_driver velodyne_imugps
//rosbag play /home/xx/catkin_lidar_imu_gps/2020-11-02-14-17-08.bag

using namespace sensor_msgs;
using namespace std;
ros::Publisher cloud_pub;
//ros::Publisher velodyne_pub;
int i=0;//调试
int j=1 ;//控制转雷达坐标系的循环
int k=0,kk=1,k_=0;//控制5帧循环
double gx,gy,gz ;//转到初始坐标系的平移基准
 
Eigen::Matrix4d matrix_1 ;
Eigen::Matrix4d matrix_2 ;
Eigen::Matrix4d matrix_3 ;
Eigen::Matrix4d matrix_4 ;
Eigen::Matrix4d matrix_5 ;
Eigen::Matrix4d matrix_6 ;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
pcl::PointCloud<pcl::PointXYZ> cloud_new1; 
pcl::PointCloud<pcl::PointXYZ> cloud_new2;
pcl::PointCloud<pcl::PointXYZ> cloud_new3; 
pcl::PointCloud<pcl::PointXYZ> cloud_new4; 
pcl::PointCloud<pcl::PointXYZ> cloud_new5; 
pcl::PointCloud<pcl::PointXYZ> cloud_new6; 

////////////////////////// 把经纬坐标投影到Web墨卡托 /////////////////////////sys

void lonLat2WebMercator(double lon,double lat,double *x,double *y)
{
   *x = lon * 20037508.34 / 180;
   double ly = log(tan((90+ lat)*PI/360))/(PI/180);
   *y =  ly *20037508.34/180;

}

void GetDirectDistance(double srcLon, double srcLat,double gz, double destLon, double destLat,double tz,double *x,double *y,double *z)
{
     double x1,x2,y1,y2;
     lonLat2WebMercator(srcLon,srcLat,&x1,&y1);
     lonLat2WebMercator(destLon,destLat,&x2,&y2);
     
     //cout<<"x1="<<x1<<endl;
     //cout<<"x2="<<x2<<endl;
     *x=x2-x1;
     *y=y2-y1;
     *z=tz-gz;
     
}
 
////////////////////////////  回调函数 ////////////////////////////////////////////////////////sys
pcl::visualization::CloudViewer viewer("viewer");
void chatterCallback(const  boost::shared_ptr<const sensor_msgs::PointCloud2>& pc2,const  boost::shared_ptr<const sensor_msgs::Imu>& msg)
{
  i++;
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg (*pc2, cloud);//cloud is the output
  sensor_msgs::PointCloud2 cloud_publish;
 if(k==0)
 {
   //cout<<"第一次"<<endl;
   gx = msg->angular_velocity.y;//把gps值放到这里
   gy = msg->angular_velocity.x;
   gz = msg->angular_velocity.z;
   //cout<<"gx"<<gx<<endl;//45.7142
   //cout<<"gy"<<gy<<endl;//126.628
   //cout<<"gz"<<gz<<endl;//141.751
   k++;
 }
  if (k!=0)
  {
    // cout<<"iiiiiiii="<<i<<endl;
     
      
/////////////////////// 提取出四元数与gps ///////////////////////////////////

     double x,y,z,w,tx,ty,tz;
     x = msg->orientation.x;//四元数
     y = msg->orientation.y; 
     z = msg->orientation.z; 
     w = msg->orientation.w;
     tx = msg->angular_velocity.y;//把gps值放到这里
     ty = msg->angular_velocity.x;
     tz = msg->angular_velocity.z; 
     //cout<<" x="<< x<<endl;
     //cout<<" y="<< y<<endl;
     //cout<<" z="<< z<<endl;
     //cout<<" w="<< w<<endl;

     ///////////////////// 平移计算  /////////////////////////////////////////////

      double nx,ny,nz;
      GetDirectDistance(gx,gy,gz,tx,ty,tz,&nx,&ny,&nz);//计算GPS变化量
      //cout<<"nx= "<<nx<<endl;
      //cout<<"ny= "<<ny<<endl;
      //cout<<"nz= "<<nz<<endl;

 //////////////////////// 四元数转换为旋转矩阵 /////////////////////////////////// 
  
     Eigen::Quaterniond quaternion(w,x,y,z);
     /*
     Eigen::Quaterniond quaternion;  
     quaternion.x() = x;  
     quaternion.y() = y;  
     quaternion.z() = z;  
     quaternion.w() = w; 
     */
     Eigen::Matrix3d rotation_matrix;
     rotation_matrix=quaternion.toRotationMatrix();//quaternion.matrix();
     //cout<<"rotation_matrix="<<rotation_matrix<<endl;
     Eigen::Matrix4d T_matrix;//4x4旋转矩阵赋值
     T_matrix(0,0)=rotation_matrix(0,0);
     T_matrix(1,0)=rotation_matrix(1,0);
     T_matrix(2,0)=rotation_matrix(2,0);
     T_matrix(3,0)=0;
     T_matrix(0,1)=rotation_matrix(0,1);
     T_matrix(1,1)=rotation_matrix(1,1);
     T_matrix(2,1)=rotation_matrix(2,1);
     T_matrix(3,1)=0;
     T_matrix(0,2)=rotation_matrix(0,2);
     T_matrix(1,2)=rotation_matrix(1,2);
     T_matrix(2,2)=rotation_matrix(2,2);
     T_matrix(3,2)=0;
     T_matrix(0,3)=ny;
     T_matrix(1,3)=nx;
     T_matrix(2,3)=-nz;
     T_matrix(3,3)=1;
     //cout<<"T_matrix="<<T_matrix<<endl;
   /*
      rotation_matrix=   -0.132107     0.991235 -7.72432e-05
                         -0.990425    -0.131996    0.0404286
                          0.0400641   0.00541743     0.999182

      rotation_matrix=   0.955879     0.29309   0.0198578
                         -0.293673    0.955069   0.0399872
                         -0.00724572  -0.0440546    0.999003

      T_matrix=           0.955879     0.29309   0.0198578     15.7504
                          -0.293673    0.955069   0.0399872    -4.96262
                        -0.00724572  -0.0440546    0.999003    0.794558
                             0           0           0           1
    */
 //////////////////// 四元数转换为欧拉角 ///////////////////////////////////
   /*
      Eigen::Quaterniond quaternion(w,x,y,z);
      //Eigen::Vector3d eulerAngle(yaw,pitch,roll);
      //绕z轴yaw偏航  绕y轴pitch俯仰(不能超过90度)   绕x轴roll:横滚
      Eigen::Vector3d eulerAngle=quaternion.matrix().eulerAngles(2,1,0);//初始化欧拉角(Z-Y-X )

      //////////////// 欧拉角算旋转矩阵 ////////////////////////////////////
      Eigen::Matrix3d rotation_matrix;
      //rotation_matrix=yawAngle*pitchAngle*rollAngle;
      rotation_matrix = Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitZ()) * 
                        Eigen::AngleAxisd(eulerAngle[1], Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitX()); 
     //cout<<"rotation_matrix="<<rotation_matrix<<endl;
     */
     /* 
      rotation_matrix=  -0.132107    0.991235 -7.7243e-05
                        -0.990425   -0.131996   0.0404286
                        0.0400641  0.00541743    0.999182    //zyx
     */
    
   ///////////////////// 求旋转矩阵的逆 ////////////////////////////////////////
      
     if(j==1)
     { matrix_1 =T_matrix.inverse();}//求逆
     
     if(j==2)
     { matrix_2 =T_matrix.inverse();}//求逆
     if(j==3)
     { matrix_3 =T_matrix.inverse();}
     if(j==4)
     { matrix_4 =T_matrix.inverse();}
     if(j==5)
     { matrix_5 =T_matrix.inverse();}
     if(j==6)
     {
       matrix_6 =T_matrix.inverse();//求逆 
       j=0;
     }
     j++;
     //cout<<"matrix_1="<<matrix_1<<endl;
         
   ////////////////////////// 点云处理  ////////////////////////////////////////
   
       float x_pass;
       float y_pass;
       float z_pass;  
        
       for (int i = 0; i <cloud.size(); i++)
       {
         // 按惯导——雷达坐标赋值//点云变成惯导坐标系中的点
        /*  
           x_pass=-1*cloud.points[i].y+0.14;
           y_pass=cloud.points[i].z;
           z_pass=-1*cloud.points[i].x-0.3;
         */
         
        // 按飞机——雷达坐标赋值//点云变成机体坐标系中的点
           
           x_pass= cloud.points[i].z ; 
           y_pass=-cloud.points[i].y+0.14;
           z_pass= cloud.points[i].x+0.30;
             
           ///////////////////////////////////////////
     
           //把点云坐标转成矩阵
           Eigen::MatrixXd m(4,1);
           m(0,0)=x_pass;
           m(1,0)=y_pass;
           m(2,0)=z_pass;
           m(3,0)=1;
           //cout<<" x_pass= "<< x_pass<<endl;
           //Eigen::MatrixXd n=matrix_33*m;// 矩阵与旋转矩阵相乘
           //Eigen::MatrixXd n=rotation_matrix*m;
            Eigen::MatrixXd n=T_matrix*m;
           // cout<<" n= "<< n<<endl;
   
           /////////////// 把计算后的坐标赋值给点云 ////////////////
           // 按惯导——雷达坐标赋值
        /*
            x_pass= n(0,0) +ny;
            y_pass= n(1,0) -nz;
            z_pass= n(2,0) +nx;
        */
           // 按飞机——雷达坐标赋值
           
           // x_pass= n(0,0) ;//+ny;
            //y_pass= n(1,0) ;//+nx;
           // z_pass= n(2,0) ;//-nz;
           
           // pcl::PointXYZ point_1(  n(0,0),  n(1,0),  n(2,0)  );
            //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            /*
            ///////////////// 2帧迭代写入 ///////////////// 
            if(k==3||k==4) 
            { 
              cloud_new3.push_back(point_1);
              
            }if(k==2||k==3)
            {
              cloud_new2.push_back(point_1);
            }if(k==1||k==2||k==4)
            {
              cloud_new1.push_back(point_1);
            }*/
            ///////////////// 5帧迭代写入(全转回初始坐标系) ///////////////// 
     /*         
            if(k==6||k==7||((k>=2&&k<=4)&&(k_==1)))
            { 
              cloud_new6.push_back(point_1);
              
            }if((k>=5&&k<=7)||((2<=k<=3)&&(k_==1)))
            {
              cloud_new5.push_back(point_1);
            }if((k>=4&&k<=7)||(k==2&&k_==1))
            {
              cloud_new4.push_back(point_1);
            }
            if(k>=3&&k<=7) 
            { 
              cloud_new3.push_back(point_1);
              
            }if(k>=2&&k<=6)
            {
              cloud_new2.push_back(point_1);
            }if( k>=1&&k<=5||k==7) 
            {
              cloud_new1.push_back(point_1);
            }
      */       
            //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
             
            ///////////////// 5帧迭代写入(转回雷达坐标系) ///////////////// 

             //x=n_6(0,0) &惯导坐标系    x=n_6(2,0)-0.3 &雷达坐标系
             //y=n_6(1,0)              y=0.14-n_6(1,0)
             //z=n_6(2,0)              z=n_6(0,0)

            if(k==6||k==7||((k>=2&&k<=4)&&(k_==1)))
            { 
              Eigen::MatrixXd n_6=matrix_6*n;// 与旋转矩阵的逆相乘
              pcl::PointXYZ point_6( (n_6(2,0)-0.3),  (0.14-n_6(1,0)), n_6(0,0) );
              cloud_new6.push_back(point_6);
              
            }if((k>=5&&k<=7)||((k==2||k==3)&&(k_==1)))
            {
              Eigen::MatrixXd n_5=matrix_5*n;// 与旋转矩阵的逆相乘
              pcl::PointXYZ point_5((n_5(2,0)-0.3),  (0.14-n_5(1,0)),  n_5(0,0) );
              cloud_new5.push_back(point_5);
            }if((k>=4&&k<=7)||(k==2&&k_==1))
            {
              Eigen::MatrixXd n_4=matrix_4*n;// 与旋转矩阵的逆相乘
              pcl::PointXYZ point_4((n_4(2,0)-0.3),  (0.14-n_4(1,0)),  n_4(0,0) );
              cloud_new4.push_back(point_4);
            }
            if(k>=3&&k<=7) 
            { 
              Eigen::MatrixXd n_3=matrix_3*n;// 与旋转矩阵的逆相乘
              pcl::PointXYZ point_3((n_3(2,0)-0.3),  (0.14-n_3(1,0)),  n_3(0,0) );
              cloud_new3.push_back(point_3);
              
            }if(k>=2&&k<=6)
            {
              Eigen::MatrixXd n_2=matrix_2*n;// 与旋转矩阵的逆相乘
              pcl::PointXYZ point_2((n_2(2,0)-0.3),  (0.14-n_2(1,0)),  n_2(0,0) );
              cloud_new2.push_back(point_2);
            }if( k>=1&&k<=5||k==7) 
            
            {
              Eigen::MatrixXd n_1=matrix_1*n;// 与旋转矩阵的逆相乘
              pcl::PointXYZ point_1((n_1(2,0)-0.3),  (0.14-n_1(1,0)),  n_1(0,0) );
              cloud_new1.push_back(point_1);
            }
            
            ////////////////// 调试-转换到第一个坐标系 /////////////////////////
     /*  
            Eigen::MatrixXd n_1=matrix_1*n;// 与旋转矩阵的逆相乘
            pcl::PointXYZ point_1(  n_1(0,0),  n_1(1,0),  n_1(2,0) );
            cloud_new1.push_back(point_1);
      */   
            /*
            //////////////// 调试3帧拼接 /////////////////////////
            if( k==40||k==70 ||k==100||k==130) //k==65
            {
              cloud_new1.push_back(point_1);
            }
            */
             
       }//for
       
///////////////// 2帧迭代 ///////////////sys
   /* if(kk==4)
     {
       cloud_ptr=cloud_new3.makeShared();
       viewer.showCloud(cloud_ptr);
       cloud_new2.clear(); 
       k_++;
       kk=1; 
     }if(kk==3 )
     {
       cloud_ptr=cloud_new2.makeShared();
       //viewer.showCloud(cloud_ptr);
       cloud_new1.clear();  
        
     }if((kk==1&&k_==0)||kk==2)
     {
       cloud_ptr=cloud_new1.makeShared();
       //viewer.showCloud(cloud_ptr);
       cloud_new3.clear(); 
     }
*/
////////////////////// 5帧迭代  /////////////////////////////
    
    if(kk==10)
     {
       cloud_ptr=cloud_new6.makeShared();
       pcl::toROSMsg(cloud_new6, cloud_publish);
      
       viewer.showCloud(cloud_ptr);//5
       cloud_new5.clear(); 
       kk=4; 
     }if(kk==9)
     {
       cloud_ptr=cloud_new5.makeShared();
       pcl::toROSMsg(cloud_new5, cloud_publish);
       viewer.showCloud(cloud_ptr);//5
      
       cloud_new4.clear();  
        
     }if(kk==8)
     {
       cloud_ptr=cloud_new4.makeShared();//5
       pcl::toROSMsg(cloud_new4, cloud_publish);
       viewer.showCloud(cloud_ptr);
       cloud_new3.clear(); 
     }
      if(kk==7)
     {
       cloud_ptr=cloud_new3.makeShared();//5
       pcl::toROSMsg(cloud_new3, cloud_publish);
       viewer.showCloud(cloud_ptr);
       cloud_new2.clear(); 
       
       
     }if(kk==6 )
     {
       cloud_ptr=cloud_new2.makeShared();
       pcl::toROSMsg(cloud_new2, cloud_publish);
       viewer.showCloud(cloud_ptr);//5
       cloud_new1.clear();  
        
     }if((kk>=1&&kk<=3)||kk==5||(kk==4&&k_==0)) 
     {
       cloud_ptr=cloud_new1.makeShared();
       pcl::toROSMsg(cloud_new1, cloud_publish);
       viewer.showCloud(cloud_ptr);
       if(kk==5)
       {  cloud_new6.clear(); }
       
     }
     ////////////// 控制循环 /////////////////////////
      if(k==7)
      {  k=1; 
         k_=1;
      }

      kk++;
      k++;
    /////////////// 发布点云消息 //////////////////////
      cloud_publish.header.stamp = msg->header.stamp;
      cloud_publish.header.frame_id = "cloud_new";
      cloud_pub.publish(cloud_publish); 
      ////////////////// 调试 /////////////////////////sys
      //cloud_ptr=cloud_new1.makeShared();
      //viewer.showCloud(cloud_ptr);
      //pcl::io::savePCDFileASCII("/home/xx/200.pcd",*cloud_ptr);
   
  } //else
  
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ros_velodyne");
  ros::NodeHandle n;
  cloud_pub = n.advertise<sensor_msgs::PointCloud2>("cloud_new", 1000);
  
  //ros::Subscriber sub = n.subscribe("imu_data", 1000, chatterCallback);

 //message_filters::Subscriber<sensor_msgs::PointCloud2>velodyne_sub(n,"velodyne_points", 1);
 message_filters::Subscriber<sensor_msgs::PointCloud2> lidar(n,"/velodyne_points",1000);             // topic1 输入
 message_filters::Subscriber<sensor_msgs::Imu> imu_gps(n,"/imu_gps",1000);   // topic2 输入

 typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Imu> MySyncPolicy;
 message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(1000),lidar, imu_gps); //queue size=10      // 同步
 sync.registerCallback(boost::bind(&chatterCallback, _1, _2));                   // 回调
 cout<<"hello world"<<endl;
  
 //velodyne_pub = n.advertise<sensor_msgs::PointCloud2>("new_velodyne_points", 20);  
  //ros::Rate loop(5);
  ros::spin();
  return 0;
}