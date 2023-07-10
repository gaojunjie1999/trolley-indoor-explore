#ifndef LIDAR_POINT_H
#define LIDAR_POINT_H 

#include<pcl/point_types.h>
#include<pcl/point_cloud.h>

namespace pcl
{
struct PointXYZITR
{
    PCL_ADD_POINT4D;          
    float intensity;
    uint16_t ring;
    uint16_t pcaketnum;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  
} EIGEN_ALIGN16;       
}
 
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZITR,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint16_t,pcaketnum,pcaketnum)
                                  )

#endif