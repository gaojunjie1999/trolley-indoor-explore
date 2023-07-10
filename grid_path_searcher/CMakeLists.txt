cmake_minimum_required(VERSION 2.8.3)
project(velodyne_pcl_parser)

find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    std_msgs
    nav_msgs
    visualization_msgs 
    cv_bridge
    message_filters
    tf
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
    include 
    SYSTEM 
    third_party
    ${catkin_INCLUDE_DIRS} 
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -O3 -Wall") # -Wextra -Werror

add_executable(velodyne_pcl_parser
    src/demo_node.cpp
    src/ground_remove2.cpp
    )

target_link_libraries(velodyne_pcl_parser 
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES} 
    ${OpenCV_LIBRARIES}
)

