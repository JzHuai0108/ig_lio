# Livox publish pointcloud msg format. copied from https://github.com/Livox-SDK/livox_ros_driver2/blob/master/msg/CustomMsg.msg at Jul 3 2025.

std_msgs/Header header    # ROS standard message header
uint64 timebase           # The time of first point
uint32 point_num          # Total number of pointclouds
uint8  lidar_id           # Lidar device id number
uint8[3]  rsvd            # Reserved use
CustomPoint[] points      # Pointcloud data

