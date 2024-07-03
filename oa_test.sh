#!/bin/bash

# Loop 500 times using index i
for ((i=1; i<=500; i++))
do
    cd /sbel/chrono_fork/build
    # Run the first command in the background
    ./bin/demo_ROS_dART_SCM straight_line_x &
    # Save the PID of the last background process
    pid1=$!

    # Sleep for 2 seconds
    sleep 7

    cd /sbel/Desktop/ros_ws/

    # Run the second command in the background, passing i as a parameter
    ros2 run lidar_obstacle_detect_avoid lidar_ODA --ros-args -p use_sim_time:=true -p exp_index:=$i &

    # Wait for 40 seconds
    sleep 100

    # Kill both processes
    kill -9 $pid1 
    ps -A | grep -e "ros2"  | awk '{print $1}' | xargs kill -2
    ps -A | grep -e "lidar_ODA"  | awk '{print $1}' | xargs kill -9


done
