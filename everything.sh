#!/bin/bash


cd ~/jetson-reinforcement/build/aarc64/bin

# Run Gazebo/ROS terminal
gnome-terminal -e ./gazebo-edo.sh 0 0

# Run camera terminal
gnome-terminal -e ./camera.sh

# Source IPs and run arm terminal
gnome-terminal -e ./arm.sh
