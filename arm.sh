#!/bin/bash

cd ~/jetson_ctrl_ws
export ROS_MASTER_URI=http://10.42.0.49:11311
export ROS_IP=10.42.0.1
source devel/setup.bash
cd src
cd edoProjectFinal
python armManipulation.py
