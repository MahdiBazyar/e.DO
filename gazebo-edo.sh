#!/bin/sh

# USAGE: ./gazebo-edo.sh CAMOPTION SERVEROPTION
# CAMOPTION:
#   0: Use single camera
#   1: Use double camera 2 images
# SERVEROPTION:
#   0: Create local gazebo server
#   1: Connect to e.DO ROS server over LAN
# To start on e.DO's ROS Server, run "./gazebo-edo.sh 0 1

echo "Single/Double set to $1"
echo $1 > select.txt

echo "Loading file '$3'"
echo "$3" >> select.txt

if [ ! -d "drqn_checkpoints" ]; then
  echo "No checkpoint directory found. Creating directory 'drqn_checkpoints'"
  mkdir drqn_checkpoints
else
  echo "Found checkpoint directory 'drqn_checkpoints'"
fi  

echo " "
echo "configuring Gazebo7 plugin paths"
echo "previous GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "script directory $SCRIPT_DIR"

MY_PLUGIN_PATH=$SCRIPT_DIR/../lib
echo "plugin path $MY_PLUGIN_PATH"

export GAZEBO_PLUGIN_PATH=$MY_PLUGIN_PATH:$GAZEBO_PLUGIN_PATH
echo "GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"
echo " "

if [ "$2" -eq 1 ]; then
  echo "Using e.DO ROS Server"
  echo "===================="
  export ROS_MASTER_URI=http://10.42.0.49:11311
  echo "ROS_MASTER_URI=$ROS_MASTER_URI"
  export ROS_IP=10.42.0.21
  echo "ROS_IP=$ROS_IP"
else
  echo "Starting ROSCORE on Local Machine"
  roscore &
fi

echo " "

echo "starting Gazebo7 Client (gzclient)"
gnome-terminal -e 'sh -c "echo \"\033]0; Gazebo7 Client (gzclient)\007\"; \
				echo \"launching Gazebo7 Client (gzclient)\"; \
				echo \"Press Ctrl+Q or close window to quit\n\"; \
				sleep 2; \
				gzclient --verbose; \
				pkill gzserver"' # pkill -INT gzserver

echo "starting Gazebo7 Server (gzserver) with ROS compatibility\n"
rosrun gazebo_ros gzserver gazebo-edo.world --verbose

echo "Gazebo7 Server (gzserver) has exited."
if [ "$2" -ne 1 ]; then
  killall roscore
  sleep 2
fi
#cd
#cd ~/jetson_ctrl_ws
#export ROS_MASTER_URI=http://10.42.0.49:11311
#export ROS_IP=10.42.0.1
#source devel/setup.bash
#cd src
#cd edoProjectFinal/
#python3 armManipulation.py 

