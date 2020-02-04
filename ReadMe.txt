=================
  Prerequisites
=================
* e.DO is on
* e.DO's brakes are disengaged
* You are in home directory (the default for opening a new ternimal)

===================================
  Steps for Initial System Setup
===================================

(1) Position the object in desired location in the camera range.

#Running Gazebo
(2) Open a terminal and type "cd jetson-reinforcement/build/aarc64/bin/"
(3) Type "./gazebo.sh 0 0"

#Running the camera
(4) Open a new terminal and type "cd jetson-reinforcement/build/aarc64/bin" 
(5) Type "python finalP1.py"

#Setting up the ROS IP's
(6) Open a new terminal and type "cd jetson_ctrl_ws"
(7) Type "export ROS_MASTER_URI=http://10.42.0.49:11311"
(8) Type "export ROS_IP=10.42.0.1
(9) Type "source devel/setup.bash"

#Getting the arm Manipuation running
(10) Type "cd src/edoProjectFinal"
(11) Type "python armManipulation.py"


====================================
  Instructions for Grabbing Object
====================================

(1) In the Camera Terminal, press 'y' to get object location whenever you move the object.
(2) Wait for a win to be displayed from the Roscore terminal.
(3) In the Arm Terminal, enter 'y' to have e.DO grab the object.

