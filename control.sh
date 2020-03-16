#!/bin/bash

qjoypad --notray
cd ~/jetson_ctrl_ws
export ROS_MASTER_URI=http://192.168.12.1:11311
export ROS_IP=192.168.12.68
source devel/setup.bash
rosrun edo_manual_ctrl edo_manual_ctrl

gnome-terminal -e printf "                  e.DO manual jog gamepad controls
======================================================================
Matricom G-Pad
Ubuntu 16.04, QJoyPad 4.1
----------------------------------------------------------------------

[] = Keyboard key

Joint 1 +/-: [q]/[a] -> left analog l/r
Joint 2 +/-: [w]/[s] -> left analog down/up
Joint 3 +/-: [e]/[d] -> L1/R1
Joint 4 +/-: [r]/[f] -> right analog l/r
Joint 5 +/-: [t]/[g] -> right analog down/up
Joint 6 +/-: [y]/[h] -> d-pad l/r

Gripper open/close: [i]/[k] -> button A/button B
Velocity +/-: [u]/[j] -> d-pad u/d
Exit: [x] -> button X

-	[y] -> button Y
-	[Enter] -> Start
-	[Backspace] -> Select
-	[Ctrl] -> left analog press
-	[C] -> right analog press
-	[Z] -> Mode\n"

