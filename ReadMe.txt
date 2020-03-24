======================================================
     Comau e.DO Project Operation Guidelines v2.0
                   Updated 3-24-2020
======================================================

This document will guide you through the setup and operation of e.DO in the Comau Lift lab.

Currently, the project is fairly modular, meaning it is comprised of a few moving parts that can be run independently of each other. To that end, there is one file to launch all of these components at once in addition to their respective individual launch files.

=============
Prerequisites
=============
1. e.DO must be turned on and its brakes disengaged via the Android app or manual control. 
2. Additionally, the TX2 should be using the ethernet connection.

---------------------------------------------------------------------------------------------------------------

==============================================
Calibrating e.DO using the Android application
==============================================
1. Press the green switch on the side of e.DO's base to turn it on.
2. On the tablet, connect to e.DO's wireless connection, "edowifi240.xxx...." (Password: edoedoedo; it may take a while for this to appear after first turning on e.DO).
3. Launch the e.DO app.
4. Select the option to connect to the default IP address.
5. Press the button to disengage e.DO's brakes.
6. Select the option to begin calibrating the joints.
7. Go through each joint in the app, pressing the + and - buttons until the white marks on the robot's joints are aligned with the corresponding engravings.

Connection errors may occur at any point during this process, which will require you to turn off e.DO for about 30 seconds and start over.


===============================
Running the entire main project
===============================
1. To start, we will assume that you would like to run everything, which includes the camera, simulation, and arm manipulation. To do so, open a terminal (Ctrl + Alt + T) and navigate to the launcher directory by entering the following command: 

"cd jetson-reinforcement/build/aarc64/bin/" (No quotation marks)

Alternatively, you can navigate to that directory in the GUI, right-click in an open space, and select "Open in terminal".

2. In that same terminal, run the launch script by entering the following command:

"./everything.sh"

From here, several new windows and terminals will open, indicating a successful launch.


==================
Running the camera
==================
1. To run the camera individually, perform the same steps detailed above, except instead of entering the command "./everything.sh", enter "./camera.sh".


======================
Running the simulation
======================
1. To run the simulation individually, perform the same steps detailed above, except instead of entering the command "./camera.sh", enter "./gazebo-edo.sh 0 0".

2. While running the simulation, it is useful to take advantage of the TX2's GPUs. To do so, navigate to the home directory in a terminal ("cd ~/") and enter the following command:

"./jetson_clocks.sh"
nvidia sudo password: "nvidia"

(Currently, this is unable to be done automatically when launching everything.sh.) 


===============
Running the arm
===============
1. To run the arm manipulation individually, perform the same steps detailed above, except instead of entering the command "./gazebo-edo.sh 0 0", enter "./arm.sh".

---------------------------------------------------------------------------------------------------------------

=========
Operation
=========
From here, the operation of the various components is fairly self-explanatory from within their respective terminals and other documentation. The basic flow is as follows:

1. The camera detects an object, sending its coordinates to the simulation.
2. Once the simulated e.DO has achieved a win at this object position, its angles are updated.
3. You issue the command in the Arm Terminal for the physical e.DO to attempt to grab the obect (options 1-4).

---------------------------------------------------------------------------------------------------------------

======================
Manual control
======================
1. Not part of the main project is the ability to control e.DO manually using the keyboard and/or a controller. To do so, connect to the wireless access point "edowifi240.xxx...." (Password: edoedoedo).

2. Next, perform the same steps as above, except this time, enter "./control.sh".

***Important note: At the time of writing, the control.sh script is not on the TX2. It will need to be copied to the directory with the other scripts and have its permissions modified with the following command: 

"sudo chmod -R 777 control.sh"
nvidia sudo password: "nvidia"

Currently, there is no controller in the lab (I used my own). If you end up using one, you will need to create a new QJoyPad layout to map the keyboard keys to it.

---------------------------------------------------------------------------------------------------------------


Have fun!

Nate Smith
e.DO Team Winter 2020
