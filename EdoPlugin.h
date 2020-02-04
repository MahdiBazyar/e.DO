/** @file EdoPlugin.h
 *  @brief Class definition for EdoPlugin
 *  @author Ashwini Magar, Jack Shelata, Alessandro Piscioneri
 *  @date June 1, 2018
 */
#ifndef __GAZEBO_EDO_PLUGIN_H__
#define __GAZEBO_EDO_PLUGIN_H__

#include "deepRL.h"

#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>
#include <iostream>
#include <gazebo/transport/TransportTypes.hh>
#include <gazebo/msgs/MessageTypes.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Plugin.hh>
#include <ros/ros.h>                          // For Live ROS communication
#include "std_msgs/Int8MultiArray.h"

#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <syslog.h>
#include <time.h>
#include <ostream>      // added for outputting text to file to be
                        // for ROS commands to e.DO
#include <fstream>      // added for reading input from text file
                        // for image number selection
//#include "MovementCommandQueue.h"
#include <vector>

namespace gazebo
{

/***************************************************************
**                Class(es) Definition
****************************************************************/

/** @brief EdoPlugin Class definition
 */
class EdoPlugin : public ModelPlugin
{
public:
  EdoPlugin();

  // Read and import the eDo model from ~/.gazebo/models/edo_sim
  virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/);

  // Callback function called by the Gazebo World constantly
  virtual void OnUpdate(const common::UpdateInfo & /*_info*/);

  // Center servo positions
  float resetPosition( uint32_t dof );

  // Create the agent
  bool createAgent();

  // Load existing agen
  bool loadAgent(std::string filename);

  //  created. Save the agent
  bool saveAgent(std::string filename);

  // Update the agent
  bool updateAgent();

  // Update the joints of the agent
  bool updateJoints();

  // Callback function for when new image is recieved from the virtual camera
  void onCamerasMsg(ConstImagesStampedPtr &_msg);

  // Callback function for when new image is recieved from the virtual camera
  void onCameraMsg(ConstImageStampedPtr &_msg);

  // Callback function for when there is a collision of any kind
  void onCollisionMsg(ConstContactsPtr &contacts);

  // Active degrees of freedom in the eDo
  // Switch to 2 and change joint controller to do 2D, 2 joint training
  static const uint32_t DOF = 6;//changed 3 to 5 MAHMOUD ELMASRI

private:
  float ref[DOF];               // Joint reference postions
  float vel[DOF];               // Joint velocity control
  float dT[3];                  // Inverse Kinematic delta theta

  rlAgent*  agent;              // AI learning agent instance
  bool      newState;           // True if a new frame needs to be processed
  bool      newReward;          // True if a new reward's been issued
  bool      endEpisode;         // True if this episode is over
  float     rewardHistory;      // Value of the last reward issued
  float     totalReward;       // total of all rewards per episode
  Tensor*   inputState;         // pyTorch input object to the agent
  void*     inputBuffer[2];     // [0] for CPU and [1] for GPU
  size_t    inputBufferSize;
  size_t    inputRawWidth;
  size_t    inputRawHeight;
  float     jointRange[DOF][2]; // Min/Max range of each arm joint
  float     actionJointDelta;   // Amount of offset caused to a joint by an
                                // action
  float     actionVelDelta;     // Amount of velocity offset caused to a joint
                                // by an action
  int       maxEpisodeLength;   // Maximum number of frames to win an episode
                                // (or <= 0 for unlimited)
  int       episodeFrames;      // Frame counter for the current episode
  bool      testAnimation;      // True for test animation mode
  bool      loopAnimation;      // Loop the test animation while true
  uint32_t  animationStep;      //
  float     resetPos[DOF];      //
  float     lastGoalDistance;   //
  float     avgGoalDelta;       //
  int       successfulGrabs;    //
  int       totalRuns;          //
  int       runsFromLastWin;    //
  int       runHistoryIdx;      //
  int       runHistoryMax;      //
  bool      runHistory[20];     //
  bool      needsRandom;        // (Created) Flag for randomizing
                                // prop location
  int       runsSinceRandom;    // (Created) Flag to wait at least
                                // 20 episodes
                                // before randomizing again
  bool      endSequence;        // Created for outputting end of
                                // sequence
  bool      winState;           //cory added
  float     real_score;
  float     best_score;
  ros::NodeHandle nh;           // ROS NodeHandle for sending joint angles
  ros::Publisher test_pub;      // ROS Publisher to publish joint angles
  std_msgs::Int8MultiArray winSequence;       // Message to publish sequence
  std::vector<int> winSequenceVec;            // Vector to hold angles while
                                              // sequence is in progress
  std::string prevJointValues;                // Created so that outputted
                                              // joint angles are all different

  physics::ModelPtr model;                    // Pointer to the eDo model
  event::ConnectionPtr updateConnection;      //
  physics::JointController* j2_controller;    // Pointer to the Joint Controller

  gazebo::transport::NodePtr cameraNode;      // Pointer to the camera node
  gazebo::transport::SubscriberPtr cameraSub; // Pointer to camera subscriber
  gazebo::transport::SubscriberPtr cameraSubSecond; // Pointer to camera subscriber //New edo team code 
  gazebo::transport::NodePtr collisionNode;   // Pointer to the collision node
  // Pointer to the collision subscriber for prop
  gazebo::transport::SubscriberPtr collisionPropSub;

  std::ofstream outfile;        // added for writing to file
  std::ifstream infile;         // added for reading from file
  bool select;                  // added to select single or
                                // double image
  std::string loadfile;         // file to load specified in CLA
  int INPUT_HEIGHT,INPUT_WIDTH; // Change to int to allow for selection
                                // at runtime

  float endJoints[6] = {0, 28.65, 0, 0, 0, 0};
  float newEndJoints[6] = {0, 28.65, 0, 0, 0, 0};              // cory added to stor cumulated joint angles
};

} // namespace gazebo


#endif
