/** @file EdoPlugin.cpp
 *  @brief Plugin for e.DO robot and Gazebo simulator for reinforced learning.
 *  Resources: http://github.com/dusty-nv/jetson-reinforcementg
 *  @author Ashwini Magar, Jack Shelata, Alessandro Piscioneri
 *  @date June 1, 2018
 */
#define FILE_VERSION  1   /**< File version number */

#include "EdoPlugin.h"
#include "PropPlugin.h"
#include "GazeboUtils.h"

#include "cudaMappedMemory.h"
#include "cudaPlanar.h"
#include <time.h>


#define PI 3.141592653589793238462643383279502884197169f  /**< Value of PI */

// Log  file
#define LOG_FILE    true             /**< Set to true to create a log file */
//#define LOG_FILE_NAME   "statistic_3D_3AX.log"  /**< Log file name */
#define LOG_FILE_NAME   "statistic_2D_2AX.log"  /**< Log file name */

// Joint ranges for the edo segments and rotating base.
#define JOINT_MIN     -0.0873f      /**< Joint Min */   // Was -0.7854f
#define JOINT_MAX     1.5708f       /**< Joint Max */
#define BASE_JOINT_MIN  -0.7854f    /**< Joint 1/Base Min */  // Was -1.5708
#define BASE_JOINT_MAX  0.7854f     /**< Joint 1/Base Max */  // Was 1.5708
#define ROTATE4MAX  0f
//#define BASE_JOINT_MIN  -0.7854f      // FOR 3+ AXIS BASE
//#define BASE_JOINT_MAX  0.7854f       // FOR 3+ AXIS BASE

// Turn on velocity based control
#define VELOCITY_CONTROL true      /**< Set to true for velocity control */
#define VELOCITY_MIN -0.2f          /**< Min velocity */
#define VELOCITY_MAX  0.2f          /**< Max velocity */

// Parameters for the DQN API agent
#define INPUT_CHANNELS 3      /**< RGB Channels */
#define OPTIMIZER "RMSprop"   /**< Optimizer for algorithm */
#define LEARNING_RATE 0.001f//0.0003f  /**< RL learning rate */    // original=0.03
#define REPLAY_MEMORY 10000   /**< Replay memory */       // Was 20000
#define BATCH_SIZE 32        /**< Batch size */          // Was 512
#define GAMMA 0.99f            /**< Gamma */  //was 0.9f

#define MODE 0  /**< 0: Save new model 1: Resume training model 2: Test model */
#if MODE == 0

  #define EPS_START 0.9f  /**< EPS Start */ // Was 0.0f for test 0.9 for train
  #define EPS_END 0.05f   /**< EPS End */   // Was 0.0f for test 0.05 for train
  #define EPS_DECAY 200   /**< EPS Decay */ // Was 100
  #define LOADM false     /**< Set to false for training and true for testing */
  #define TESTMODE false  /**< Set to true for testing/inference */
  #define SAVEMODE true   /**< Set to true for saving */
  #define ALLOW_RANDOM true  /**< Set to true to allow for random actions */ //edo team

#elif MODE == 1

  #define EPS_START 0.9f
  #define EPS_END 0.05f
  #define EPS_DECAY 200
  #define LOADM false
  #define TESTMODE true
  #define SAVEMODE true
  #define ALLOW_RANDOM true

#else //MODE (Load model)

  #define EPS_START 0.0f
  #define EPS_END 0.0f
  #define EPS_DECAY 200
  #define LOADM true
  #define TESTMODE true
  #define SAVEMODE false
  #define ALLOW_RANDOM true

#endif //MODE

//#define EPS_DECAY 200       /**< EPS Decay */           // Was 100

#define USE_LSTM true       /**< Set to true to us Long Short Term Memory */
#define LSTM_SIZE 256       /**< Long Short Term Memory Size */
#define DEBUG_DQN true      /**< Set to true for Debug DQN */

// Define ObjectNames
#define WORLD_NAME "edo_world"      /**< Gazebo world name */
#define PROP_NAME  "tube"           /**< Gazebo prop name */
#define GRIP_NAME  "gripper_middle" /**< Gazebo gripper name */

// Define RewardParameters
#define REWARD_WIN 1.0f          /**< Reward value for a win */  // Was 20
#define REWARD_LOSS -1.0f        /**< Reward value for a loss */ // Was -20
#define REWARD_MULTIPLIER 5.0f   /**< Reward multplier */ // Was 25
#define GAMMA_FALLOFF 0.35f       /**< Gamma falloff */

#define COLLISION_FILTER "ground_plane::link::collision"  /**< Gazebo ground
                                                            plane collision
                                                            name */
#define COLLISION_ITEM   "tube::link::tube_collision"     /**< Gazebo prop
                                                            collision name */
#define COLLISION_POINT "edo_sim::gripper_middle::middle_collision" /**< Gazebo
                                                                      gripper
                                                                      collision
                                                                      name */
// Animation steps
#define ANIMATION_STEPS 1000  /**< Steps for reset animation */ // Was 1000
#define DEBUG false         /**< Set to true for Debug output in stdout */
#define LOCKBASE false    /**< Locks Joint 1/base rotation */

#define EPSILON 0.0000001 /**< Added for float comparisons */

#define PRINT_ANGLES false  /**< Set true to output joint angles continuously */
#define SAVE_ANGLES true    /**< Set true to save joint angles continuously */
                            //^^SET TO TRUE TO PRINT WINS TO "/relay"^^

ros::Rate loop_rate(100);

int i =0;//EDO TEAM
namespace gazebo
{

// register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(EdoPlugin)

/***************************************************************
 **                Function(s) Definition
 ****************************************************************/

/** @brief Calculate the degrees from radians
 *  @param rad - radian angle
 *  @return int - degree conversion of radian angle
 *  @exception None
 */
int radToDeg(float rad){
  return rad * 180 / PI;
} //radToDeg()



/** @brief Construct EdoPlugin object
 *  @param None
 *  @return EdoPlugin object
 *  @exception None
 */
EdoPlugin::EdoPlugin() : ModelPlugin(),
cameraNode(new gazebo::transport::Node()),
collisionNode(new gazebo::transport::Node())
{
  printf("EdoPlugin::EdoPlugin()\n");

  agent            = NULL;
  inputState       = NULL;
  inputBuffer[0]   = NULL;
  inputBuffer[1]   = NULL;
  inputBufferSize  = 0;
  inputRawWidth    = 0;
  inputRawHeight   = 0;
  actionJointDelta = 0.15f;
  actionVelDelta   = 0.1f;    // 0.05f;
  maxEpisodeLength = 100;     // Was 85
  episodeFrames    = 0;

  newState         = false;
  newReward        = false;
  endEpisode       = false;
  totalReward      = 0.0f;
  rewardHistory    = 0.0f;
  testAnimation    = true;
  loopAnimation    = false;
  animationStep    = 10;//EDO TEA
  lastGoalDistance = 0.0f;
  avgGoalDelta     = 0.0f;
  successfulGrabs  = 0;
  totalRuns        = 0;
  runHistoryIdx    = 0;
  runHistoryMax    = 0;
  needsRandom      = true;   // Added by Jack for randomization of object //Change to true by Adel
  runsSinceRandom  = 0;       // Added by Jack for randomization of object
  endSequence      = true;    // Added by Jack for print joint values
  prevJointValues  = "";      // Added by Jack for print joint values
  select           = false;   // Added by Jack to select single or double image
                              // true for double, false for single.
  INPUT_HEIGHT     = 128;//64; //Adel changed from 64 to 128
  INPUT_WIDTH      = 128;//64;      // Changed so selection can be made at runtime //Adel changed from 64 to 128
  real_score       = 0.0f;    // For saving checkpoints
  best_score       = 0.5f;    // Added to save checkpoint at best score

  // Added to create infile stream
  infile.open("select.txt");
  if(!infile.is_open()){
    std::cout << "Failed to open 'select.txt'\nDefaulting to single "
      << "image mode\n";
  }
  else{
    std::string temp;
    infile >> temp;
    if(temp == "1"){
      select = true;
      INPUT_HEIGHT = 128; //changed from 128 to 64 Mahmoud
      INPUT_WIDTH = 128;
    }
    if(TESTMODE){
      infile >> loadfile;
    }
  }
  infile.close();
  if(infile.is_open()){
    std::cout << "Failed to close 'select.txt'\n";
  }
  // Added to create outfile stream
  outfile.open("commands.txt");

  // zero the run history buffer
  memset(runHistory, 0, sizeof(runHistory));

  // set the default reset position for each joint
  for( uint32_t n=0; n < DOF; n++ )
    resetPos[n] = 0.0f;

  resetPos[1] = 0.25;   // make the edo canted forward a little --was 0.25
                        // If this is changed, change reset val in Relay.cpp

  // set the initial positions and velocities to the reset

  for( uint32_t n=0; n < DOF; n++ )
   {
    ref[n] = resetPos[n]; //JOINT_MIN;
    vel[n] = 0.0f;
   }
  

  // set the joint ranges

  for( uint32_t n=0; n < DOF; n++ )
    {
      jointRange[n][0] = JOINT_MIN;
      jointRange[n][1] = JOINT_MAX;
      if(DOF ==3)
        {
         jointRange[3][0] = 0.0f;
         jointRange[3][1] = 0.0f;
        }
    }
  

  // if the base is freely rotating, set it's range separately
  
  if( !LOCKBASE )
    {
     jointRange[0][0] = BASE_JOINT_MIN;
     jointRange[0][1] = BASE_JOINT_MAX;
    }
  

  if (LOG_FILE) //Log on a file the parameters
  {
    FILE *f;
    f = fopen(LOG_FILE_NAME, "w");
    if (f)
    {
      fprintf(f,"----- LOG FILE -----\nFILE_NAME: EdoPlugin.cpp\n"
          "FILE_VERSION: %d\n", FILE_VERSION);
      time_t t = time(NULL);
      struct tm tm = *localtime(&t);
      fprintf(f, "DATE: %d-%d-%d %d:%d:%d\n", tm.tm_year + 1900,
          tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
      fprintf(f, "REWARD_WIN: %f\n", REWARD_WIN);
      fprintf(f, "REWARD_LOSS: %f\n", REWARD_LOSS);
      fprintf(f, "REWARD_MULTIPLIER: %f\n", REWARD_MULTIPLIER);
      fprintf(f, "GAMMA_FALLOFF: %f\n", GAMMA_FALLOFF);
      fprintf(f, "INPUT_CHANNELS: %d\n", INPUT_CHANNELS);
      fprintf(f, "OPTIMIZER: %s\n", OPTIMIZER);
      fprintf(f, "LEARNING_RATE: %f\n", LEARNING_RATE);
      fprintf(f, "REPLAY_MEMORY: %d\n", REPLAY_MEMORY);
      fprintf(f, "BATCH_SIZE: %d\n", BATCH_SIZE);
      fprintf(f, "GAMMA: %f\n", GAMMA);
      fprintf(f, "EPS_START: %f\n", EPS_START);
      fprintf(f, "EPS_END: %f\n", EPS_END);
      fprintf(f, "EPS_DECAY: %d\n", EPS_DECAY);
      fprintf(f, "USE_LSTM: %d\n", USE_LSTM);
      fprintf(f, "LSTM_SIZE: %d\n", LSTM_SIZE);
      fprintf(f, "ALLOW_RANDOM: %d\n", ALLOW_RANDOM);
      fprintf(f, "DEBUG_DQN: %d\n", DEBUG_DQN);
      fprintf(f, "\n\n\n");
      fprintf(f, "WINS,TOTAL,WIN PERCENTAGE,Random, totalReward\n");
      fclose(f);
    }
  }
  int argc= 0;
  char **argv;
  ros::init(argc, argv, "edo_plugin");
  test_pub = nh.advertise<std_msgs::Int8MultiArray>("jnt_state", 100);// changed from relay too usb_jnt
} // EdoPlugin::EdoPlugin()

/** @brief Load EdoPlugin, save the model, initialize the controller,
 *  camera, and collision subscribers
 *  @param _parent - Gazebo provided pointer to the parent model
 *  @param sdf - Gazebo provided pointer to SDF elements
 *  @return void
 *  @exception None
 */
void EdoPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
{
  printf("EdoPlugin::Load('%s')\n", _parent->GetName().c_str());

  // Create DQN agent
  if( !createAgent() )
    return;

  // Store the pointer to the model
  this->model = _parent;
  this->j2_controller = new physics::JointController(model);
  
  // Create our node for camera communication
  cameraNode->Init();
  if(!select){
    // Single camera sub
    cameraSub = cameraNode->Subscribe("/gazebo/" WORLD_NAME
        "/camera/link/camera/image",
        &EdoPlugin::onCameraMsg, this);

    /*cameraSubSecond = cameraNode->Subscribe("/gazebo/" WORLD_NAME
        "/cameraSecond/link/camera/image",
        &EdoPlugin::onCameraMsg, this);*/
    
  }
  else{
    //  Multi camera sub
    cameraSub = cameraNode->Subscribe("/gazebo/" WORLD_NAME
        "/multicamera/link/multicamera/images",
        &EdoPlugin::onCamerasMsg, this);   
  }
  // Create our node for collision detection
  collisionNode->Init();

  collisionPropSub = collisionNode->Subscribe("/gazebo/" WORLD_NAME "/"
      PROP_NAME "/link/my_contact",
      &EdoPlugin::onCollisionMsg, this);

  // Listen to the update event.
  // This event is broadcast every simulation iteration.
  this->updateConnection =
    event::Events::ConnectWorldUpdateBegin(boost::bind(&EdoPlugin::OnUpdate,
          this, _1));
} // EdoPlugin::Load()

/** @brief Create the DQN agent to control the e.DO motions
 *  @param None
 *  @return bool - returns true if agent is created successfully
 *  @exception None
 */
bool EdoPlugin::createAgent()
{
  if( agent != NULL )
    return true;

  // Create DQN agent
  agent = dqnAgent::Create(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS,
      DOF*2,/*+(DOF-1)*(DOF-1),*/ OPTIMIZER, LEARNING_RATE, REPLAY_MEMORY,
      BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, USE_LSTM, LSTM_SIZE,
      ALLOW_RANDOM, DEBUG_DQN, LOADM);


  if(TESTMODE){
    printf("Model loading\n");
    loadAgent(loadfile);
  }
  if( !agent )
  {
    printf("EdoPlugin - failed to create DQN agent\n");
    return false;
  }

  // Allocate the python tensor for passing the camera state
  inputState = Tensor::Alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

  if( !inputState )
  {
    printf("EdoPlugin - failed to allocate %ux%ux%u Tensor\n",
        INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
    return false;
  }

  return true;
} // EdoPlugin::createAgent()

/** @brief Saves the agent with filename "edoCheckpoint" in
 *  jetson-reinforcement/build/<SYS_ARCH>/bin/ directory.
 *  Calls existing DQN function in python
 *  @param None
 *  @return bool - returns true if agent is saved successfully
 *  @exception None
 */
bool EdoPlugin::saveAgent(std::string filename)
{
  return agent->SaveCheckpoint(filename.c_str());
} // EdoPlugin::saveAgent()

/** @brief Loads the agent with filename "edoCheckpoint" in
 *  jetson-reinforcement/build/<SYS_ARCH>/bin/ directory.
 *  Calls existing DQN function in python
 *  @param None
 *  @return bool - returns true if agent is loaded successfully
 *  @exception None
 */
bool EdoPlugin::loadAgent(std::string filename)
{
  return agent->LoadCheckpoint(filename.c_str());
} // EdoPlugin::loadAgent()

/** @brief Callback function called by camera subscriber when new image is
 *  available. This version supports single image.
 *  @param _msg - Gazebo provided ConstImageStampedPtr containing Image
 *  @return void
 *  @exception None
 */
void EdoPlugin::onCameraMsg(ConstImageStampedPtr &_msg)
{
  // don't process the image if the agent hasn't been created yet
  if( !agent )
    return;

  // check the validity of the message contents
  if( !_msg )
  {
    printf("EdoPlugin - recieved NULL message\n");
    return;
  }

  // retrieve image dimensions
  const int width  = _msg->image().width();
  const int height = _msg->image().height();
  const int bpp    = (_msg->image().step() / _msg->image().width()) * 8;
  //  ^^^ bpp: bits per pixel ^^^
  const int size   = _msg->image().data().size();

  if( bpp != 24 )
  {
    printf("EdoPlugin - expected 24BPP uchar3 image from camera, got %i\n",
        bpp);
    return;
  }

  // allocate temp image if necessary
  if( !inputBuffer[0] || size != inputBufferSize )
  {
    if( !cudaAllocMapped(&inputBuffer[0], &inputBuffer[1], size) )
    {
      printf("edoPlugin - cudaAllocMapped() failed to allocate %i bytes\n",
          size);
      return;
    }

    printf("edoPlugin - allocated camera img buffer %ix%i  %i bpp  %i bytes\n",
        width, height, bpp, size);

    inputBufferSize = size;
    inputRawWidth   = width;
    inputRawHeight  = height;
  }

  memcpy(inputBuffer[0], _msg->image().data().c_str(), inputBufferSize);
  newState = true;

  if(DEBUG){
    printf("camera %i x %i  %i bpp  %i bytes\n", width, height, bpp, size);
  }
} // EdoPlugin::onCameraMsg()


/** @brief Callback function called by camera subscriber when new images are
 *  available. This version supports double image.
 *  @param _msg - Gazebo provided ConstImagesStampedPtr containing Images
 *  @return void
 *  @exception None
 */
void EdoPlugin::onCamerasMsg(ConstImagesStampedPtr &_msg)
{
  // don't process the image if the agent hasn't been created yet
  if( !agent )
    return;

  // check the validity of the message contents
  if( !_msg )
  {
    printf("EdoPlugin - recieved NULL message\n");
    return;
  }


  //Combining two images top and side view
  gazebo::msgs::Image combined_image;
  combined_image.set_width(128); //Adel changed from 64 to 128
  combined_image.set_height(128);//changed 128 to 64
  combined_image.set_pixel_format(_msg->image(0).pixel_format());
  combined_image.set_step(_msg->image(0).step());
  std::string new_data;
  new_data = _msg->image(0).data();
  new_data.resize(128*128*3);//changed 128*64*3 to 64*64*64 //Adel changed from 64 to 128 the second 128
  new_data = new_data + _msg->image(1).data();
  combined_image.set_data(new_data);

  // retrieve image dimensions
  const int width  = combined_image.width();
  const int height = combined_image.height();
  const int bpp    = (combined_image.step() / combined_image.width()) * 8;
  //  ^^^ bpp: bits per pixel ^^^
  const int size   = combined_image.data().size();


  if( bpp != 24 )
  {
    printf("EdoPlugin - expected 24BPP uchar3 image from camera, got %i\n",
        bpp);
    return;
  }

  // allocate temp image if necessary
  if( !inputBuffer[0] || size != inputBufferSize )
  {
    if( !cudaAllocMapped(&inputBuffer[0], &inputBuffer[1], size) )
    {
      printf("edoPlugin - cudaAllocMapped() failed to allocate %i bytes\n",
          size);
      return;
    }

    printf("edoPlugin - allocated camera img buffer %ix%i  %i bpp  %i bytes\n",
        width, height, bpp, size);

    inputBufferSize = size;
    inputRawWidth   = width;
    inputRawHeight  = height;
  }

  memcpy(inputBuffer[0], combined_image.data().c_str(), inputBufferSize);
  newState = true;

  if(DEBUG){
    printf("camera %i x %i  %i bpp  %i bytes\n", width, height, bpp, size);
  }
} // EdoPlugin::onCamerasMsg()

/** @brief Callback function called by collision subscriber when there is a
 *  collision with the prop.
 *  @param contacts - Gazebo provided ConstContactsPtr containing
 *  contact model names
 *  @return void
 *  @exception None
 */
void EdoPlugin::onCollisionMsg(ConstContactsPtr &contacts)
{

  if( testAnimation )
    return;

  for (unsigned int i = 0; i < contacts->contact_size(); ++i)
  {

    //  Filters out collisions where the ground is collision2.
    //  Continue skips the remainderof that for loop.
    if(strcmp(contacts->contact(i).collision2().c_str(), COLLISION_FILTER) == 0)
      continue;

    if(DEBUG){
      std::cout << "Collision between[" << contacts->contact(i).collision1()
        << "] and [" << contacts->contact(i).collision2() << "]\n";
    }

    if (strcmp(contacts->contact(i).collision2().c_str(), COLLISION_POINT) == 0)
    {

      if(DEBUG)
        printf("Give max reward and execute gripper\n");

      // REWARD_WIN was issued if the collision point is gripper_middle.

      rewardHistory = REWARD_WIN * REWARD_MULTIPLIER +(1.0f-(float(episodeFrames)/float(maxEpisodeLength)))*REWARD_WIN*100.0f;
      //rewardHistory = (1.0f - (float(episodeFrames) / float(maxEpisodeLength))) * REWARD_WIN;
      newReward  = true;
      endEpisode = true;
      endSequence = true;
      return;                 // multiple collisions in the for loop
      // above could mess with win count

    }
    else{
      rewardHistory = REWARD_LOSS;
      newReward  = true;
      endEpisode = true;
      endSequence = true;
    }

  }
} // EdoPlugin::onCollisionMsg()

/** @brief Called by updateJoints(). Upon receiving new frame, update the agent.
 *  @param None
 *  @return bool - returns true if agent was successfully updated
 *  @exception None
 */
bool EdoPlugin::updateAgent()
{

  // convert uchar3 input from camera to planar BGR
  if( CUDA_FAILED(cudaPackedToPlanarBGR((uchar3*)inputBuffer[1], inputRawWidth,
          inputRawHeight, inputState->gpuPtr,
          INPUT_WIDTH, INPUT_HEIGHT)) )
  {
    printf("edoPlugin - failed to convert %zux%zu image to %ux%u planar "
        "BGR image\n", inputRawWidth, inputRawHeight,
        INPUT_WIDTH, INPUT_HEIGHT);

    return false;
  }

  // select the next action
  int action = 0;

  if( !agent->NextAction(inputState, &action) )
  {
    printf("edoPlugin - failed to generate agent's next action\n");
    return false;
  }

  // make sure the selected action is in-bounds
  if( action < 0 || action >= DOF * 2 )
  {
    printf("edoPlugin - agent selected invalid action, %i\n", action);
    return false;
  }

  if(DEBUG){printf("edoPlugin - agent selected action %i\n", action);}

  // action 0 does nothing, the others index a joint
  /*if( action == 0 )
    return false; // not an error, but didn't cause an update

    action--;*/ // with action 0 = no-op, index 1 should map to joint 0


#if VELOCITY_CONTROL
  // if the action is even, increase the joint velocity by the delta parameter
  // if the action is odd,  decrease the joint velocity by the delta parameter
  float velocity = vel[action/2] + actionVelDelta *
    ((action % 2 == 0) ? 1.0f : -1.0f);

  if( velocity < VELOCITY_MIN )
    velocity = VELOCITY_MIN;

  if( velocity > VELOCITY_MAX )
    velocity = VELOCITY_MAX;

  vel[action/2] = velocity;

  for( uint32_t n=0; n < DOF; n++ )
  {
    ref[n] += vel[n];

    if( ref[n] < jointRange[n][0] )
    {
      ref[n] = jointRange[n][0];
      vel[n] = 0.0f;
    }
    else if( ref[n] > jointRange[n][1] )
    {
      ref[n] = jointRange[n][1];
      vel[n] = 0.0f;
    }
  }
#else
  // index the joint, considering each DoF has 2 actions (+ and -)
  const int jointIdx = action / 2;

  /*Added by Ashwini: To change eDO base(joint 1)
    movements differently than other joints*/
  float joint;
  if(action>1){
    joint= ref[jointIdx] + actionJointDelta *
      ((action % 2 == 0) ? 1.0f : -1.0f);
  }
  else{
    joint = ref[jointIdx] + actionJointDelta *
      ((action % 2 == 0) ? 0.5f : -0.5f);
  }

  // if(action< DOF*2){
  // limit the joint to the specified range
  if( joint < jointRange[jointIdx][0] )
    joint = jointRange[jointIdx][0];

  if( joint > jointRange[jointIdx][3] ) //kiki
    joint = jointRange[jointIdx][3];
    



  //std::cout<<" Update Joint: "<<jointIdx <<" Value: "<<joint;
  ref[jointIdx] = joint;


  /*}
    else{
  // To move joint 2 and joint 3 simultaneously using 4 extra actions
  int tempAction = action - DOF*2;

  float joint1 , joint2;

  if (tempAction == 0)
  {
  joint1= ref[1] + actionJointDelta * 1.0;
  joint2 = ref[2] + actionJointDelta * 1.0;
  }
  if (tempAction == 1)
  {
  joint1 = ref[1] + actionJointDelta * 1.0;
  joint2 = ref[2] + actionJointDelta * -1.0;
  }
  if (tempAction == 2)
  {
  joint1 = ref[1] + actionJointDelta * -1.0;
  joint2 = ref[2] + actionJointDelta * 1.0;
  }
  if (tempAction == 3)
  {
  joint1 = ref[1] + actionJointDelta * -1.0;
  joint2 = ref[2] + actionJointDelta * -1.0;
  }

  // limit the joint to the specified range
  if( joint1 < jointRange[1][0] )
  joint1= jointRange[1][0];

  if( joint1 > jointRange[1][1] )
  joint1 = jointRange[1][1];

  // limit the joint to the specified range
  if( joint2 < jointRange[2][0] )
  joint2 = jointRange[2][0];

  if( joint2 > jointRange[2][1] )
  joint2 = jointRange[2][1];

  ref[1] = joint1;
  ref[2] = joint2;
  }*/

#endif

  return true;
} // EdoPlugin::updateAgent()

/** @brief Update joint reference positions
 *  @param None
 *  @return bool - returns true if positions have been modified
 *  @exception None
 */
bool EdoPlugin::updateJoints()
{
  //printf("EdoPlugin updateJoints");
  if( testAnimation ) // test sequence
  {
    const float step = (JOINT_MAX - JOINT_MIN) *
      (float(1.0f) / float(ANIMATION_STEPS));

    // return to base position
    for( uint32_t n=0; n < DOF; n++ )
    {
      if( ref[n] < resetPos[n])
        ref[n] += step;
      else if( ref[n] > resetPos[n])
        ref[n] -= step;

      if( ref[n] < jointRange[n][0])
        ref[n] = jointRange[n][0];
      else if( ref[n] > jointRange[n][1])
        ref[n] = jointRange[n][1];
    }

    animationStep++;

    // reset and loop the animation
    if( animationStep > ANIMATION_STEPS )
    {
      animationStep = 0;

      if( !loopAnimation )
        testAnimation = false;
    }
    else if( animationStep == ANIMATION_STEPS / 2 )
    {
      //if(runsSinceRandom % 10 == 0){
        //RandomizeProps();
      //}
      ResetPropDynamics();
    }

    return true;
  }

  else if( newState && agent != NULL )
  {
    // update the AI agent when new camera frame is ready
    episodeFrames++;

    if(DEBUG){printf("episode frame = %i\n", episodeFrames);}

    // reset camera ready flag
    newState = false;

    if( updateAgent() )
      return true;
  }

  return false;
} // EdoPlugin::updateJoints()


/** @brief Get the servo center for a particular degree of freedom
 *  @param dof - Degree of freedom to get reset value for
 *  @return float - reset value for dof
 */
float EdoPlugin::resetPosition( uint32_t dof )
{
  return resetPos[dof];
} // EdoPlugin::resetPosition()


/** @brief called by the world update start event
 *  @param updateInfo - Gazebo provided callback message. Called every
 *  iteration.
 *  @return void
 */
void EdoPlugin::OnUpdate(const common::UpdateInfo& updateInfo)
{
  // determine if we have new camera state and need to update the agent
  const bool hadNewState = newState && !testAnimation;

  // update the robot positions with vision/DQN
  if( updateJoints() )
  {
    /*  Originally Commented Out
        printf("%f  %f  %f  %s\n", ref[0], ref[1], ref[2],
        testAnimation ? "(testAnimation)" : "(agent)");
        */
    double angle(1);  // Unused. Unknown origin.

#if LOCKBASE
    
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_2"),       0);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_2"),  ref[1]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_3"),  ref[2]);    
    j2_controller->SetJointPosition(this->model->GetJoint("joint_3"),  ref[3]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_5"),  ref[4]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_5"),  ref[5]);
#else
    // Change to 2,0,1,3,4,5 to do 2D, 2 joint training
    // Change to 0,1,2,3,4,5 to do 3D, 3 joint training
    
    
    j2_controller->SetJointPosition(this->model->GetJoint("joint_1"),  ref[0]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_2"),  ref[1]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_3"),  ref[2]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_4"),  ref[3]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_5"),  ref[4]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_6"),  ref[5]);
    
    
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_1"),  ref[0]);
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_2"),  ref[1]);
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_3"),  ref[2]);
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_4"),  -1.00);
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_5"),  ref[4]);
    //j2_controller->SetJointPosition(this->model->GetJoint("joint_6"),  ref[5]);
    
   /* for (int i = 0; i < DOF; i++){
		std::cout << "ref[" << i << "] = " << ref[i] << std::endl;
	}*/
	
    
    /*j2_controller->SetJointPosition(this->model->GetJoint("joint_2"),  ref[0]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_2"),  ref[1]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_3"),  ref[2]);    
    j2_controller->SetJointPosition(this->model->GetJoint("joint_3"),  ref[3]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_5"),  ref[4]);
    j2_controller->SetJointPosition(this->model->GetJoint("joint_5"),  ref[5]);*/
    
#endif
    // added for outputing joint angles either to file only or stdout and file
    if(PRINT_ANGLES || SAVE_ANGLES){
      if(endSequence && totalRuns == 0){
        if(PRINT_ANGLES){
          std::cout << "------------------\nBEGIN SEQUENCE "
            << totalRuns << "\n------------------\n";
        }
        outfile << "------------------\nBEGIN SEQUENCE "
          << totalRuns << "\n------------------\n";
        endSequence = false;
      }
      std::string newJointValues;
      //for(int i = 0; i < 6; ++i){
        //newJointValues += std::to_string(radToDeg(ref[i])) +  " ";
        // Change to 2,0,1,3,4,5 to do 2D, 2 joint training
        // Change to 0,1,2,3,4,5 to do 3D, 3 joint training
        if(!testAnimation){
          winSequenceVec.push_back(radToDeg(ref[0]));
          winSequenceVec.push_back(radToDeg(ref[1]));
          winSequenceVec.push_back(radToDeg(ref[2]));
          winSequenceVec.push_back(radToDeg(ref[3]));
          winSequenceVec.push_back(radToDeg(ref[4]));
          winSequenceVec.push_back(radToDeg(ref[5]));
          
          winSequenceVec.clear();
          for(int i = 0; i < 6; i++){
              endJoints[i] = newEndJoints[i];
              newEndJoints[i] = radToDeg(ref[i]);
          }
          
          //endJoints[2] = endJoints[3];
          //endJoints[3] = 0;

          //endJoints[4] = endJoints[5];
          //endJoints[5] = 0;

          for(int i = 0; i < 6; i++){
            winSequenceVec.push_back(endJoints[i]);

          }

        }
      //}
      if(newJointValues != prevJointValues){
        if(PRINT_ANGLES){
          std::cout << newJointValues << "\n";
        }
        outfile << newJointValues << "\n";
        outfile.flush();
        prevJointValues = newJointValues;
      }
      if(endSequence && !testAnimation){
        if(PRINT_ANGLES){
          std::cout << "\n------------------\n" << rewardHistory
            << " END SEQUENCE "<< totalRuns
            << "\n------------------\n\n";
          std::cout << "------------------\nBEGIN SEQUENCE "
            << totalRuns + 1 << "\n------------------\n";
        }
        outfile << "\n------------------\n" << rewardHistory
          << " END SEQUENCE " << totalRuns << "\n------------------\n\n";
        outfile << "------------------\nBEGIN SEQUENCE "
          << totalRuns + 1 << "\n------------------\n";
        endSequence = false;
        if(rewardHistory > 0){
          if(winState){
            outfile.open("results.txt", std::ios::trunc);
            for(int i = 0; i < 6; i++){
               std::cout << "\n" << "Joint " << i + 1 << " Movement Angle: " << endJoints[i];
               outfile << endJoints[i] << std::endl;            
            }
            std::cout << "\n";
            outfile.close();
          




      
            winState = false;
            winSequence.data.clear();
            winSequence.data.insert(winSequence.data.begin(),
                                  winSequenceVec.begin(), winSequenceVec.end());
            test_pub.publish(winSequence);
            ros::spinOnce();
            loop_rate.sleep();
          }
       }
       winSequenceVec.clear();
     }
   }
 }
  // episode timeout
  if( maxEpisodeLength > 0 && episodeFrames > maxEpisodeLength )
  {
    printf("EdoPlugin - triggering EOE, episode has exceeded %i frames\n",
        maxEpisodeLength);
    rewardHistory = REWARD_LOSS * REWARD_MULTIPLIER;  //0;
    newReward     = true;
    endEpisode    = true;
    endSequence   = true;
  }

  // if an EOE reward hasn't already been issued, compute an intermediary reward
  if( hadNewState && !newReward )
  {
    // retrieve the goal prop model object
    PropPlugin* prop = GetPropByName(PROP_NAME);

    if( !prop )
    {
      printf("EdoPlugin - failed to find Prop '%s'\n", PROP_NAME);
      return;
    }

    //Remember where the user moved the prop to for when it's reset
    prop->UpdateResetPose();


    //TODO Compute distance between cylinder centre and gripper-middle

    // Get the bounding box for the prop object
    const math::Box& propBBox = prop->model->GetBoundingBox();
    const math::Vector3& propCenter = propBBox.GetCenter(); // Cory Simms: Added to get the point at the center of the prop

    physics::LinkPtr gripper  = model->GetLink(GRIP_NAME);
    const math::Box& gripBBox = gripper->GetBoundingBox();



    if( !gripper )
    {
      printf("EdoPlugin - failed to find Gripper '%s'\n", GRIP_NAME);
      return;
    }



    //If the robot impacts the ground, count it as a loss

    const float groundContact = 0.00f;

    if( gripBBox.min.z <= groundContact || gripBBox.max.z <= groundContact/* ||
    *** FOR DETECTING BASE CONTACT ***
    ((gripBBox.min.x <= 0.2613 || gripBBox.max.x <= 0.2613) &&
    (gripBBox.min.y <= 0.2613 || gripBBox.max.y <= 0.2613) &&
    (gripBBox.min.z <= 0.135 || gripBBox.max.z <= 0.135))*/)
    {
      //for( uint32_t n=0; n < 10; n++ )
      printf("GROUND CONTACT, EOE\n");

      rewardHistory = REWARD_LOSS;
      newReward     = true;
      endEpisode    = true;
      endSequence   = true;
    }
    else
    {


      //Compute reward from distance to the goal
      //const float distGoal = BoxDistance(gripBBox, propBBox);
      // Cory Simms: added to get a more accurate distance was ^^^^
     const float distGoal = PointBoxDistance(propCenter, gripBBox); //- .0225f;


      //Issue an interim reward based on the delta of the distance to the object
      if( episodeFrames > 1 )
      {

        const float distDelta  = lastGoalDistance - distGoal;

        const float alpha = 0.0f;

        //Compute the smoothed moving average of the
        //delta of the distance to the goal
        avgGoalDelta  =(avgGoalDelta * alpha) + (distDelta * (1.0f - alpha));
       if(DEBUG){
        printf("distDelta: %f avgGoalDelta %f \n", distDelta, avgGoalDelta);
      }

        float distpenalty= (1.0f-exp(distGoal));

        if(avgGoalDelta > 0.01f){
          rewardHistory = (REWARD_WIN + distpenalty*0.1f)* 0.1f;
          //printf("If Distgoal: %0.3f, avgGoalDelta: %0.3f, rewardHistory: %0.3f \n",distGoal, avgGoalDelta, rewardHistory);
        }
       else{
         rewardHistory = REWARD_LOSS - distGoal* 2.0f;
         //printf("Else Distgoal: %0.3f, avgGoalDelta: %0.3f, rewardHistory: %0.3f \n",distGoal, avgGoalDelta, rewardHistory);
       }
       // we want to be moving
       /*if (std::abs(avgGoalDelta) < 0.01f){
         rewardHistory -= 1.0f;
       }*/
          newReward     = true;
        //printf("distGoal=%0.3f, distpenalty=%0.3f, rewardHistory=%0.3f\n",distGoal, distpenalty,rewardHistory);
      }

      lastGoalDistance = distGoal;
    }
  }

  // issue rewards and train DQN
  if( newReward && agent != NULL )
  {
    if(DEBUG){
      printf("EdoPlugin - issuing reward %f, EOE=%s  %s\n", rewardHistory,
          endEpisode ? "true" : "false",
          (rewardHistory > 0.1f) ? "POS+" :
          (rewardHistory > 0.0f) ? "POS" :
          (rewardHistory < 0.0f) ? "    NEG" : "       ZERO");
    }
    totalReward += rewardHistory;
    agent->NextReward(rewardHistory, endEpisode);

    // reset reward indicator
    newReward = false;

    // reset for next episode
    if( endEpisode )
    {
      testAnimation    = true;  // reset the robot to base position
      loopAnimation    = false;
      endEpisode       = false;
      episodeFrames    = 0;
      lastGoalDistance = 0.0f;
      avgGoalDelta     = 0.0f;

      // track the number of wins and agent accuracy
      if( rewardHistory >= REWARD_WIN )
      {
        runHistory[runHistoryIdx] = true;
        successfulGrabs++;
        winState = true;
      }
      else
        runHistory[runHistoryIdx] = false;

      const uint32_t RUN_HISTORY = sizeof(runHistory);
      runHistoryIdx = (runHistoryIdx + 1) % RUN_HISTORY;
      totalRuns++;

      runsSinceRandom++;      // Added by to prevent
      // 2 randomizations in a row

      printf("%s  wins = %03u of %03u (%0.2f)  ",
          (rewardHistory >= REWARD_WIN) ? "WIN " : "LOSS",
          successfulGrabs, totalRuns,
          float(successfulGrabs)/float(totalRuns));

      if( totalRuns >= RUN_HISTORY )
      {
        uint32_t historyWins = 0;

        for( uint32_t n=0; n < RUN_HISTORY; n++ )
        {
          if( runHistory[n] )
            historyWins++;
        }

        if( historyWins > runHistoryMax )
          runHistoryMax = historyWins;

        printf("%02u of last %u  (%0.2f)  (max=%0.2f)",
            historyWins, RUN_HISTORY,
            float(historyWins)/float(RUN_HISTORY),
            float(runHistoryMax)/float(RUN_HISTORY));
//////////////////////////////////////////////////////////////////////changed 500 to %500
        if(runsSinceRandom % 500 == 0 ||
            (float(historyWins)/float(RUN_HISTORY) == 1.0 &&
            runsSinceRandom >= 20)){
          PropPlugin* prop = GetPropByName(PROP_NAME);
          prop->Randomize();
          prop->UpdateResetPose();
          runsSinceRandom = 0;
        }
      }

      if (LOG_FILE) // Log on a file at the end of every episode
      {
        FILE *f;
        f = fopen(LOG_FILE_NAME, "a");
        if (f)
        {
          fprintf(f, "%d, %d, %g, %d, %f\n", successfulGrabs, totalRuns,
              float(successfulGrabs)/float(totalRuns),
              (runsSinceRandom == 0 && totalRuns != 0) ? 1 : 0 ,  totalReward);
            fclose(f);
        }
      }
       totalReward =0.0f;

      printf("\n");
      real_score = float(successfulGrabs)/float(totalRuns);

      if(SAVEMODE && totalRuns % 2000 == 0 && totalRuns > 0){
        printf("Save checkpoint...\n");
        std::string filename("VedoCheckpoint" + std::to_string(totalRuns));
        saveAgent(filename);
      }
/*
     Originally commented out
     printf("Current Accuracy:  %0.4f (%03u of %03u)  (reward=%+0.2f %s)\n",
     float(successfulGrabs)/float(totalRuns), successfulGrabs,
     totalRuns, rewardHistory,
     (rewardHistory >= REWARD_WIN ? "WIN" : "LOSS"));

*/
      for( uint32_t n=0; n < DOF; n++ )
        vel[n] = 0.0f;
      }
    }
  } // EdoPlugin::OnUpdate()
} // namespace gazebo
