// belongs in directory /home/nvidia/jetson-reinforcement/gazebo
// must run makefile in /home/nvidia/jetson-reinforcement/build after each update

/** @file PropPlugin.cpp
 *  @brief Plugin for the prop in Gazebo e.DO simulation for reinforcement
 *  learning. Resources: http://github.com/dusty-nv/jetson-reinforcement
 *  @author Ashwini Magar, Jack Shelata, Alessandro Piscioneri
 *  @date June 1, 2018
 */

#include "PropPlugin.h"
#include <iostream>
#include <fstream>
#include <stdlib.h> 

using namespace std;

namespace gazebo
{

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(PropPlugin)

//---------------------------------------------------------------------------------------
std::vector<PropPlugin*> props;

size_t GetNumProps()
{
	return props.size();
}

PropPlugin* GetProp( size_t index )
{
	return props[index];
}

PropPlugin* GetPropByName( const char* name )
{
	if( !name )
		return NULL;

	const size_t numProps = props.size();

	for( size_t n=0; n < numProps; n++ )
	{
		if( strcmp(props[n]->model->GetName().c_str(), name) == 0 )
			return props[n];
	}

	return NULL;
}

void ResetPropDynamics()
{
	const size_t numProps = props.size();

	for( size_t n=0; n < numProps; n++ )
		props[n]->ResetDynamics();
}

void RandomizeProps()
{
	const size_t numProps = props.size();

	for( size_t n=0; n < numProps; n++ )
		props[n]->Randomize();
}

//---------------------------------------------------------------------------------------


// Plugin init
void PropPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
{
	printf("PropPlugin::Load('%s')\n", _parent->GetName().c_str());

	// Store the pointer to the model
	this->model = _parent;

	// Store the original pose of the model
	UpdateResetPose();

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&PropPlugin::OnUpdate, this, _1));

	// Track this object in the global Prop registry
	props.push_back(this);
}


// UpdateResetPose
void PropPlugin::UpdateResetPose()
{
	this->originalPose = model->GetWorldPose();
}


// Called by the world update start event
void PropPlugin::OnUpdate(const common::UpdateInfo & /*_info*/)
{
	// Apply a small linear velocity to the model.
	//this->model->SetLinearVel(math::Vector3(.03, 0, 0));

   /*const math::Pose& pose = model->GetWorldPose();
	
	printf("%s location:  %lf %lf %lf\n", model->GetName().c_str(), pose.pos.x, pose.pos.y, pose.pos.z);
	
	const math::Box& bbox = model->GetBoundingBox();

	printf("%s bounding:  min=%lf %lf %lf  max=%lf %lf %lf\n", model->GetName().c_str(), bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
   */
	/*const math::Box& bbox = model->GetBoundingBox();

  	const math::Vector3 center = bbox.GetCenter();
	const math::Vector3 bbSize = bbox.GetSize();

	printf("%s bounding:  center=%lf %lf %lf  size=%lf %lf %lf\n", model->GetName().c_str(), center.x, center.y, center.z, bbSize.x, bbSize.y, bbSize.z); 
	*/
}


// Reset the model's dynamics and pose to original

void PropPlugin::ResetDynamics()
{
	model->SetAngularAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetAngularVel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearVel(math::Vector3(0.0, 0.0, 0.0));

  //Read from txt file //Adel and Tyler
  //originalPose = math::Pose(0.63, 0, 0.027500, 0, 0, 0);
  fstream file;
  file.open("/home/nvidia/jetson-reinforcement/build/aarch64/bin/newX.txt");
  char charcter;
  string x;
 
  while(!file.eof())
  {
    file>>charcter;
    if(file.eof())
    {
      break;
    }
    x+= charcter;
  }
 
  double newX = atof(x.c_str());
  newX = newX/100;
  file.close();


 //added by Nate for xy import 
  fstream xyFile;
  xyFile.open("/home/nvidia/jetson-reinforcement/build/aarch64/bin/xy.txt");
  char xyChar;
  string y;
  
  while (!xyFile.eof()){
	  xyFile >> xyChar;
	  
	  if (xyFile.eof()){
			break;
		}
	  y += xyChar;
	  
  }
  double xy = atof(y.c_str());
  xy /= 100;
  
  xyFile.close();
  
 

  //Added by Hawraa to add depth input to Prop. 
  fstream Zfile;
  Zfile.open("/home/nvidia/jetson-reinforcement/build/aarch64/bin/newZ.txt");
  char charc;
  string z;

  
  while(!Zfile.eof())
  {
    Zfile>>charc;
    if(Zfile.eof())
    {
      break;
    }
    z+= charc;
  }
 
  double newZ = atof(z.c_str());
  newZ = newZ/100;
  Zfile.close();
  originalPose = math::Pose(newX, xy, newZ, 0, 0, 0);
  //std::cout << originalPose << std::endl;
  //model->SetWorldPose(math::Pose(newX, 0.0, 0.0250));//math::Pose
  
  model->SetWorldPose(originalPose);
}


inline float randf( float rand_min, float rand_max )
{
	const float r = float(rand()) / float(RAND_MAX);
	return (r * (rand_max - rand_min)) + rand_min;
}


// Randomize
void PropPlugin::Randomize()
{
	model->SetAngularAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetAngularVel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearVel(math::Vector3(0.0, 0.0, 0.0));

	math::Pose pose = originalPose;
  
  //  Jack Shelata Comment: Change to adjust random range (randf(minf, maxf))
  //  Set y to 0.0 always for simple 2 DOF model
	pose.pos.x = randf(0.27f, 0.66f);     // Approx range of e.DO in x dir 90deg
	pose.pos.y = 0.0f;//randf(-0.28f, 0.28f);//*0.0f;  // Width of paper
	pose.pos.z = 0.0f;
  //  Jack Shelata Added: Adjusts random position until prop is within the arm's reach
  while((pose.pos.x * pose.pos.x) + (pose.pos.y * pose.pos.y) > 0.49 || 
        (pose.pos.x * pose.pos.x) + (pose.pos.y * pose.pos.y) < 0.25){
    pose.pos.x = randf(0.27f, 0.66f);
	  pose.pos.y = 0.0f;//randf(-0.28f, 0.28f);

  }
	//  Jack Shelata Added: To check random positon
	printf("\nProp Random Position:  x: %f  y: %f", pose.pos.x, pose.pos.y);
	model->SetWorldPose(pose);
}

}

