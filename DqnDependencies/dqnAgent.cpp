/*
 * deepRL
 */

#include "dqnAgent.h"


// constructor
dqnAgent::dqnAgent()
{

}


// destructor
dqnAgent::~dqnAgent()
{

}

// Modified by Ashwini for model loading functionality
// Create
dqnAgent* dqnAgent::Create(uint32_t width, uint32_t height, uint32_t channels, uint32_t numActions,
					  const char* optimizer, float learning_rate, uint32_t replay_mem, uint32_t batch_size,
					  float gamma, float epsilon_start,  float epsilon_end,  float epsilon_decay,
					  bool use_lstm, int lstm_size, bool allow_random, bool debug_mode, bool load_mode)
{
	if( width == 0 || height == 0 || channels == 0 || numActions == 0 )
		return NULL;

	dqnAgent* agent = new dqnAgent();

	if( !agent )
		return NULL;
		
// Modified by Ashwini for model loading functionality
	if( !agent->Init(width, height, channels, numActions, "DQN",
				  DEFAULT_NEXT_ACTION, DEFAULT_NEXT_REWARD, DEFAULT_LOAD_MODEL, DEFAULT_SAVE_MODEL,
				  optimizer, learning_rate, replay_mem, batch_size, gamma,
				  epsilon_start, epsilon_end, epsilon_decay,
				  use_lstm, lstm_size, allow_random, debug_mode, load_mode))
	{
		return NULL;
	}

	return agent;
}



// NextAction
bool dqnAgent::NextAction( Tensor* state, int* action )
{
	if( !state || !action )
		return false;

	return rlAgent::NextAction(state, action);
}


// NextReward
bool dqnAgent::NextReward( float reward, bool end_episode )
{
	return rlAgent::NextReward(reward, end_episode);
}

//  Jack Shelata Added: Trying to add saving functionality
bool dqnAgent::SaveCheckpoint( const char *filename )
{
  return rlAgent::SaveCheckpoint(filename );
}

//  Jack Shelata Added: Trying to add saving functionality
bool dqnAgent::LoadCheckpoint( const char *filename )
{
  return rlAgent::LoadCheckpoint(filename );
}
