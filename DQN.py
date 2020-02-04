# -*- coding: utf-8 -*-
## @package DQN
#  This module define DQN , DRQN and dueling DQN algorithms along with training routine
#
import argparse
import math
import logging
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

if use_cuda:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')


# parse command line
parser = argparse.ArgumentParser(description='PyTorch DQN runtime')
parser.add_argument('--width', type=int, default=128, metavar='N', help='width of virtual screen')#//Adel changed from 64 to 128
parser.add_argument('--height', type=int, default=128, metavar='N', help='height of virtual screen')#//Adel changed from 64 to 128
parser.add_argument('--channels', type=int, default=3, metavar='N', help='channels in the input image')
parser.add_argument('--actions', type=int, default=6, metavar='N', help='number of output actions from the neural network')#Adel Mohamed default from 3 to 10
parser.add_argument('--optimizer', default='RMSprop', help='Optimizer of choice')
parser.add_argument('--learning_rate', type=float, default=0.001, metavar='N', help='optimizer learning rate')
parser.add_argument('--replay_mem', type=int, default=10000, metavar='N', help='replay memory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')#//Adel changed from 64 to 128
parser.add_argument('--gamma', type=float, default=0.9, metavar='N', help='discount factor for present rewards vs. future rewards')
parser.add_argument('--epsilon_start', type=float, default=0.9, metavar='N', help='epsilon_start of random actions')
parser.add_argument('--epsilon_end', type=float, default=0.05, metavar='N', help='epsilon_end of random actions')
parser.add_argument('--epsilon_decay', type=float, default=200, metavar='N', help='exponential decay of random actions')
parser.add_argument('--allow_random', type=int, default=1, metavar='N', help='Allow DQN to select random actions')
parser.add_argument('--debug_mode', type=int, default=0, metavar='N', help='debug mode')
parser.add_argument('--use_lstm', type=int, default=1, metavar='N', help='use LSTM layers in network')
parser.add_argument('--lstm_size', type=int, default=256, metavar='N', help='number of inputs to LSTM')
parser.add_argument('--load_mode', type=int, default=0, metavar='N', help='load mode') #To enable resume-traning
args = parser.parse_args()

#converting arguments to python variables
input_width    = args.width
input_height   = args.height
input_channels = args.channels
num_actions    = args.actions
optimizer 	= args.optimizer
learning_rate 	= args.learning_rate
replay_mem 	= args.replay_mem
batch_size 	= args.batch_size
gamma 		= args.gamma
epsilon_start 	= args.epsilon_start
epsilon_end 	= args.epsilon_end
epsilon_decay 	= args.epsilon_decay
allow_random 	= args.allow_random
debug_mode		= args.debug_mode
use_lstm		= args.use_lstm
lstm_size      = args.lstm_size
load_mode   	= args.load_mode #To enable resume-traning

print('[deepRL]  use_cuda:       ' + str(use_cuda))
print('[deepRL]  use_lstm:       ' + str(use_lstm))
print('[deepRL]  lstm_size:      ' + str(lstm_size))
print('[deepRL]  input_width:    ' + str(input_width))
print('[deepRL]  input_height:   ' + str(input_height))
print('[deepRL]  input_channels: ' + str(input_channels))
print('[deepRL]  num_actions:    ' + str(num_actions))
print('[deepRL]  optimizer:      ' + str(optimizer))
print('[deepRL]  learning rate:  ' + str(learning_rate))
print('[deepRL]  replay_memory:  ' + str(replay_mem))
print('[deepRL]  batch_size:     ' + str(batch_size))
print('[deepRL]  gamma:          ' + str(gamma))
print('[deepRL]  epsilon_start:  ' + str(epsilon_start))
print('[deepRL]  epsilon_end:    ' + str(epsilon_end))
print('[deepRL]  epsilon_decay:  ' + str(epsilon_decay))
print('[deepRL]  allow_random:   ' + str(allow_random))
print('[deepRL]  debug_mode:     ' + str(debug_mode))
print('[deepRL]  model load:     ' + str(load_mode))# added by Ashwini for model loading functionality


#Data in Transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

## Replay Memory
#
#
class ReplayMemory(object):
	#Constructor
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

	## push() save a Transition
	#  @param self and arguments
	#  @return none
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

	##sample() samples randomly
	#  @param self and batch size
	#  @return sampled memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

	##_len() returns memory length
	#  @param self
	#  @return  memory length
    def __len__(self):
        return len(self.memory)


## Defines Deep Q-Network
#
class DQN(nn.Module):
	## The constructor.
	def __init__(self):
		print('[deepRL]  DQN::__init__()')
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
		self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn4 = nn.BatchNorm2d(32)
		self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn5 = nn.BatchNorm2d(32)

		self.head = None

	## forward()creates forward pass network
   	#  @param self and input image Data
   	#  @return convolutional output
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
		x = F.relu(self.bn4(self.conv4(x)))
		#x = F.relu(self.bn5(self.conv5(x)))

		y = x.view(x.size(0), -1)

		if self.head is None:
			print('[deepRL]  nn.Conv2d() output size = ' + str(y.size(1)))
			self.head = nn.Linear(y.size(1), num_actions)

			if use_cuda:
				self.head.cuda()

		return self.head(y)

## Defines Dueling DQN Structure
#
#
class Dueling_DQN(nn.Module):
	## The constructor
    def __init__(self):
		print('[deepRL]  Duelling DQN::__init__()')
		super(Dueling_DQN, self).__init__()
		'''self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
    #self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
    #self.bn3 = nn.BatchNorm2d(128)
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
		self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
    #self.bn4 = nn.BatchNorm2d(128)
		#self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)'''
		self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(128)
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
		#self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
		#self.bn4 = nn.BatchNorm2d(128)
		#self.conv5 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
		#self.bn5 = nn.BatchNorm2d(128)


		self.fc1_adv = nn.Linear(in_features=7*7*128, out_features=512)#Adel mohamed to increase size from 64 to 128
		self.fc1_val = nn.Linear(in_features=7*7*128, out_features=512)#Adel mohamed to increase size from 64 to 128

		self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
		self.fc2_val = nn.Linear(in_features=512, out_features=1)
		self.relu = nn.ReLU()

	## forward()creates forward pass network
    #  @param self and input image data
    #  @return data related to value and advantage network
    def forward(self, x):
		x = self.relu(self.bn1(self.conv1(x))) #Edo team apply normaliztion
		x = self.relu(self.bn2(self.conv2(x))) #Edo team apply normaliztion
		x = self.relu(self.bn3(self.conv3(x))) #Edo team apply normaliztion
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
	  #x = self.relu(self.bn4(self.conv4(x))) #Edo team apply normaliztion
	  #x = self.relu(self.bn5(self.conv5(x)))

		x = x.view(x.size(0), -1)

		adv = self.relu(self.fc1_adv(x))
		val = self.relu(self.fc1_val(x))

		adv = self.fc2_adv(adv)
		val = self.fc2_val(val).expand(x.size(0), num_actions)

		x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), num_actions)
		return x

## @Defines Deep Recurrent Q-Network (with LSTM)
#
class DRQN(nn.Module):
	## The constructor
	def __init__(self):
		print('[deepRL]  DRQN::__init__()')
		super(DRQN, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
		self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn4 = nn.BatchNorm2d(32)
		#self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		#self.bn5 = nn.BatchNorm2d(32)

		#LSTM and head needs to update if there is any change in number of action or input size
		#Dynamic update is possible from forward function but it will create problem in model loading
		self.lstm = nn.LSTMCell(800, lstm_size) #was None
		self.head = nn.Linear(lstm_size, num_actions) # was None

	## forward()creates forward pass network
	#  @param self and input image
	#  @return data from LSTM
	def forward(self, inputs):
		x, (hx, cx) = inputs
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
    #New Adel mohamed to increase size from 64 to 128
		#if input_width >= 128 and input_height >= 128:
		x = F.relu(self.bn4(self.conv4(x)))
		#x = F.relu(self.bn5(self.conv5(x)))


		y = x.view(x.size(0), -1)

		if self.lstm is None:
			print('[deepRL]  nn.Conv2d() output size = ' + str(y.size(1)))
			self.lstm = nn.LSTMCell(y.size(1), lstm_size)

			if use_cuda:
				self.lstm.cuda()

		if self.head is None:
			self.head = nn.Linear(lstm_size, num_actions)

			if use_cuda:
				self.head.cuda()

		hx, cx = self.lstm(y, (hx, cx))
		y = hx

		return self.head(y), (hx, cx)

	## Initialise the states (An image)
	#  @param self and batch dimension
	#  @return tuple of the LSTM hidden and cell states (hx,cx)
	def init_states(self, batch_dim):
		hx = Variable(torch.zeros(batch_dim, lstm_size))
		cx = Variable(torch.zeros(batch_dim, lstm_size))
		return hx, cx

	## reset states
	#  @param self and states
	#  @return states
	def reset_states(self, hx, cx):
		hx[:, :] = 0
		cx[:, :] = 0
		return hx.detach(), cx.detach()


#
# Create model instance
#
print('[deepRL]  creating DQN model instance')

lstm_actor_hx = lstm_actor_cx = None
lstm_batch_hx = lstm_batch_cx = None
lstm_final_hx = lstm_final_cx = None

if use_lstm:

	model= DRQN()#to create DRQN instance
	print(model)
	#params = list(model.parameters())
	#print("Learnable Parameters: ",len(params))
	#print("Initial   conv1   weights: ",params[0])#.size())

	lstm_actor_hx, lstm_actor_cx = model.init_states(1)
	lstm_batch_hx, lstm_batch_cx = model.init_states(batch_size)
	lstm_final_hx, lstm_final_cx = model.init_states(batch_size)

	print('[deepRL]  LSTM (hx, cx) size = ' + str(lstm_actor_hx.size(1)))


else:
	model = DQN()               #to create DQN instance
	#model = Dueling_DQN()      #to create Dueling DQN instance

print('[deepRL]  DQN model instance created')

if use_cuda:
    model.cuda()

memory = ReplayMemory(replay_mem)
steps_done = 0
temp=0


## load_model() helps to load checkpoint, handles layers like Batchnorm
#  @param filename.
#  @return none
def load_model(filename):
	global lstm_actor_hx
	global lstm_actor_cx

	print('[deepRL] loading model checkpoint from ' + filename)
	#1 model.load_state_dict(torch.load(filename)

	#2 model.load_state_dict(torch.load('last_brain_direct.pth.tar'))

	#3
	pretrained_dict =torch.load('drqn_checkpoints/'+filename+'.pth.tar')

	model_dict =model.state_dict()

	'''
	#Below code is useful to debug layers
	last_state = pretrained_dict['state_dict']
	last_step = pretrained_dict['steps']
	last_optimizer = pretrained_dict['optimizer']
	#print("Optimizer-",last_optimizer, "last_step-", last_step)

	print("Model parameters:")
	for v in model_dict.keys():
		print(v)
	print("Pretrained Model parameters:")
	for w in last_state.keys():
		print(w)'''

	# 1. filter out unnecessary keys
	#pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(pretrained_dict)



## save_model() saves checkpoint
#  @param filename.
#  @return none
def save_model(filename):
	print('[deepRL]  saving model checkpoint to ' + filename)
	#1
	#torch.save(model, 'drqn_checkpoints/'+filename+'.pth.tar')


	#2
	torch.save(model.state_dict(),'drqn_checkpoints/'+filename+'.pth.tar')

	#3 Below code is useful to debug layers
	'''torch.save({'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
					'steps' : steps_done,
                   }, 'drqn_checkpoints/last_brain.pth.tar')'''

if (optimizer == 'Adam'):
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

elif (optimizer == 'RMSprop'):
	optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

else:
	print('Optimizer Error. Make sure you have choosen the right optimizer and learning rate')
	sys.exit()


## Select agent action
#  @param state and allow ramdom flag
#  @return action
def select_action(state, allow_rand):
	global steps_done
	global lstm_actor_hx
	global lstm_actor_cx
	sample = random.random()

	eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
		math.exp(-1. * steps_done / epsilon_decay)

	steps_done += 1
	if not allow_rand or sample > eps_threshold:
		if use_lstm:
			action, (lstm_actor_hx, lstm_actor_cx) = model(
				(Variable(state, volatile=True).type(FloatTensor), (lstm_actor_hx, lstm_actor_cx)))
			#print("Q-values", action.data)
			action = action.data.max(1)[1].unsqueeze(0)
		else:
			action = model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].unsqueeze(0)

		#print('select_action = ' + str(action))
		return action
	else:
		#print('[deepRL]  DQN selected exploratory random action')
		return LongTensor([[random.randrange(num_actions)]])

print('[deepRL]  DQN script done init')


## Training routine-optimizes Q-function
#  @param none
#  @return none
def optimize_model():
	global last_sync
	global lstm_batch_hx
	global lstm_batch_cx
	global lstm_final_hx
	global lstm_final_cx

	if use_lstm:
		lstm_batch_hx, lstm_batch_cx = model.reset_states(lstm_batch_hx, lstm_batch_cx)
		lstm_final_hx, lstm_final_cx = model.reset_states(lstm_final_hx, lstm_final_cx)#target

	# sample a batch of transitions from the replay buffer
	if len(memory) < batch_size:
		return

	transitions = memory.sample(batch_size)

	# Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
		                                batch.next_state)))

	# We don't want to backprop through the expected action values and volatile
	# will save us on temporarily changing the model parameters'
	# requires_grad to False!
	non_final_next_states = Variable(torch.cat([s for s in batch.next_state
		                                      if s is not None]),
		                           volatile=True)
	#print(non_final_next_states)
	state_batch = Variable(torch.cat(batch.state))
	action_batch = Variable(torch.cat(batch.action))
	reward_batch = Variable(torch.cat(batch.reward))

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken
	if use_lstm:
		model_batch, (lstm_batch_hx, lstm_batch_cx) = model((state_batch, (lstm_batch_hx, lstm_batch_cx)))
	else:
		model_batch = model(state_batch)

	state_action_values = model_batch.gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	next_state_values = Variable(torch.zeros(batch_size).type(Tensor))

	if use_lstm:
		final_batch, (lstm_final_hx, lstm_final_cx) = model((non_final_next_states, (lstm_final_hx, lstm_final_cx)))
		next_state_values[non_final_mask] = final_batch.max(1)[0]
	else:
		next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

	# Now, we don't want to mess up the loss with a volatile flag, so let's
	# clear it. After this, we'll just end up with a Variable that has
	# requires_grad=False
	next_state_values.volatile = False
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * gamma) + reward_batch

	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in model.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

#
# C/C++ API hooks
#
last_action = None
last_state = None
curr_state = None
last_diff = None
curr_diff = None

## callback for infering next action based on new state
#  @param state
#  @return next action
def next_action(state_in):
	global last_state
	global curr_state
	global last_action
	global curr_diff
	global last_diff
	global temp

	#to verify correct weights have been loaded
	if(temp==0):
		print("Using weights:")
		params = list(model.parameters())
		print("Loaded  conv1  weights: ",params[0])#.size())
		temp=temp+1

	#print('state = ' + str(state.size()))
	state = state_in.clone().unsqueeze(0)
	#print('state = ' + str(state.size()))

	if curr_state is not None:
		last_state = curr_state

	if curr_diff is not None:
		last_diff = curr_diff

	curr_state = state
	#print("curr_state: ",curr_state.abs().sum())
	last_action = select_action(curr_state, allow_random)

	if last_state is not None:
		curr_diff = state - last_state
		#print(' curr_diff = ' + str(curr_diff.abs().sum()) + ' ' + str(curr_diff.max()) + ' ' + str(curr_diff.min()))
		#last_action = select_action(curr_diff, allow_random) #--

	'''else:
		curr_state = None
		curr_diff = None
		last_action = None'''

	if last_action is not None:
		#print('ret action = ' + str(last_action[0][0]))
		return last_action[0][0]
	else:
		#print('invalid action')
		return -1






## callback for recieving reward and peforming training
#  @param reward, end_episode
#  @return none
def next_reward(reward, end_episode):
	global last_state
	global curr_state
	global last_action
	global curr_diff
	global last_diff
	global lstm_actor_hx
	global lstm_actor_cx

	#print('reward = ' + str(reward))
	reward = Tensor([reward])

	if last_diff is not None and curr_diff is not None and last_action is not None:
		# store the transition in memory
		#memory.push(last_diff, last_action, curr_diff, reward)
		memory.push(last_state, last_action, curr_state, reward)
		#if end_episode:
		#	memory.push(curr_diff, last_action, None, reward)

		# perform one step of optimization on the target network
		#
		if(load_mode==False):
			optimize_model()

	if end_episode:
		last_state = None
		curr_state = None
		last_action = None
		curr_diff = None
		last_diff = None

		if use_lstm:
			lstm_actor_hx, lstm_actor_cx = model.reset_states(lstm_actor_hx, lstm_actor_cx)
