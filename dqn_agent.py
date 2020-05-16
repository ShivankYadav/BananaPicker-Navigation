import random
import numpy as np
from collections import namedtuple, deque

from model import Qnetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

#hyper_params
LR = 5e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
GAMMA = 0.99
TAU = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
	"""Interacts with and learns from the environment"""

	def __init__(self, state_size, action_size, seed):
		"""Initialize an Agent object.

		Params
		======
		state_size (int): dimension of each state
		action_size (int): dimension of each action
		seed (int): random seed
		"""
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		# init both target and local DQN networks and upload them to gpu if avialable
		self.Qnetwork_local = Qnetwork(state_size, action_size, seed).to(device)
		self.Qnetwork_target = Qnetwork(state_size, action_size, seed).to(device)
		# init optimizer
		self.optimizer = optim.Adam(self.Qnetwork_local.parameters(), lr = LR)

		#init replay buffer memory
		self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
		self.t_step = 0


	def step(self, state, action, reward, next_state, done):
		""" save the experience in buffer and learn every UPDATE_EVERY steps
		"""
		self.memory.add(state, action, reward, next_state, done)

		self.t_step = (self.t_step + 1) % UPDATE_EVERY 
		if self.t_step == 0:
			# if enough data in memory call learn method.
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	def learn(self, experiences, gamma):
		""" Function to update the model. Gradient descent on batch of experience tuples
		Params
		======
		experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
		gamma (float): discount factor
		"""
		# decouple information
		states, actions, rewards, next_states, dones = experiences 
		Q_target_av = self.Qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) #since max() returns 2 values
		Q_targets = rewards + gamma*(Q_target_av)*(1-dones)
		Q_expected = self.Qnetwork_local(states).gather(1, actions) # get q value for corrosponding action along dimension 1 of 64,4 matrix

		# applying gradient descent
		loss = F.mse_loss(Q_expected, Q_targets)
		self.optimizer.zero_grad()
		loss.backward() # apply pytorch autograd
		self.optimizer.step()

		# soft update the target network according to hyper parameter TAU
		self.soft_update(self.Qnetwork_local, self.Qnetwork_target, TAU)

	def soft_update(local_model, target_model, tau):
		"""Function to update the weights of target network"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data) # update inplace


	def act(self, state, eps = 0.01):
		""" Returns the action from the state as per current policy
		# pass the state to the Qnetwork_local and select the action using e-greedy policy

		Params
		======
		state (ndarray_like): current state
		eps (float): epsilon, for epsilon-greedy action selection
		"""

		# convert state to tensor, add batch dimension and upload to device if available.
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.Qnetwork_local.eval() #put dropout and batchnorm in eval mode
		# pause history creation for autograd.
		with torch.no_grad():
			q_values = self.Qnetwork_local(state)
		self.Qnetwork_local.train() # switch back to train mode

		# follow epsilon greedy policy to select action
		if random.random() > eps:
			return np.argmax(q_values.cpu().data.numpy())  # since uint dimesion  therefore axis not specified
		else:
			return random.choice(np.arange(self.action_size))




class ReplayBuffer():
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, BUFFER_SIZE, BATCH_SIZE, seed):
		"""Initialize a ReplayBuffer object.

		Params
		======
		action_size (int): dimension of each action
		buffer_size (int): maximum size of buffer
		batch_size (int): size of each training batch
		seed (int): random seed
		"""
		self.buffer_size = BUFFER_SIZE
		self.batch_size = BATCH_SIZE
		self.seed = random.seed(seed)
		#init memory
		self.memory = deque(maxlen = BATCH_SIZE)
		#init experience namedtuple
		self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])

	def add(self, state, action, reward, next_state, done):
		"""Append data into memory in form of exprience"""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		"""	sample a mini-batch of experience tupples from memory 
		"""
		experiences = random.sample(self.memory, k = self.batch_size)

		# np.vstack converts single dim [] to vector if dim x 1.
		# seperate the data, convert it to torch tensor and upload it to device before returning
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).asType(np.uint8)).float().to(device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		return len(self.memory)



