import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
	"""Actor (Policy) Model."""
	def __init__(self, state_size, action_size, seed):
		"""Initialize parameters and build model.
		Params
		======
		state_size (int): Dimension of each state
		action_size (int): Dimension of each action
		seed (int): Random seed used to initialize similar weights in target and local instances in dqn_agent file
		"""
		super(Qnetwork, self).__init__() #init nn.Module
		self.seed = torch.manual_seed(seed)

		#init the model architecture
		self.fc1 = nn.Linear(state_size, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.out = nn.Linear(64, action_size)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.out(x)