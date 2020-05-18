import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
	"""Actor (Policy) Model."""
	def __init__(self, state_size, action_size, seed):
		"""Initialize parameters and build model.
		Params
		======
		state_size (int): Dimension of each state
		action_size (int): Dimension of each action
		seed (int): Random seed used to initialize similar weights in target and local instances in dqn_agent file
		"""
		super(QNetwork, self).__init__() #init nn.Module
		self.seed = torch.manual_seed(seed)

		#init the model architecture
		self.fc1 = nn.Linear(state_size, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 128)
		self.out = nn.Linear(128, action_size)

		self.bn = nn.BatchNorm1d(num_features=state_size)

	def forward(self, state):
		x = F.relu(self.fc1(self.bn(state)))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.out(x)