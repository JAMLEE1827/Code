import torch.nn as nn

class CNNEncoder(nn.Module):
	def __init__(self, in_channels, size, channels=(32, 64, 128)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(CNNEncoder, self).__init__()
		self.size = size
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.BatchNorm2d(channels[0]),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		))

		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.BatchNorm2d(channels[i+1]),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.BatchNorm2d(channels[i+1]),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Dropout(0.4)))

		flattened_dim = channels[-1] * (self.size // 32) * (self.size // 32)
		intermediate_dim = 512
		self.fc1 = nn.Linear(flattened_dim, intermediate_dim)
		self.fc2 = nn.Linear(intermediate_dim, 64)
		self.dropout = nn.Dropout(0.5)  # Dropout after the FC
		self.final_activation = nn.ReLU(inplace=True)  # Final activation

	def forward(self, x):
		for stage in self.stages:
			x = stage(x)
		x = x.view(x.size(0), -1)  # Flatten
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return self.final_activation(x)