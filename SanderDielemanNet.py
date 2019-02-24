import torch.nn as nn
import torch.nn.functional as F

class SanderDielemanNet(nn.Module): 
	def __init__(self, num_classes=37): 
		super(SanderDielemanNet, self).__init__() 
		# Convolutional and MaxPool layers 
		self.conv1 = nn.Conv2d(3, 32, 6)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(32, 64, 5)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(64, 128, 3)
		self.conv4 = nn.Conv2d(128, 128, 3)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		# Dense layers
		self.fc1 = nn.Linear(128*2*2, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, num_classes)

	def forward(self, x): 
		# Convolutional and MaxPool layers 
		x = F.relu(self.conv1(x)) 
		x = self.pool1(x) 
		x = F.relu(self.conv2(x)) 
		x = self.pool2(x) 
		x = F.relu(self.conv3(x)) 
		x = F.relu(self.conv4(x)) 
		x = self.pool4(x)
		# Dense layers 
		x = x.view(-1, 128*2*2)
		x = F.relu(self.fc1(x)) 
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x)) 
		return(x)


