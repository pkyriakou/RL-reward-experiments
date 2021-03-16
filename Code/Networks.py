import torch
import torch.nn as nn
import torch.nn.functional as F

import math

#Linear	
class Linear(nn.Module):

	def __init__(self,input_size,output_size):
		super(Linear, self).__init__()

		self.input_size=input_size
		self.fc = nn.Linear(self.input_size, output_size) 

	def forward(self, inputs) :
		# print(inputs.shape,self.input_size)
		out= torch.tensor(inputs).reshape(1,self.input_size)
		out = self.fc(out)
		return out
	

class CNN(nn.Module):
	def __init__(self,output_size):
		super(CNN, self).__init__()

		self.conv1_1 = nn.Conv2d(1, 10, kernel_size=3)
		self.conv1_2 = nn.Conv2d(10, 10, kernel_size=3)

		self.conv2_1 = nn.Conv2d(1, 10, kernel_size=5)

		self.mp = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(100, 50)
		self.fc2 = nn.Linear(50, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		x=x.unsqueeze(0)

		x_1 = self.relu(self.mp(self.conv1_1(x)))
		x_1 = self.relu(self.mp(self.conv1_2(x_1)))

		x_2 = self.relu(self.mp(self.conv2_1(x)))

		x = torch.cat((x_1.view(-1),x_2.view(-1)))  # flatten the tensor
		x = self.relu(self.fc1(x))
		x = self.fc2(x)

		return x.unsqueeze(0)

class Deep(nn.Module):
	def __init__(self,input_size,output_size):
		super(Deep, self).__init__()

		self.input_size=input_size
		self.h1_size=int((input_size+output_size)/2)
		self.h2_size=int((input_size+output_size)/2)

		self.fc1 = nn.Linear(self.input_size, self.h1_size) 
		self.fc2 = nn.Linear(self.h1_size, self.h2_size) 
		self.fc3 = nn.Linear(self.h2_size, output_size) 
		self.relu = nn.ReLU()

	def forward(self, inputs) :
		out = self.fc1(inputs)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out
