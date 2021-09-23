import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures

import math


class Linear(nn.Module):  # Class defining a linear model that can also polynomial basis functions

	def __init__(self, input_size, output_size, degree):
		super(Linear, self).__init__()

		self.input_size = input_size
		self.degree = degree

		# we create a dummy vector of the same shape as the environment observation
		dummy = numpy.random.randint(2, size=(1,self.input_size))
		# we raise to a polynomial of the degree chosen
		dummy_poly = PolynomialFeatures(self.degree, include_bias=False)
		# we save the size of the augmented environment vector
		size = dummy_poly.fit(dummy).n_output_features_
		# size = self.input_size*self.degree
		self.fc = nn.Linear(size, output_size)

	def forward(self, inputs):
		out = torch.tensor(inputs).reshape(1,self.input_size)
		out = self.fc(make_features(out, self.degree).float())
		return out


def make_features(x, degree):  # Method that takes data and degree as input and transforms the data
	# by raising them to a specific degree

	poly = PolynomialFeatures(degree, include_bias=False)
	out = torch.tensor(poly.fit_transform(x))

	return out


def make_features_naive(x, degree):  # Method that takes data and degree as input and transforms the data
	# by raising them to a specific degree - naively

	tmp = [x ** i for i in range(1, degree+1)]
	out = torch.cat(tmp, 1)

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
