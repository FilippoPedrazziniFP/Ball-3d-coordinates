from __future__ import absolute_import

import cv2
import numpy as np
from numpy.linalg import pinv

from camera_calibration.cam_cal import CameraCalibration

_CLASSES = 3
_INPUT_DIMENSION = 4
_HIDDEN_SIZE = 3

_INITIAL_LEARNING_RATE = 0.01
_NUMBER_OF_EPOCHS = 500

class FFNNCameraCalibration(CameraCalibration):
	""" The camera calibration is done following the 
		Neurocalibration paper in which they try to compute the 
		12 parameters of the projection matrix using a
		neural network approach. Given a certain number of samples
		they use this information to minimize the projection error. """
	def __init__(self):
		super(FFNNCameraCalibration, self).__init__()

	def compute_camera_calibration(self):
		""" Having W and V which are respectevely the intrisic parameters
			and the extrinsic ones, we can now compute the 
			entire projection matrix, de-normalizing the values """
		W, V = self.train()

		W[0][0] = W[0][0] * (self.S1_x_max - self.S1_x_min) + self.S1_x_min
		W[1][1] = W[1][1] * (self.S1_y_max - self.S1_y_min) + self.S1_y_min

		V[0][0] = V[0][0] * (self.S2_x_max - self.S2_x_min) + self.S2_x_min
		V[1][1] = V[1][1] * (self.S2_y_max - self.S2_y_min) + self.S2_y_min
		V[2][2] = V[2][2] * (self.S2_z_max - self.S2_z_min) + self.S2_z_min

		P = np.dot(W, np.transpose(V))

		return P

	def loss(self, model, X, D):
		""" Computation of the loss in order
			to keep track of the learning """

		V = model['V']
		W = model['W']
		L = model['L']

		""" Inference """
		Y = np.dot(X, V)
		Z = np.dot(Y, W)
		O = np.multiply(Z, L)

		return 1/2 * np.sum(np.power((D - O), 2))

	def normalize_2d(self, data):
		""" The method computes the 2D normalization 
			saving the parameters for computing the 
			inverse normalization """
		min_value = np.min(data[:,0])
		max_value = np.max(data[:,0])

		data[:,0] = (data[:,0] - min_value) / (max_value - min_value)

		self.S1_x_min = min_value
		self.S1_x_max = max_value

		min_value = np.min(data[:,1])
		max_value = np.max(data[:,1])

		data[:,1] = (data[:,1] - min_value) / (max_value - min_value)

		self.S1_y_min = min_value
		self.S1_y_max = max_value

		return data

	def normalize_3d(self, data):
		""" The method computes the 3D normalization 
			saving the parameters for computing the 
			inverse normalization """
		min_value = np.min(data[:,0])
		max_value = np.max(data[:,0])

		data[:,0] = (data[:,0] - min_value) / (max_value - min_value)

		self.S2_x_min = min_value
		self.S2_x_max = max_value

		min_value = np.min(data[:,1])
		max_value = np.max(data[:,1])

		data[:,1] = (data[:,1] - min_value) / (max_value - min_value)

		self.S2_y_min = min_value
		self.S2_y_max = max_value

		min_value = np.min(data[:,2])
		max_value = np.max(data[:,2])

		data[:,2] = (data[:,2] - min_value) / (max_value - min_value)

		self.S2_z_min = min_value
		self.S2_z_max = max_value

		return data

	def train(self):
		""" The method computes the training, initializing
			the weights, computing the gradient at each epoch
			and updating them accordingly to the learning step."""
		model = {}

		""" Features: adding 1 column of ones to fit the specifications."""
		X = self.get_unity_object_points()

		""" Normalization of the 3D points """
		X = self.normalize_3d(X)

		ones = np.ones((X.shape[0], 1))
		X = np.concatenate((X, ones), axis=1)
		
		""" Labels: adding 1 column of ones to fit the specifications. """
		D = self.get_unity_img_points()

		""" Normalization of the 2D points """
		D = self.normalize_2d(D)

		ones = np.ones((D.shape[0], 1))
		D = np.concatenate((D, ones), axis=1)
		
		np.random.seed(0)

		""" Initialization of the weigths with a random value
			between -1 and 1 """
		V = np.random.uniform(low=-1, high=1, size=(4, 3))
		W = np.random.uniform(low=-1, high=1, size=(3, 3))
		L = np.ones((X.shape[0], 3)) 

		lr = _INITIAL_LEARNING_RATE

		model = {'V': V, 'W': W, 'L': L}

		for epoch in range(0, _NUMBER_OF_EPOCHS):
			for i in range(0, X.shape[0]):
				""" Inference """
				Y = np.dot(X, V)
				Z = np.dot(Y, W)
				O = np.multiply(Z, L)

				""" W update """
				for k in range(0, W.shape[0]):
					for j in range(0, W.shape[1]):
						W[j][k] += + lr * (D[i][k] - O[i][k]) * Y[i][j]

				""" B update """

				""" V update """
				for l in range(0, V.shape[0]):
					for j in range(0, V.shape[1]):
						err = 0
						for k in range(0, 3):
							err += (D[i][k] - O[i][k]) * W[k][j] * X[i][l]
						V[l][j] += + lr * L[i][0] * err

				""" L update """
				sum_err = 0
				for k in range(0, 3):
					err = D[i][k] - O[i][k]
					fact = 0
					for j in range(0, 3):
						fact += W[k][j] * Y[i][j]
					sum_err += err * fact
				L[i] += + lr * sum_err

			model = {'V': V, 'W': W, 'L': L}

			if epoch % 10 == 0:
				print("Loss at epoch (%s): %s" %(epoch, self.loss(model, X, D)))


		return model['W'], model['V']


	



	