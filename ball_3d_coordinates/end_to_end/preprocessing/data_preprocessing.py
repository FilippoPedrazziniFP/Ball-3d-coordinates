import math
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from random import randint

import ball_3d_coordinates.util.util as util

class ConvPreprocessor(object):
	def __init__(self, max_x, max_y, max_z, input_trace):
		super(ConvPreprocessor, self).__init__()
		self.max_x = max_x
		self.max_y = max_y
		self.max_z = max_z
		self.input_trace = input_trace

	def fit_transform(self, X, y):

		# Rescale the features /255
		X = self.rescale_features(X)

		# Rescale labels /IMG_DIM and transform it into a numpy array
		y = self.rescale_labels(y)

		# Generate the Data
		X, y = self.generate_data(X, y)

		# Reshaping the Label 
		y = np.squeeze(y)
		
		# Train, test, validation split
		X_train, X_test, X_val, y_train, y_test, y_val = self.train_test_validation_split(X, y)

		return X_train, y_train, X_test, y_test, X_val, y_val

	def transform(self, X):

		# Rescale the features for the prediction
		X = self.rescale_features(X)

		return X

	def rescale_labels(self, labels):
		# labels["x"] = labels["x"].apply(lambda x: x / self.max_x)
		# labels["y"] = labels["y"].apply(lambda x: x / self.max_y)
		# labels["z"] = labels["z"].apply(lambda x: x / self.max_z)
		return labels.values

	def rescale_features(self, features):
		features = features.astype('float32')
		features /= 255.0
		return features

	def train_test_validation_split(self, features, labels, val_samples=50, test_samples=100):

		X_test = features[0:test_samples]
		y_test = labels[0:test_samples]

		X_val = features[test_samples:test_samples + val_samples]
		y_val = labels[test_samples:test_samples + val_samples]

		X_train = features[test_samples + val_samples:]
		y_train = labels[test_samples + val_samples:]
		
		return X_train, X_test, X_val, y_train, y_test, y_val

	def generate_data(self, features, labels):
		data_x = []
		data_y = []
		for i in range(0, len(features)):
			initial_idx = randint(0, len(features)-self.input_trace-1)
			x = features[initial_idx:initial_idx+self.input_trace,:]
			y = labels[initial_idx+int(self.input_trace/2):initial_idx+int(self.input_trace/2)+1,:]
			data_x.append(x)
			data_y.append(y)
		data_x = np.asarray(data_x)
		data_y = np.asarray(data_y)
		
		return data_x, data_y
