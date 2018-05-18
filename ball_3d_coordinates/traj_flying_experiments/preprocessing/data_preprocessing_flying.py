import pandas as pd
import numpy as np

from random import randint
from sklearn.preprocessing import StandardScaler


class TmpFlyingPreprocessor(object):
	def __init__(self, number_of_samples, input_trace):
		super(TmpFlyingPreprocessor, self).__init__()
		self.number_of_samples = number_of_samples
		self.input_trace = input_trace

	def fit_transform(self, X, y, noise=False):

		# Apply Noise
		if noise==True:
			X = self.apply_noise(X)

		# Standardize Features and transform into numpy array
		X = self.standardize_features(X)

		# Transform the labels from DF to numpy array
		y = self.transform_labels(y)
		
		# Generate Sequences 
		X, y = self.generate_data(X, y)

		# Train, test, validation split
		X_train, y_train, X_test, y_test, X_val, y_val = self.train_test_validation_split(X, y)

		# Reshape the labels for training
		y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
		y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
		y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))

		return X_train, y_train, X_test, y_test, X_val, y_val

	def standardize_features(self, features):
		sc = StandardScaler().fit(features)
		features = sc.transform(features)
		return features

	def transform_labels(self, labels):		
		return labels.values

	def generate_data(self, features, labels):
		data_x = []
		data_y = []
		for i in range(0, len(features)-1):
			initial_idx = randint(0, len(features)-self.input_trace-1)
			x = features[initial_idx:initial_idx+self.input_trace,:]
			y = labels[initial_idx+int(self.input_trace/2):initial_idx+int(self.input_trace/2)+1]
			data_x.append(x)
			data_y.append(y)
		data_x = np.asarray(data_x)
		data_y = np.asarray(data_y)
		
		return data_x, data_y

	def train_test_validation_split(self, features, labels, val_samples=10, test_samples=50):

		X_test = features[0:test_samples]
		y_test = labels[0:test_samples]

		X_val = features[test_samples:test_samples + val_samples]
		y_val = labels[test_samples:test_samples + val_samples]

		X_train = features[test_samples + val_samples:]
		y_train = labels[test_samples + val_samples:]
		
		return X_train, y_train, X_test, y_test, X_val, y_val

	def apply_noise(self, features, mean=0.0, std_dev_size=2.5, std_dev_x_y=3.0):

		# To the x
		noise = np.random.normal(mean, std_dev_x_y, len(features))
		features["relative_x"] = features["relative_x"] + noise

		# To the y
		noise = np.random.normal(mean, std_dev_x_y, len(features))
		features["relative_y"] = features["relative_y"] + noise

		# To the size
		noise = np.random.normal(mean, std_dev_size, len(features))
		# Clip the noise
		noise[noise > std_dev_size] = std_dev_size
		noise[noise < -std_dev_size] = -std_dev_size
		features["size"] = features["size"] + noise

		return features
		