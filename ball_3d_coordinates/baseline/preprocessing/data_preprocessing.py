import pandas as pd
import numpy as np

from random import randint

class BaselinePreprocessor(object):
	"""The class contains all the static methods to 
	import and preprocess the dataset."""
	def __init__(self, input_trace):
		super(BaselinePreprocessor, self).__init__()
		self.input_trace = input_trace

	def fit_transform(self, X, y, noise=False):

		# Concatenate features and labels
		data = pd.concat([X, y], axis=1)

		# Filter the data 
		data = self.filtering_noisy_samples(data.values)

		# Get list of indexes
		train_list_complete, train_list = self.get_indexes_list(data)

		# Splitting the data into features and labels
		X, y = self.features_labels_split(data)

		# Apply Noise
		if noise==True:
			X = self.apply_noise(X)

		# Generate the sequences
		X, y = self.generate_data(X, y, train_list_complete, train_list)

		# Train, test, validation split
		X_train, y_train, X_test, y_test, X_val, y_val = self.train_test_validation_split(X, y)

		return X_train, y_train, X_test, y_test, X_val, y_val

	def get_indexes_list(self, data):
		train_list = []
		for i in range(0, len(data)):
			if data[i][2] != data[i-1][2] and i != 0:
				train_list.append(i-1)

		train_list_complete = []
		for idx in train_list:
			train_list_complete.append(np.arange(idx-self.input_trace + 1, idx + 1))
		train_list_complete = np.asarray(train_list_complete).flatten()
		return train_list_complete, train_list

	def filtering_noisy_samples(self, data):
		data_proc = []
		current_idx = 0
		for i, sample in enumerate(data):
			if data[i][2] != data[i-1][2] and i != 0:
				if data[i-1][4] < 6:
					data_proc.extend(data[current_idx:i,:])
				current_idx = i
		data_proc = np.asarray(data_proc)
		return data_proc

	def features_labels_split(self, data):
		features = data[:,0:2]
		labels = data[:,3:]
		return features, labels

	def standardize_features(self, features):
		sc = StandardScaler().fit(features)
		features = sc.transform(features)
		return features

	def generate_data(self, features, labels, train_list_complete, train_list):
		data_x = []
		data_y = []
		for i in range(0, len(train_list)-1):
			for k in range(0, 10):
				initial_idx = randint(train_list[i], train_list[i+1])
				while initial_idx in train_list_complete:
					initial_idx = randint(train_list[i], train_list[i+1])

				x = features[initial_idx:initial_idx+self.input_trace,:]
				y = labels[initial_idx+int(self.input_trace/2):initial_idx+int(self.input_trace/2)+1,:]

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
		features[:,0] = features[:,0] + noise

		# To the y
		noise = np.random.normal(mean, std_dev_x_y, len(features))
		features[:,1] = features[:,1] + noise
		
		return features