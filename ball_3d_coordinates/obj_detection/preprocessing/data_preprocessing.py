import math
import pandas as pd

from sklearn.utils import shuffle

import ball_3d_coordinates.util.util as util

class ConvPreprocessor(object):
	def __init__(self, number_of_samples_train):
		super(ConvPreprocessor, self).__init__()
		self.number_of_samples_train = number_of_samples_train

	def fit_transform(self, X, y):

		# Rescale the features /255
		X = self.rescale_features(X)

		# Rescale labels /IMG_DIM and transform it into a numpy array
		y = self.rescale_labels(y)
		
		# Shuffle the data
		X, y = shuffle(X, y, random_state=0)

		# Train, test, validation split
		X_train, X_test, X_val, y_train, y_test, y_val = self.train_test_validation_split(
			X[0:self.number_of_samples_train], y[0:self.number_of_samples_train])

		return X_train, y_train, X_test, y_test, X_val, y_val

	def transform(self, X):

		# Rescale the features for the prediction
		X = self.rescale_features(X)

		return X

	def rescale_labels(self, labels):
		labels['relative_x'] = labels['relative_x'].apply(lambda x: x / util.IMG_WIDTH)
		labels['relative_y'] = labels['relative_y'].apply(lambda x: x / util.IMG_HEIGHT)
		labels['size'] = labels['size'].apply(lambda x: math.sqrt(x))
		return labels.values

	def rescale_features(self, features):
		features = features.astype('float32')
		features /= 255.0
		return features

	def train_test_validation_split(self, features, labels, val_samples=10, test_samples=50):

		X_test = features[0:test_samples]
		y_test = labels[0:test_samples]

		X_val = features[test_samples:test_samples + val_samples]
		y_val = labels[test_samples:test_samples + val_samples]

		X_train = features[test_samples + val_samples:]
		y_train = labels[test_samples + val_samples:]
		
		return X_train, X_test, X_val, y_train, y_test, y_val

