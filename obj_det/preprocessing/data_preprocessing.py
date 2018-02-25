from __future__ import absolute_import

import glob
import progressbar
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

import pandas as pd

_DATA_PATH = '../data/obj_det_fixed_camera/'
_METADATA_PATH = 'data.txt'
_CAMERA_DATA = 'camera.txt'

bar = progressbar.ProgressBar()

class Preprocessing(object):
	""" The class contains all the static methods to 
	import and preprocess the dataset. """
	def __init__(self):
		super(Preprocessing, self).__init__()

	@staticmethod
	def get_features(number_of_samples, img_height, img_width, input_channels):
		""" Load images """
		images_list = sorted(glob.glob(_DATA_PATH + '*.png'))
		features = []
		for file_image in bar(images_list[0:number_of_samples]):
			img = imread(file_image)
			# img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
			features.append(img)
		return np.asarray(features)

	@staticmethod
	def get_labels(number_of_samples, img_height, img_width, input_channels):
		features = pd.read_csv(_DATA_PATH + _METADATA_PATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		labels = features[["relative_x", "relative_y", "size"]]
		""" Standardize features """
		labels['relative_x'] = labels['relative_x'].apply(lambda x: x / img_width)
		labels['relative_y'] = labels['relative_y'].apply(lambda x: x / img_height)
		max_size = labels['size'].max()
		labels['size'] = labels['size'].apply(lambda x: x / max_size)
		return labels.values[0:number_of_samples]

	@staticmethod
	def get_data(number_of_samples, img_height, img_width, input_channels):
		""" The method accesses the data directory and loads images and labels """
		features = Preprocessing.get_features(number_of_samples, img_height, img_width, input_channels)
		cv2.imwrite('image_1.png', features[0])
		labels = Preprocessing.get_labels(number_of_samples, img_height, img_width, input_channels)
		return features, labels

	@staticmethod
	def check_img(images):
		""" The method shows the image """
		idx = random.randint(0, len(images))
		imshow(images[idx])
		print(images[idx].shape)
		plt.show()
		return

	@staticmethod
	def create_generators(X_train, y_train, X_val, y_val, X_test, y_test):
		""" The method creates the generators for fitting the model """
		train_datagen = ImageDataGenerator(rescale=1./255)
		
		""" Creating the iterators for the model """
		train_generator = train_datagen.flow(
			x=X_train,
			y=y_train,
			batch_size=32,
			shuffle=True,
			seed=0
			)

		validation_generator = train_datagen.flow(
			x=X_val,
			y=y_val,
			batch_size=1,
			shuffle=True,
			seed=0
			)

		test_generator = train_datagen.flow(
			x=X_test,
			y=y_test,
			batch_size=1,
			shuffle=True,
			seed=0
			)
		
		""" Look at what is happening in the generator """
		"""for x, y in train_generator:
			plt.imshow(x[0])
			break"""

		return train_generator, validation_generator, test_generator

	@staticmethod
	def preprocessing(number_of_samples, img_height, img_width, input_channels):
		""" The method does all the necessary steps to load and preprocess the data """
		features, labels = Preprocessing.get_data(number_of_samples, img_height, img_width, input_channels)
		# Preprocessing.check_img(features)
		X_train, X_test, y_train, y_test = Preprocessing.train_test_split(features, labels)
		X_train, X_val, y_train, y_val = Preprocessing.train_validation_split(X_train, y_train)

		train_generator, validation_generator, test_generator = Preprocessing.create_generators(X_train, y_train, X_val, y_val, X_test, y_test)

		return train_generator, validation_generator, test_generator

	@staticmethod
	def train_validation_split(features, labels, split_percentage=0.1):
		""" The method returns the data splitted into train and test based on the split percentage """
		X_train, X_test, y_train, y_test = train_test_split(features, labels, 
			test_size=split_percentage, random_state=0, shuffle=True)
		return X_train, X_test, y_train, y_test

	@staticmethod
	def train_test_split(features, labels, split_percentage=0.2):
		""" The method returns the data splitted into train and test based on the split percentage """
		X_train, X_test, y_train, y_test = train_test_split(features, labels, 
			test_size=split_percentage, random_state=0, shuffle=True)
		return X_train, X_test, y_train, y_test