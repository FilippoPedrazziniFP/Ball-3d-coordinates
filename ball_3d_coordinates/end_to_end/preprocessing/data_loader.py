import pandas as pd
import glob
import cv2
import numpy as np

import ball_3d_coordinates.util.util as util

class Loader(object):
	def __init__(self, number_of_samples):
		super(Loader, self).__init__()
		self.number_of_samples = number_of_samples
		
	def get_labels(self):
		features = pd.read_csv(util.GEN_DATA_PATH + util.METAPATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		labels = features[["x", "y", "z"]]
		# Rescale from 0 to max
		labels["x"] = labels["x"].apply(lambda x: x + 600)
		labels['z'] = labels['z'].apply(lambda x: x + 450)
		return labels.iloc[0:self.number_of_samples]

	def load_data(self):
		features = self.get_features()
		labels = self.get_labels()
		return features, labels

	def get_features(self):
		images_list = sorted(glob.glob(util.GEN_DATA_PATH + '*.png'))
		return np.asarray(images_list[0:self.number_of_samples])

	def get_image_features(self, images):
		features = []
		for images_list in images:
			sequence = []
			for file_image in images_list:
				img = cv2.imread(file_image, 1)
				sequence.append(img)
			features.append(sequence)
		return np.asarray(features)
		