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
		labels = features[["relative_x", "relative_y", "size"]]
		return labels.iloc[0:self.number_of_samples]

	def load_data(self):
		features = self.get_features()
		labels = self.get_labels()
		return features, labels

	def get_features(self):
		images_list = sorted(glob.glob(util.GEN_DATA_PATH + '*.png'))
		features = []
		for file_image in images_list[0:self.number_of_samples]:
			img = cv2.imread(file_image, 1)
			features.append(img)
		return np.asarray(features)
		