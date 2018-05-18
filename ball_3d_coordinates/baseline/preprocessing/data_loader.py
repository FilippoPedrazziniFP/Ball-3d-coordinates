import pandas as pd
import numpy as np

import ball_3d_coordinates.util.util as util

class Loader(object):
	def __init__(self):
		super(Loader, self).__init__()

	def get_labels_and_features(self):
		features = pd.read_csv(util.TRAJ_DATA_PATH + util.METAPATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		labels = features[["x", "y", "z"]]
		features = features[["relative_x", "relative_y", "traj"]]
		return features, labels

	def get_camera_points(self):
		camera_parameters = pd.read_csv(util.TRAJ_DATA_PATH + util.CAMERA_PATH, header = None, sep=',')
		return camera_parameters.values[0]

	def load_data(self):
		features, labels = self.get_labels_and_features()
		camera_parameters = self.get_camera_points()
		return features, labels, camera_parameters
