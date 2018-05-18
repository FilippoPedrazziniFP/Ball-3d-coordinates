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
		features = features[["relative_x", "relative_y", "size", "traj"]]
		return features, labels

	def load_data(self):
		features, labels = self.get_labels_and_features()
		return features, labels