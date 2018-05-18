import pandas as pd

import ball_3d_coordinates.util.util as util

class Loader(object):
	def __init__(self, predictions=False):
		super(Loader, self).__init__()
		self.predictions = predictions

	def get_conv_predictions(self):
		features = pd.read_csv(util.PREDICTIONS_DATA_PATH, sep=',', index_col=0)
		return features
	
	def get_labels(self):
		features = pd.read_csv(util.GEN_DATA_PATH + util.METAPATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		labels = features[["x", "y", "z"]]
		return labels

	def get_labels_and_features(self):
		features = pd.read_csv(util.GEN_DATA_PATH + util.METAPATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		labels = features[["x", "y", "z"]]
		features = features[["relative_x", "relative_y", "size"]]
		return features, labels

	def load_data(self):
		if self.predictions == True:
			features = self.get_conv_predictions()
			labels = self.get_labels()
		else:
			features, labels = self.get_labels_and_features()
		return features, labels