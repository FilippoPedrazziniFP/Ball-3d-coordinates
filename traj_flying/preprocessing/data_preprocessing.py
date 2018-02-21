from __future__ import absolute_import

from PIL import Image
import glob
import os
import pandas as pd
import numpy as np
from random import randint
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import pinv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from camera_calibration.cam_cal_dic import CameraCalibrationBuilder
from camera_calibration.cam_cal_unity import UnityCameraCalibration
from sklearn.preprocessing import scale, normalize

_DATASET_PATH = './../data/'
_METADATA_PATH = 'data.txt'
_CAMERA_DATA = 'camera.txt'

_K = 10

class Preprocessing(object):
	""" The class contains all the static methods to 
	import and preprocess the dataset. """
	def __init__(self):
		super(Preprocessing, self).__init__()

	@staticmethod
	def compute_camera_projection(img_points, obj_points, calibration_method):
		""" The method computes the camera calibration using certain points in the filed.
			At each frame it has to compute the mapping in order to have a better prediction. """
		calibration = CameraCalibrationBuilder.build(calibration_method)
		img_points = np.reshape(img_points, (-1, 2))
		obj_points = np.reshape(obj_points, (-1, 3))
		proj_matrix = calibration.compute_camera_calibration(img_points, obj_points)

		return proj_matrix

	@staticmethod
	def features_transformation(features, camera_features, calibration_method='linear'):
	    """ The method computes the feature transformation, in order to have
	    as network input: X, Y, Size, Projection Matrix """
	    data = []
	    for i, camera in enumerate(camera_features):
	        img_points = camera[36:]
	        obj_points = camera[0:36]
	        proj_matrix = Preprocessing.compute_camera_projection(img_points, obj_points, calibration_method)
	        data.append(np.append(features[i], proj_matrix.flatten(), axis=0))
	    
	    data = np.asarray(data)
	    return data

	@staticmethod
	def features_transformation_ground(features, camera_features, calibration_method='linear'):
	    """ The method computes the feature transformation, in order to have
	    as network input: X, Y, Size, Projection Matrix """
	    data = []
	    for i, camera in enumerate(camera_features):
	        img_points = camera[24:]
	        obj_points = camera[0:24]
	        proj_matrix = Preprocessing.compute_camera_projection(img_points, obj_points, calibration_method)
	        data.append(np.append(features[i], proj_matrix.flatten(), axis=0))
	    
	    data = np.asarray(data)
	    return data

	@staticmethod
	def import_video_data_proj(directory, calibration_method='linear', ground_calibration=False):
		""" Returns a numpy array with (index_of_the_image, x_position, y_position, z_position, size) """
		features = pd.read_csv(directory + '/' + _METADATA_PATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size"]
		
		camera_features = pd.read_csv(directory + '/' + _CAMERA_DATA, header = None, sep=',')
		camera_features.columns = ["x_c_1", "y_c_1", "z_c_1", 
									"x_c_2", "y_c_2", "z_c_2", 
									"x_c_3", "y_c_3", "z_c_3",
									"x_c_4", "y_c_4", "z_c_4",

									"x_mc_1", "y_mc_1", "z_mc_1", 
									"x_mc_2", "y_mc_2", "z_mc_2", 
									"x_mc_3", "y_mc_3", "z_mc_3",
									"x_mc_4", "y_mc_4", "z_mc_4",

									"x_cb_1", "y_cb_1", "z_cb_1", 
									"x_cb_2", "y_cb_2", "z_cb_2", 
									"x_cb_3", "y_cb_3", "z_cb_3",
									"x_cb_4", "y_cb_4", "z_cb_4",

									"xi_c_1", "yi_c_1",
									"xi_c_2", "yi_c_2", 
									"xi_c_3", "yi_c_3",
									"xi_c_4", "yi_c_4",

									"xi_mc_1", "yi_mc_1",
									"xi_mc_2", "yi_mc_2", 
									"xi_mc_3", "yi_mc_3",
									"xi_mc_4", "yi_mc_4",

									"xi_cb_1", "yi_cb_1",
									"xi_cb_2", "yi_cb_2", 
									"xi_cb_3", "yi_cb_3",
									"xi_cb_4", "yi_cb_4" ]
		if ground_calibration == True:
			camera_features = camera_features[["x_c_1", "y_c_1", "z_c_1", 
									"x_c_2", "y_c_2", "z_c_2", 
									"x_c_3", "y_c_3", "z_c_3",
									"x_c_4", "y_c_4", "z_c_4",

									"x_mc_1", "y_mc_1", "z_mc_1", 
									"x_mc_2", "y_mc_2", "z_mc_2", 
									"x_mc_3", "y_mc_3", "z_mc_3",
									"x_mc_4", "y_mc_4", "z_mc_4",

									"xi_c_1", "yi_c_1",
									"xi_c_2", "yi_c_2", 
									"xi_c_3", "yi_c_3",
									"xi_c_4", "yi_c_4",

									"xi_mc_1", "yi_mc_1",
									"xi_mc_2", "yi_mc_2", 
									"xi_mc_3", "yi_mc_3",
									"xi_mc_4", "yi_mc_4" ]]
			features = Preprocessing.features_transformation_ground(features.values, 
				camera_features.values, calibration_method)
		else:
			features = Preprocessing.features_transformation(features.values, 
				camera_features.values, calibration_method)
		
		return features

	@staticmethod
	def import_video_data(directory):
		""" Returns a numpy array with (index_of_the_image, x_position, y_position, z_position) """
		features = pd.read_csv(directory + '/' + _METADATA_PATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size"]
		return features.values

	@staticmethod
	def import_fixed_dataset(data_path):
		""" The method returns features and labels from the metadata of each video. """
		data = []
		for subdir, dirs, files in os.walk(_DATASET_PATH + data_path + '/'):
			for video in dirs:
				data.append(Preprocessing.import_video_data(_DATASET_PATH + data_path + '/' + video))
		return data

	@staticmethod
	def import_moving_dataset(data_path, calibration_method='linear', ground_calibration=False):
		""" The method returns features and labels from the metadata of each video. """
		data = []
		for subdir, dirs, files in os.walk(_DATASET_PATH + data_path + '/'):
			for video in dirs:
				data.append(Preprocessing.import_video_data_proj(_DATASET_PATH + data_path + '/' + video))
		return data

	@staticmethod
	def standardize_features(X_train, X_test):
		
		sc = StandardScaler().fit(X_train)

		X_train = sc.transform(X_train)
		X_test = sc.transform(X_test)

		return X_train, X_test

	@staticmethod
	def labels_to_categorical(labels):
		""" The method takes the y columns and transorm it into categorical (0, 1) """
		y_labels = labels[:,1]
		y_labels_bool = y_labels < 5.5
		y_labels_bool = np.reshape(y_labels_bool, (-1, 1))
		enc = OneHotEncoder()
		enc.fit(y_labels_bool)
		binary_labels = enc.transform(y_labels_bool).toarray()
		return binary_labels

	@staticmethod
	def labels_to_binary(labels):
		""" The method takes the y columns and transorm it into categorical (0, 1) """
		y_labels = labels[:,1]
		y_labels_bool = y_labels < 5.5
		binary_labels = y_labels_bool.astype(int)
		return binary_labels

	@staticmethod
	def filtering_noisy_samples(data, number_of_samples):
		""" The method filters the data based on the position of the ball in the last 
			moment of the trajectory """
		data_preprocessed = []
		current_idx = 0
		for i, sample in enumerate(data[0:number_of_samples,:]):
			if data[i][7] != data[i-1][7] and i != 0:
				if data[i-1][2] < 6:
					data_preprocessed.extend(data[current_idx:i,:])
				current_idx = i

		train_list = []
		for i in range(0, len(data_preprocessed)):
			if data_preprocessed[i][7] != data_preprocessed[i-1][7] and i != 0:
				train_list.append(i-1)

		""" Deleting last feature """
		data_preprocessed = np.asarray(data_preprocessed)[:,0:7]
		return data_preprocessed, train_list

	@staticmethod
	def import_flex_video_data(directory):
		""" Returns a numpy array with (index_of_the_image, x_position, y_position, z_position) """
		features = pd.read_csv(directory + '/' + _METADATA_PATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		return features.values

	@staticmethod
	def import_flex_video_data_proj(directory, calibration_method='linear', ground_calibration=False):
		""" Returns a numpy array with (index_of_the_image, x_position, y_position, z_position, size) """
		features = pd.read_csv(directory + '/' + _METADATA_PATH, header = None, sep=',')
		features.columns = ["frame_index", "x", "y", "z", "relative_x", "relative_y", "size", "traj"]
		
		camera_features = pd.read_csv(directory + '/' + _CAMERA_DATA, header = None, sep=',')
		camera_features.columns = ["x_c_1", "y_c_1", "z_c_1", 
									"x_c_2", "y_c_2", "z_c_2", 
									"x_c_3", "y_c_3", "z_c_3",
									"x_c_4", "y_c_4", "z_c_4",

									"x_mc_1", "y_mc_1", "z_mc_1", 
									"x_mc_2", "y_mc_2", "z_mc_2", 
									"x_mc_3", "y_mc_3", "z_mc_3",
									"x_mc_4", "y_mc_4", "z_mc_4",

									"x_cb_1", "y_cb_1", "z_cb_1", 
									"x_cb_2", "y_cb_2", "z_cb_2", 
									"x_cb_3", "y_cb_3", "z_cb_3",
									"x_cb_4", "y_cb_4", "z_cb_4",

									"xi_c_1", "yi_c_1",
									"xi_c_2", "yi_c_2", 
									"xi_c_3", "yi_c_3",
									"xi_c_4", "yi_c_4",

									"xi_mc_1", "yi_mc_1",
									"xi_mc_2", "yi_mc_2", 
									"xi_mc_3", "yi_mc_3",
									"xi_mc_4", "yi_mc_4",

									"xi_cb_1", "yi_cb_1",
									"xi_cb_2", "yi_cb_2", 
									"xi_cb_3", "yi_cb_3",
									"xi_cb_4", "yi_cb_4" ]
		if ground_calibration == True:
			camera_features = camera_features[["x_c_1", "y_c_1", "z_c_1", 
									"x_c_2", "y_c_2", "z_c_2", 
									"x_c_3", "y_c_3", "z_c_3",
									"x_c_4", "y_c_4", "z_c_4",

									"x_mc_1", "y_mc_1", "z_mc_1", 
									"x_mc_2", "y_mc_2", "z_mc_2", 
									"x_mc_3", "y_mc_3", "z_mc_3",
									"x_mc_4", "y_mc_4", "z_mc_4",

									"xi_c_1", "yi_c_1",
									"xi_c_2", "yi_c_2", 
									"xi_c_3", "yi_c_3",
									"xi_c_4", "yi_c_4",

									"xi_mc_1", "yi_mc_1",
									"xi_mc_2", "yi_mc_2", 
									"xi_mc_3", "yi_mc_3",
									"xi_mc_4", "yi_mc_4" ]]
			features = Preprocessing.features_transformation_ground(features.values, 
				camera_features.values, calibration_method)
		else:
			features = Preprocessing.features_transformation(features.values, 
				camera_features.values, calibration_method)
		
		return features

	@staticmethod
	def generate_traj_data(features, labels, train_list_complete, splitting_train_list, input_trace):
		""" The method generates the data in order to have the same distribution of sample
			in both train and test set """
		data_x = []
		data_y = []
		for i in range(0, len(splitting_train_list)-1):
			for k in range(0, _K):
				initial_idx = randint(splitting_train_list[i], splitting_train_list[i+1])
				while initial_idx in train_list_complete:
					initial_idx = randint(splitting_train_list[i], splitting_train_list[i+1])

				x = features[initial_idx:initial_idx+input_trace,:]
				y = labels[initial_idx+int(input_trace/2):initial_idx+int(input_trace/2)+1,:]

				data_x.append(x)
				data_y.append(y)

		data_x = np.asarray(data_x)
		data_y = np.asarray(data_y)

		return data_x, data_y

	@staticmethod
	def traj_preprocessing(data_path, number_of_samples, 
		standardize=True, calibration_method=None, noise=None, 
		ground_calibration=False, size_noise=False, constraint_noise=False,
		input_trace=25, input_dim=3):
		""" The method computes the preprocessing needed to fit the data using 
	        the trajectories """
		if data_path == 'flex_traj_fixed_camera' or data_path == 'flex_traj_fixed_camera_bigger':
			data = Preprocessing.import_flex_video_data(_DATASET_PATH + data_path)
		elif data_path == 'flex_traj_moving_camera':
			data = Preprocessing.import_flex_video_data_proj(_DATASET_PATH + data_path)
		
		data_preprocessed, train_list = Preprocessing.filtering_noisy_samples(data, number_of_samples)
	
		train_list_complete = []
		for idx in train_list:
			train_list_complete.append(np.arange(idx-input_trace + 1, idx + 1))
		train_list_complete = np.asarray(train_list_complete).flatten()

		features, labels = Preprocessing.features_labels_split(data_preprocessed)

		if noise is not None:
			if size_noise == True:
				features = Preprocessing.apply_noise_to_size(features, noise[0], noise[1], constraint_noise)
			else:
				features = Preprocessing.apply_noise_to_data(features, noise[0], noise[1], constraint_noise)

		features, labels = Preprocessing.generate_traj_data(features, labels, train_list_complete, train_list, input_trace)

		""" Reshaping for fitting the scaler """
		features = np.reshape(features, (-1, input_trace*input_dim))

		X_train, X_test, y_train, y_test  = Preprocessing.train_test_split(features, labels)

		if standardize == True:
		    X_train, X_test = Preprocessing.standardize_features(X_train, X_test)

		""" Reshaping for fitting the scaler """
		X_train = np.reshape(X_train, (-1, input_trace, input_dim))
		X_test = np.reshape(X_test, (-1, input_trace, input_dim))

		X_train, X_val, y_train, y_val = Preprocessing.train_validation_split(X_train, y_train)

		return X_train, X_test, X_val, y_train, y_test, y_val

	@staticmethod
	def generate_fly_data(features, labels, input_trace):
		""" generates the data for fitting for the flying not flying task """
		data_x = []
		data_y = []
		for i in range(0, len(features)):
			initial_idx = randint(0, len(features) - input_trace - 1)
			x = features[initial_idx:initial_idx+input_trace,:]
			y = labels[initial_idx+int(input_trace/2):initial_idx+int(input_trace/2)+1]

			data_x.append(x)
			data_y.append(y)

		data_x = np.asarray(data_x)
		data_y = np.asarray(data_y)

		return data_x, data_y


	@staticmethod
	def fly_preprocessing(data_path, number_of_samples, 
		standardize=True, calibration_method=None, noise=None, 
		ground_calibration=False, size_noise=False, constraint_noise=False,
		input_trace=25, input_dim=3):
		""" The method computes the preprocessing needed to fit the data using 
	        the trajectories """
		if data_path == 'match_fixed_camera':
			data = Preprocessing.import_video_data(_DATASET_PATH + data_path)
		elif data_path == 'match_moving_camera':
			data = Preprocessing.import_video_data_proj(_DATASET_PATH + data_path)
		
		features, labels = Preprocessing.features_labels_split(data)

		""" Transform the labels into categorical labels considering just the y coordinate """
		labels = Preprocessing.labels_to_binary(labels)

		if noise is not None:
			if size_noise == True:
				features = Preprocessing.apply_noise_to_size(features, noise[0], noise[1], constraint_noise)
			else:
				features = Preprocessing.apply_noise_to_data(features, noise[0], noise[1], constraint_noise)

		features, labels = Preprocessing.generate_fly_data(features, labels, input_trace)

		""" Reshaping the features fot fitting """
		features = np.reshape(features, (-1, input_trace*input_dim))

		X_train, X_test, y_train, y_test  = Preprocessing.train_test_split(features, labels)

		if standardize == True:
			X_train, X_test = Preprocessing.standardize_features(X_train, X_test)

		""" Reshaping for fitting the scaler """
		X_train = np.reshape(X_train, (-1, input_trace, input_dim))
		X_test = np.reshape(X_test, (-1, input_trace, input_dim))

		y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
		y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

		X_train, X_val, y_train, y_val = Preprocessing.train_validation_split(X_train, y_train)

		return X_train, y_train, X_test, y_test, X_val, y_val

	@staticmethod
	def apply_noise_to_size(features, mean, std_dev, constraint_noise):
		""" The method applies a random gaussian noise to each feature 
			in order to simulate a noisy sensor """
		noise = np.random.normal(mean, std_dev, len(features))
		if constraint_noise == True:
			noise[noise > std_dev] = std_dev
			noise[noise < -std_dev] = -std_dev
		features[:,2] = features[:,2] + noise

		return features

	@staticmethod
	def apply_noise_to_data(features, mean, std_dev, constraint_noise):
		""" The method applies a random gaussian noise to each feature 
			in order to simulate a noisy sensor """
		noisy_features = []
		for i in range(0, len(features[0])):
			noise = np.random.normal(mean, std_dev, len(features))
			noisy_feature = features[:,i] + noise
			noisy_features.append(noisy_feature)

		noisy_features = np.asarray(noisy_features).T
		return np.asarray(noisy_features)

	@staticmethod
	def calibration_inference(feature, proj_matrix):
		""" The method computes the Projection of a sample
			from the 2D coordinates to the 3D ones using a 
			given projection matrix """

		F = np.asarray([feature[0], feature[1], 1])
		C = proj_matrix
		C_inv = pinv(np.dot(np.transpose(C), C))
		E = np.dot(np.dot(C_inv, np.transpose(C)), F)

		return E[0:3]

	@staticmethod
	def apply_calibration(cal_method, X_train, X_test):
		""" The method returns the features transformed by the calibration method, 
			the new representation id then passed to the network for the final mapping."""
		
		calibration = CameraCalibrationBuilder.build(cal_method)
		proj_matrix = calibration.compute_camera_calibration()

		cal_X_train = []
		for i, sample in enumerate(X_train):
			cal_X_train.append(Preprocessing.calibration_inference(sample, proj_matrix))

		cal_X_test = []
		for i, sample in enumerate(X_test):
			cal_X_test.append(Preprocessing.calibration_inference(sample, proj_matrix))

		return np.asarray(cal_X_train), np.asarray(cal_X_test)
		
	@staticmethod
	def features_labels_split(video):
		""" The method returns the data splitted into features and labels """
		
		features = video[:,4:]
		labels = video[:, 1:4]
		return features, labels

	@staticmethod
	def train_test_split(features, labels, split_percentage=0.2):
		""" The method returns the data splitted into train and test based on the split percentage """
		X_train, X_test, y_train, y_test = train_test_split(features, labels, 
			test_size=split_percentage, random_state=0, shuffle=True)
		return X_train, X_test, y_train, y_test

	@staticmethod
	def train_validation_split(features, labels, split_percentage=0.1):
		""" The method returns the data splitted into train and test based on the split percentage """
		X_train, X_test, y_train, y_test = train_test_split(features, labels, 
			test_size=split_percentage, random_state=0, shuffle=True)
		return X_train, X_test, y_train, y_test

	@staticmethod
	def train_test_split_single(features, labels, train_list):
		""" The method splits the data using the train list information 
			in order to don't have overlapping trajectories. """
		
		split_train_idx = train_list[int(len(train_list)*0.8)]
		X_train = features[0:split_train_idx,:]
		y_train = labels[0:split_train_idx,:]

		X_test = features[split_train_idx:,:]
		y_test = features[split_train_idx:,:]

		return X_train, X_test, y_train, y_test