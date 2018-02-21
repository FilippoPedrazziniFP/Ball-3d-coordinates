from __future__ import absolute_import

import cv2
import numpy as np
from numpy.linalg import pinv

from camera_calibration.cam_cal import CameraCalibration

class UnityCameraCalibration(CameraCalibration):
	""" This class contains the methods to compute camera calibration and
		other useful fucntions. """
	def __init__(self):
		super(UnityCameraCalibration, self).__init__()

	def compute_camera_calibration(self):
		""" The method returns the unity camera calibration, used
			to perfect mapping the 2D world into the 3D one. """
		return

	def get_unity_proj_matrix(self):
		""" The method returns the Projection_matrix from unity """
		proj_matrix = [[0.82585, 0.0, 0.0, 0.0],
						[0.0, 1.40195, 0.0, 0.0],
						[0.0, 0.0, -1.00133, -2.00133],
						[0.0, 0.0, -1.0, 0.0]]
		return proj_matrix

	def get_unity_world_to_camera(self):
		""" The method returns the World_to_Camera matrix from unity """
		cam_to_world = [[1.0, 0.0, 0.0, 0.0], 
						[0.0, 0.86603, 0.5, 40.19],
						[0.0, 0.5, -0.86603, -669.61520],
						[0.0, 0.0, 0.0, 1.0]]
		return cam_to_world

	def compute_intrinsic_camera_mtx(self, img_height, img_width, h_fov, v_fov):
		""" The method returns the intrinsic camera matrix composed by:
			
			c_x: half of the width image
			c_y: half of the height image
			f_x: focal length in x direction
			f_y: focal length in y direction
			
		"""
		c_x = img_width / 2
		c_y = img_height / 2
		f_x = c_x / np.tan(h_fov / 2)
		f_y = c_y / np.tan(v_fov / 2)
		matrix = np.asarray([

			[f_x, 0, c_x],
			[0, f_y, c_y],
			[0, 0, 1]
			])
		return matrix