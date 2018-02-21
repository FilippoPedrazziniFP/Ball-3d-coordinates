from __future__ import absolute_import

import cv2
import numpy as np
from numpy.linalg import pinv

""" Put all the constants in a file """


class CameraCalibration(object):
	""" This class contains the methods to compute camera calibration and
		other useful fucntions. """
	def __init__(self):
		super(CameraCalibration, self).__init__()

	def get_rotation_matrix(self):
		""" The method returns the 3X3 rotation matrix;
			which corresponds to the camera rotation in
			the 3 coordinates. """
		return self.rvecs

	def get_translation_matrix(self):
		""" The method returns the 3X1 translation matrix;
			which corresponds to the camera translation in
			the 3 coordinates """
		return self.tvecs

	def get_intrinsic_camera_matrix(self):
		""" The method returns the intrinsic camera matrix, 
		which is a 3X3 matrix with the characteristics of 
		the camera """
		return self.camera_matrix

	def get_unity_object_points(self):
		""" The method returns the points gotten from unity
			which represents the corners court """
		obj_points = [	
						[600.0, 0.0, 450.0],
						[-600.0, 0.0, -450.0],
						[600.0, 0.0, -450.0],
						[-600.0, 0.0, 450.0],
						
						[-600.0, 0.0, 0.0],
						[600.0, 0.0, 0.0],
						[0.0, 0.0, 450.0],
						[0.0, 0.0, -450.0],
						
						[-600.0, 50.0, 100.0],
						[-600.0, 50.0, -100.0],
						[600.0, 50.0, 100.0],
						[600.0, 50.0, -100.0]

						]
		return np.asarray(obj_points).astype('float32')

	def get_unity_img_points(self):
		""" The method returns the points gotten from unity 
		using the world to camera method """
		img_points = [
						[873.3, 473.5],
						[-458.3, 26.1],
						[1648.3, 26.1],
						[316.7, 473.5],

						[154.7, 380.0],
						[1035.3, 380.0],
						[595.0, 473.5],
						[595.0, 26.1], 

						[191.8, 440.2],
						[66.6, 380.0],
						[998.2, 440.2],
						[1123.4, 380.0] 

						]
		return np.asarray(img_points).astype('float32')
		



	