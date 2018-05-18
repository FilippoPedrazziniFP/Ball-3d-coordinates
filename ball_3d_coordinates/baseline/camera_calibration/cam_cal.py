import cv2
import numpy as np
from numpy.linalg import pinv

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
						[1002.785, 549.1671],
						[-541.6168, 30.22664],
						[1901.617, 30.22664],
						[357.2146, 549.1671],

						[169.3557, 440.7068],
						[1190.644, 440.7068],
						[680.0, 549.1671],
						[680.0, 30.22664], 

						[212.3757, 510.5415],
						[67.22684, 440.7068],
						[1147.624, 510.5415],
						[1292.773, 440.7068] 

						]
		return np.asarray(img_points).astype('float32')
		



	