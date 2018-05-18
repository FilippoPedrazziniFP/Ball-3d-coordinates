import cv2
import numpy as np
from numpy.linalg import pinv

from ball_3d_coordinates.baseline.camera_calibration.cam_cal import CameraCalibration

class OpenCVCameraCalibration(CameraCalibration):
	""" This class contains the methods to compute camera calibration and
		other useful fucntions. """
	def __init__(self):
		super(OpenCVCameraCalibration, self).__init__()

	def compute_camera_calibration(self, img_points, obj_points):
		""" The method computes the camera calibration and returns the projection matrix
			in order to have the mapping between the image coordinates and the the 
			real world """

		img_points = self.get_unity_img_points()
		obj_points = self.get_unity_object_points()
		ret, self.camera_matrix, dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
			[obj_points], [img_points], (1190, 701), None, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
		
		""" With this information you can compute the transformation matrix 
			
			C = MTX * [R | T] which maps an object in the real world (X, Y, Z) into a position in the 2D image.

		"""
		C = np.zeros((3, 3))
		np.fill_diagonal(C, self.rvecs)
		self.tvecs = np.reshape(np.asarray(self.tvecs), (3, 1))
		C = np.concatenate((C, self.tvecs), axis=1)

		""" Using the intrinsic parameters """
		""" C = np.dot(mtx, C) """

		return C