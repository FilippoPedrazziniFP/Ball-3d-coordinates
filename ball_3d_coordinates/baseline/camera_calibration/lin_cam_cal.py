import cv2
import numpy as np
from numpy.linalg import pinv

from ball_3d_coordinates.baseline.camera_calibration.cam_cal import CameraCalibration

class LinearCameraCalibration(CameraCalibration):
	""" This class contains the methods to compute camera calibration and
		other useful fucntions. """
	def __init__(self):
		super(LinearCameraCalibration, self).__init__()

	def compute_camera_calibration(self, img_points, obj_points):
		""" The method computes the camera calibration using the obj_points and
			the img_points; the formula can be found in the paper:
			Physics-based ball tracking and 3D trajectory reconstruction with 
			applications to shooting location estimation in basketball video.

			We have A * C = B, where 
				- A: transformation matrix
				- C: parameters vector
				- B: 2D points
			To obtain C we compute the pseudo inverse solution: C = (AT * A)-1 * AT * B
			"""
		if img_points is None or obj_points is None:
			img_points = self.get_unity_img_points()
			obj_points = self.get_unity_object_points()

		B = np.asarray(img_points).flatten()
		A = []
		j = 0
		for i in range(0, 2*len(obj_points)):
			if i % 2 == 0:
				row = [obj_points[j][0], obj_points[j][2], obj_points[j][1], 1,  0, 0, 0, 0, 
				-img_points[j][0] * obj_points[j][0], -img_points[j][0] * obj_points[j][2], -img_points[j][0] * obj_points[j][1]]
				A.append(row)
			else: 
				row = [0, 0, 0, 0, obj_points[j][0], obj_points[j][2], obj_points[j][1], 1, 
				-img_points[j][1] * obj_points[j][0], -img_points[j][1] * obj_points[j][2], -img_points[j][1] * obj_points[j][1]]
				A.append(row)
				j = j+1
		""" Transforming A into a numpya array """
		A = np.asarray(A)
		A_inv = pinv(np.dot(np.transpose(A), A))
		C = np.dot(np.dot(A_inv, np.transpose(A)), B)
		C = np.append(C, [1])
		""" Reshaping C from vector to matrix """
		C = np.reshape(C, (3, 4))
		return C