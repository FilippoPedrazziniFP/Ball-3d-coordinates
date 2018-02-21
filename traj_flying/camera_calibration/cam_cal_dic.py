from __future__ import absolute_import

from camera_calibration.ffnn_cam_cal import FFNNCameraCalibration
from camera_calibration.lin_cam_cal import LinearCameraCalibration
from camera_calibration.opencv_cam_cal import OpenCVCameraCalibration
from camera_calibration.cam_cal_unity import UnityCameraCalibration

_METHODS = {

			'ffnn' : FFNNCameraCalibration(),
			'linear' : LinearCameraCalibration(),
			'opencv' : OpenCVCameraCalibration()

			}

class CameraCalibrationBuilder(object):
	"""docstring for CameraCalibrationBuilder"""
	def __init__(self):
		super(CameraCalibrationBuilder, self).__init__()

	@staticmethod
	def build(method):
		return type(_METHODS[method])()
