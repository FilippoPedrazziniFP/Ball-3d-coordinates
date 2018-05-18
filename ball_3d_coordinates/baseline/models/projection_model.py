from __future__ import absolute_import

import numpy as np
from numpy.linalg import pinv

from camera_calibration.cam_cal_dic import CameraCalibrationBuilder

class ProjectionModel(object):
    """ The class corresponds to the model in which we use just the projection
        matrix to estimate the points """
    def __init__(self, 
            cam_cal_method='linear', 
            input_trace=None,
            output_trace=None,
            input_dimension=2,
            output_dimension=3,
            learning_rate_decay=None):
        super(ProjectionModel, self).__init__()
        self.calibration = CameraCalibrationBuilder.build(cam_cal_method)
        self.proj_matrix = self.calibration.compute_camera_calibration()

    def inference(self, feature):
        """ The projection model uses the simply 
            the projection matrix computing 
            the pseudo inverse. """
        F = np.asarray([feature[0], feature[1], 1])
        C = self.proj_matrix
        C_inv = pinv(np.dot(np.transpose(C), C))
        E = np.dot(np.dot(C_inv, np.transpose(C)), F)

        return E[0:3]

    def accuracy(self, out, labels):
        """ The method computes the RMSE between prediction and labels """
        rmse = np.sqrt(((out - labels) ** 2).mean())
        mse = ((out - labels) ** 2).mean()
        return rmse
