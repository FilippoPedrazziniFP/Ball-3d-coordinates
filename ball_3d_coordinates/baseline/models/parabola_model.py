from __future__ import absolute_import

import numpy as np
from numpy.linalg import pinv

from ball_3d_coordinates.baseline.camera_calibration.lin_cam_cal import LinearCameraCalibration

class ParabolaModel(object):
    """ The class contains all the necessary methods to make predictions based on the 
        trajectory equations and camera calibration. 

        Problem: we are not taking care of the acelleration 
        (the force that the player gives to the ball). 
        In the paper they use the 2D trajectory to estimate the 
        max_vertical_speed of the ball (no more real time).

        """
    def __init__(self, camera_parameters=None):
        super(ParabolaModel, self).__init__()
        self.obj_points, self.img_points = self.split_camera_parameters(camera_parameters)
        self.calibration = LinearCameraCalibration()
        self.proj_matrix = self.calibration.compute_camera_calibration(self.img_points, self.obj_points)
        self.g = -9.8

    def split_camera_parameters(self, camera_parameters):
        if camera_parameters is None:
            return None, None
        else:
            # Object Points list
            obj_points = []
            num_of_points = int(len(camera_parameters)/5)
            for i in range(0, num_of_points*3, 3):
                point = [camera_parameters[i], camera_parameters[i+1], camera_parameters[i+2]]
                obj_points.append(point)

            # Img Points list
            img_points = []
            for i in range(num_of_points*3, num_of_points*3 + num_of_points*2, 2):
                point = [camera_parameters[i], camera_parameters[i+1]]
                img_points.append(point)
        return obj_points, img_points

    def fit(self, features, time):
        """ The method initializes the parameters of the parabola. For each video,
            the method must be called. 

            Having the projection matrix (C), the parabola model equations (E) and the object 
            position in the 2D image at time zero (feature), we can estimate the 
            missing values (X0, Vx0, Y0, Vy0, Z0, Vz0) in a system of equations (Eq 19 paper) 
            using the pseudo inverse solution. """
        features = list(features)
        C = self.get_projection_matrix()
        D = []
        F = []
        T = time/25
        for i, feature in enumerate(features):

            X = feature[0]
            Y = feature[1]

            row_1 = [C[0][0] - X*C[2][0], C[0][0]*T - X*C[2][0]*T, C[0][1] - X*C[2][1], C[0][1]*T - X*C[2][1]*T, C[0][2] - X*C[2][2], C[0][2]*T - X*C[2][2]*T]
            row_2 = [C[1][0] - Y*C[2][0], C[1][0]*T - Y*C[2][0]*T, C[1][1] - Y*C[2][1], C[1][1]*T - Y*C[2][0]*T, C[1][2] - Y*C[2][2], C[1][2]*T - Y*C[2][2]*T]
                   
            D.append(row_1)
            D.append(row_2)

            row_1 = [X*(C[2][2]*self.g*(T**2)/2 + 1) - (C[0][2]*self.g*(T**2)/2 + C[0][3])]
            row_2 = [Y*(C[2][2]*self.g*(T**2)/2 + 1) - (C[1][2]*self.g*(T**2)/2 + C[1][3])]

            T = T + 1/25

            F.append(row_1)
            F.append(row_2)

        D = np.asarray(D)
        F = np.asarray(F)

        D_inv = pinv(np.dot(np.transpose(D), D))
        E = np.dot(np.dot(D_inv, np.transpose(D)), F)

        E = np.squeeze(E)
        self.initial_x = E[0]
        self.v_x = E[1]
        self.initial_y = E[4]
        self.v_y = E[5]
        self.initial_z = E[2]
        self.v_z = E[3]
        return

    def predict(self, t):
        """ The method returns the prediction using the
            parabola model equations. 
            
            Assuming that 1 unit in Unity represents 1 Meter in the real world
            we have that the gravity 9.8 m/s and the time, 
            due to the fact that we are extracting 25 FPS, 
            in this case is t/25 """

        t = t/25

        x = self.initial_x + self.v_x * t
        y = self.initial_y + self.v_y * t + 0.5 * self.g * (t ** 2)
        z = self.initial_z + self.v_z * t
        print(x)
        out = [x, y, z]
        return out

    def get_projection_matrix(self):
        """ The method returns the projection matrix """
        return self.proj_matrix

    def evaluate(self, X, y):
        """ The method computes the MAE between the predictions and the labels """
        mae = np.sum(np.absolute((np.array(X).astype("float") - np.array(y).astype("float"))))
        return mae