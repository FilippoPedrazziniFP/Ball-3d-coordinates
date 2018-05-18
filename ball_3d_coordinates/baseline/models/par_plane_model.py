from __future__ import absolute_import

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv

from camera_calibration.cam_cal_dic import CameraCalibrationBuilder

class ParabolaPlaneModel(object):
    """ The class contains all the necessary methods to 
        make predictions based on the 
        trajectory equations and the camera position 
        (extracted using camera calibration).
        """
    def __init__(self, 
            cam_cal_method='linear',
            input_trace=None,
            output_trace=None,
            input_dimension=2,
            output_dimension=3,
            learning_rate_decay=None):
        super(ParabolaPlaneModel, self).__init__()
        self.calibration = CameraCalibrationBuilder.build(cam_cal_method)
        self.proj_matrix = self.calibration.compute_camera_calibration()
        self.camera_position = self.compute_camera_position()
        self.g = -9.8
        self.lr = 0.1
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        """ The method initializes the 6 parameters needed 
            for the training task. """
        return np.random.uniform(0.5, 1, 5)

    def loss_function(self, Xs, Ys, Xs_1, Ys_1, Xc, Yc, Zc, parameters):
        """ The method computes the loss function as described in the paper 
            The loss is composed by two terms: the error for fitting 
            the plane and the error for fitting the displacement """

        """ First error """
        T = (parameters[3] - parameters[4]*Xs - Ys) / ((Yc - Ys) + parameters[4]*(Xc - Xs))
        H = np.array((Xc, -Xs*Yc, -Ys*Zc))
        L = np.array((Xs, Ys, 0))
        pred = np.transpose(H) * T + np.transpose(L)
        F = parameters[0]*(pred[0]**2) + parameters[1]*(pred[0]**2) + pred[0]
        err_1 = F - pred[1]

        """ Second error """
        T = (parameters[3] - parameters[4]*Xs_1 - Ys_1) / ((Yc - Ys_1) + parameters[4]*(Xc - Xs_1))
        H = np.array((Xc, -Xs_1*Yc, -Ys_1*Zc))
        L = np.array((Xs_1, Ys_1, 0))
        pred_1 = np.transpose(H) * T + np.transpose(L)
        err_2 = pred_1[0] - pred[0]

        err = err_1 + err_2
        return err

    def compute_camera_position(self):
        """ The method computes the physical position of the camera
            using the information regarding the projection matrix. 
            We extract the intrinsic camera matrix (K), 
            the rotation matrix (R) and the translation matrix (T); """

        """ Another more efficient method, in this case, should be to 
            create R, T starting from an already known K """

        """ Computing K """
        B = self.proj_matrix[:,0:3]
        b = self.proj_matrix[:,3]
        
        K_tmp = np.dot(B, B.T)
        K_norm = K_tmp/K_tmp[2][2]
        K = np.zeros((3,3))
        K[2][2] = 1.0
        K[0][2] = K_norm[0][2]
        K[1][2] = K_norm[1][2]
        K[1][1] = np.sqrt(K_norm[1][1] - np.power(K[1][2], 2))
        K[0][1] = (K_norm[1][0] - K[0][2]*K[1][2])/K[1][1]
        K[0][0] = np.sqrt(K_norm[0][0] - np.power(K_norm[0][2], 2) - np.power(K[0][1], 2))

        """ From K now is easy to compute R and T """
        R = np.dot(inv(K), B)
        T = np.dot(inv(K), b)

        camera_position = -np.dot(inv(R), T)

        return camera_position

    def derivative(self, loss_function, Xs, Ys, Xs_1, Ys_2, Xc, Yc, Zc, parameters, delta=10e-6):
        """ The method computes the derivative of the loss function with 
            respect to the passed parameters. """
        params = []
        for i in range(0, len(parameters)):
            tmp = np.zeros((len(parameters)))
            tmp[i] = delta
            result = (loss_function(Xs, Ys, Xs_1, Ys_2, Xc, Yc, Zc, parameters + tmp) - loss_function(Xs, Ys, Xs_1, Ys_2, Xc, Yc, Zc, parameters - tmp)) / (2*delta)
            params.append(result)
        return np.asarray(params)

    def train(self, features):
        """ In this method a training step consists in the
            computation of the plane which fits more the 
            parabola equation. """

        """ We have the matrix X of (X^2, X, 1) points in the trajectory
            which corresponds to the 2D position in the image.
            A is a vector corresponding our parameters (a, b, c) for 
            the trajectoy equation, and Z is a vector with the corresponding
            Z points for each observation. """

        """ The pseudo inverse solution is:

            A = (X.T * X)-1 * X.T * Z """

        C = self.get_projection_matrix()

        for i in range(0, len(features)-1):
            derivatives = self.derivative(self.loss_function, features[i][0], 
                features[i][1], features[i+1][0], features[i+1][1], self.camera_position[0], 
                self.camera_position[1], self.camera_position[2], self.parameters)
            self.parameters -= self.lr * derivatives

        self.a = self.parameters[0]
        self.b = self.parameters[1]
        self.c = self.parameters[2]
        self.d = self.parameters[3]
        self.k = self.parameters[4]

        return

    def inference(self, feature):
        """ The inference is done computing the 3D position of the ball using
            the 3D coordinates and the best plane found. """

        T = (self.d - self.k*feature[0] - feature[1]) / ((self.camera_position[1] - feature[1]) + 
            self.k*(self.camera_position[0] - feature[1]))
        H = np.array((self.camera_position[0], -feature[0]*self.camera_position[1], 
            -feature[1]*self.camera_position[2]))
        L = np.array((feature[0], feature[1], 0))

        pred = np.transpose(H) * T + np.transpose(L)
        x = pred[0]
        y = self.a*(x**2) + self.b*x + self.c
        z = pred[2]

        return np.array((x, y, z))

    def get_projection_matrix(self):
        """ The method returns the projection matrix """
        return self.proj_matrix

    def accuracy(self, out, labels):
        """ The method computes the RMSE between prediction and labels """
        rmse = np.sqrt(((out - labels) ** 2).mean())
        mse = ((out - labels) ** 2).mean()
        return rmse