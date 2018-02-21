from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.keras.layers import TimeDistributed, Activation, Dense, Conv1D, Dropout, Conv2D, Reshape, Input, Add
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import losses
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adagrad
from tensorflow.python.keras import backend as K

_HIDDEN_SIZE = 100
_FILTERS_2_LAYERS = 40
_FILTERS_1_LAYER = 100

class ConvTemp(object):
	""" This simple model tries to map the 2D input into a 3D representation. """
	def __init__(self, 
			input_trace=25,
			output_trace=1,
			input_dimension=3,
			output_dimension=3,
			learning_rate=0.1,
			learning_rate_decay=1e-8):
		super(ConvTemp, self).__init__()
		self.input_trace = input_trace
		self.output_trace = output_trace
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay

	@staticmethod
	def root_mean_squared_error(y_true, y_pred):
		return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

	def build_net(self):	

		net = models.Sequential()
		
		net.add(Conv1D(filters=_FILTERS, kernel_size=15, padding="valid", 
			input_shape=(self.input_trace, self.input_dimension)))
		net.add(Activation(K.relu))

		net.add(Conv1D(filters=_FILTERS/2, kernel_size=11, padding="valid", 
			input_shape=(self.input_trace, self.input_dimension)))
		net.add(Activation(K.relu))

		net.add(TimeDistributed(Dense(self.output_dimension)))

		net.summary()

		net.compile(
			optimizer=SGD(lr=self.learning_rate, decay=self.learning_rate_decay),
			loss=ConvTemp.root_mean_squared_error, 
			metrics=['mae'])

		return net