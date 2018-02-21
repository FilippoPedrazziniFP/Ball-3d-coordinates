from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.keras.layers import TimeDistributed, Activation, Dense, Conv1D, Dropout, Conv2D, Reshape, Input, Add, Flatten
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import losses
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adagrad
from tensorflow.python.keras import backend as K

_HIDDEN_SIZE = 100
_FILTERS = 200

class FNNTempClass(object):
	""" This simple model tries to map the 2D input into a 3D representation. """
	def __init__(self, 
			input_trace=25,
			output_trace=25,
			input_dimension=3,
			output_dimension=1,
			learning_rate=0.1,
			learning_rate_decay=1e-8):
		super(FNNTempClass, self).__init__()
		self.input_trace = input_trace
		self.output_trace = output_trace
		self.input_dimension = input_dimension
		self.output_dimension = output_dimension
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay

	def build_net(self):	

		net = models.Sequential()

		net.add(Dense(_HIDDEN_SIZE, input_dim=self.input_trace*seòf.input_dimension))
		net.add(Activation(K.relu))

		net.add(Dense(_HIDDEN_SIZE))
		net.add(Activation(K.relu))

		net.add(Dense(self.output_dimension, activation='sigmoid'))

		net.summary()

		net.compile(
			optimizer=SGD(lr=self.learning_rate, decay=self.learning_rate_decay, momentum=0.3, nesterov=True), 
			loss='binary_crossentropy', 
			metrics=['accuracy'])

		return net