from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import backend as K

_HIDDEN_SIZE = 256
_FILTERS = 64
_CLASSES = 2

class SSDKeras(object):
	""" The model tries to predict the 2D position of the ball in the image. """
	def __init__(self, 
			img_height=None,
			img_width=None,
			input_channels=None,
			output_dimension=2,
			learning_rate=2e-5,
			learning_rate_decay=1e-10):
		super(SSDKeras, self).__init__()
		self.img_height = img_height
		self.img_width = img_width
		self.input_channels = input_channels
		self.output_dimension = output_dimension
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay

	@staticmethod
	def root_mean_squared_error(y_true, y_pred):
		return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

	@staticmethod
	def mean_iou(y_true, y_pred):
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred, _CLASSES)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
		   score = tf.identity(score)
		return score

	@staticmethod
	def iou(r1, r2):
		
	    x1, y1, w1, h1 = r1
	    x2, y2, w2, h2 = r2
	    and_x1, and_y1 = max(x1, x2), max(y1, y2)
	    and_x2, and_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
	    and_w = and_x2 - and_x1
	    and_h = and_y2 - and_y1
	    if and_w <= 0 or and_h <= 0:
	    	return 0
	    and_area = and_w * and_h
	    area1 = w1 * h1
	    area2 = w2 * h2
	    or_area = area1 + area2 - and_area
	 
	    return and_area / or_area

	@staticmethod
	def iou(y_true, y_pred):
		""" -ln(intersection/union) """
		return

	def build_vgg_net(self):

		self.conv_base = VGG16(
				weights='imagenet',
				include_top=False,
				input_shape=(self.img_height, self.img_width, self.input_channels)
				)
		self.conv_base.summary()
	
		net = Sequential()
		net.add(self.conv_base)
		
		""" Shrinking the shape to one """
		net.add(GlobalMaxPooling2D())
		
		net.add(Flatten())
		net.add(Dense(256, activation='relu'))
		net.add(Dense(self.output_dimension))

		""" Putting the VGG Net weigths fixed """
		self.conv_base.trainable = False

		net.summary()

		net.compile(
			optimizer=SGD(lr=self.learning_rate, decay=self.learning_rate_decay),
			loss=mean_squared_error, 
			metrics=['mae'])

		return net

	def fine_tuning(self, net):

		""" Putting the VGG Net weigths fixed """
		set_trainable = False
		for layer in net.layers:
			if layer.name == 'block1_conv1':
				set_trainable = True
			if set_trainable:
				layer.trainable = True
			else:
				layer.trainable = False
		
		net.summary()
		
		net.compile(
			optimizer=SGD(lr=self.learning_rate, decay=self.learning_rate_decay),
			loss=mean_squared_error, 
			metrics=['mae'])

		return net

	def build_smaller_vgg_net(self):

		self.conv_base = VGG16(
				weights='imagenet',
				include_top=False,
				input_shape=(self.img_height, self.img_width, self.input_channels)
				)
		self.conv_base.summary()

		# Block 2
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()

		# Block 3
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()

		# Block 4
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		
		# Block 5
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		self.conv_base.layers.pop()
		
		self.conv_base.layers.pop()

		self.conv_base.summary()

		x = GlobalMaxPooling2D()(self.conv_base.layers[-1].output)
		x = Flatten()(x)
		x = Dense(256, activation='relu')(x)
		x = Dense(self.output_dimension)(x)
		
		net = Model(inputs=self.conv_base.input, outputs=x)

		""" Putting the VGG Net weigths fixed """
		set_trainable = False
		for layer in net.layers:
			if layer.name == 'flatten_1':
				set_trainable = True
			if set_trainable:
				layer.trainable = True
			else:
				layer.trainable = False

		net.summary()
		
		net.compile(
			optimizer=SGD(lr=self.learning_rate, decay=self.learning_rate_decay),
			loss=mean_squared_error, 
			metrics=['mae'])
		
		return net
	
	def build_net(self):
		""" If the network is so small and it can learn without problems,
			it means that I can make the ball smaller in the Unity world """
		net = Sequential()

		net.add(Conv2D(32, (5, 5), activation='relu', strides=(2,2),
			input_shape=(self.img_height, self.img_width, self.input_channels)))
		net.add(Conv2D(64, (5, 5), activation='relu', strides=(2,2)))
		net.add(Conv2D(128, (5, 5), activation='relu', strides=(2,2)))	
		
		net.add(GlobalMaxPooling2D())	
		
		net.add(Flatten())
		net.add(Dense(128, activation='relu'))
		net.add(Dense(self.output_dimension))

		net.summary()

		net.compile(
			optimizer=SGD(lr=self.learning_rate, decay=self.learning_rate_decay),
			loss=mean_squared_error, 
			metrics=['mae'])

		return net
