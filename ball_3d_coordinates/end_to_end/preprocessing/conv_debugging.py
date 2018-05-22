import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

import ball_3d_coordinates.util.util as util

class ConvDebugger(object):
	""" The class contains all the methods necessary to 
		debug convolutions showing the filters that they learn
		and the transformed images."""
	def __init__(self, net):
		super(ConvDebugger, self).__init__()
		self.net = net

	def fit(self, features, labels):
		
		layer_outputs = [layer.output for layer in self.net.layers[:7]]
		activation_model = Model(inputs=self.net.input, outputs=layer_outputs)
		sample = np.reshape(features[0], (1, util.IMG_HEIGHT, util.IMG_WIDTH, util.INPUT_CHANNELS))
		activations = activation_model.predict(sample)

		# Show a single activated image from the first layer
		self.show_one_activated_image(features[0], activations)
		
		return

	def show_one_activated_image(self, img, activations):
		
		first_layer_activation = activations[1]
		plt.imshow(img)
		plt.show()
		plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
		plt.show()
		
		return

	def check_what_filters_learn(self):
		""" Apply gradient descent to understand what each filter is able to learn """
		return

	def CAM(self, img):
		
		pred = self.net.predict(img).flatten()
		width = int(pred[0]*util.IMG_WIDTH)
		height = int(abs(pred[1]*util.IMG_HEIGHT - util.IMG_HEIGHT))
		print('PREDICTION: (%s, %s)' %(width, height))

		x_out = self.net.output[:, 0]
		y_out = self.net.output[:, 1]

		last_conv_layer = self.net.get_layer('conv2d_6')
		grads = K.gradients(x_out, last_conv_layer.output)[0]
		pooled_grads = K.mean(grads, axis=(0, 1, 2))
		iterate = K.function([self.net.input], [pooled_grads, last_conv_layer.output[0]])

		pooled_grads_value, conv_layer_output_value = iterate([img])
		for i in range(256):
			conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
		heatmap = np.mean(conv_layer_output_value, axis=-1)
		heatmap = np.maximum(heatmap, 0)
		heatmap /= np.max(heatmap)
		
		plt.figure(figsize = (10, 8))
		plt.matshow(heatmap)
		plt.show()
		
		img = np.array(img * 255, dtype = np.uint8)
		img = np.reshape(img, (util.IMG_HEIGHT, util.IMG_WIDTH, util.INPUT_CHANNELS))
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		print(img.shape)
		plt.figure(figsize = (10, 8))
		plt.imshow(img)
		plt.show()
		
		heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
		heatmap = np.uint8(255 * heatmap)
		heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
		superimposed_img = heatmap * 0.8 + img
		superimposed_img = cv2.circle(superimposed_img, (width, height), 6, (255, 0, 0), -1)
		plt.figure(figsize = (10, 8))
		plt.imshow(superimposed_img)
		plt.show()
		
		return

	def show_n_layer_activated_images(self, activations, layer_number):
		
		layer_activation = activations[layer_number]
		n_features = layer_activation.shape[-1]
		for i in range(0, n_features):
			plt.matshow(layer_activation[0, :, :, i], cmap='viridis')
			plt.show()
		
		return

	def show_all_layers_activated_image(self, activations):		
		
		layer_names = []
		for layer in self.net.layers[:8]:
			layer_names.append(layer.name)

		images_per_row = 16
		for layer_name, layer_activation in zip(layer_names, activations):
			n_features = layer_activation.shape[-1]
			height = layer_activation.shape[1]
			width = layer_activation.shape[2]
			n_cols = n_features // images_per_row
			display_grid = np.zeros((height * n_cols, images_per_row * width))

			for col in range(n_cols):
				for row in range(images_per_row):
				    channel_image = layer_activation[0, :, :, col * images_per_row + row]
				    channel_image -= channel_image.mean()
				    channel_image /= channel_image.std()
				    channel_image *= 64
				    channel_image += 128
				    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
				    display_grid[col * height : (col + 1) * height, row * width : (row + 1) * width] = channel_image

			h_scale = 1. / height
			w_scale = 1. / width
			plt.figure(figsize=(h_scale * display_grid.shape[1], w_scale * display_grid.shape[0]))
			plt.title(layer_name)
			plt.grid(False)
			plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
		plt.show()
		
		return

	def check_predictions(self, features, labels):
		
		for x, y in zip(features, labels):
			img_tmp = np.reshape(x, (1, util.IMG_HEIGHT, util.IMG_WIDTH, util.INPUT_CHANNELS))
			prediction = self.net.predict(img_tmp).flatten()
			width = int(prediction[0]*util.IMG_WIDTH)
			height_ = int(prediction[1]*util.IMG_HEIGHT)
			label = y.flatten()
			print("PREDICTION: (%s, %s)" %(width, height_))
			height = abs(height_ - util.IMG_HEIGHT)
			print("LABEL: (%s, %s)" %(int(label[0]*util.IMG_WIDTH), int(label[1]*util.IMG_HEIGHT)))
			original_img = x*255.0
			img = cv2.circle(original_img, (width, height), 6, (255, 0, 0), -1)
			cv2.imwrite('test.png', img)
			break
		
		return
		