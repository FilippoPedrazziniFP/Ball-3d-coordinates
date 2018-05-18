import numpy as np

class Visualizer(object):
	
	@staticmethod
	def get_statistics(X, y):

		print("Features: ")
		print(X.head(5))

		print("Labels: ")
		print(y.head(5))

		return