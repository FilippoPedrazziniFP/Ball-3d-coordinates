import numpy as np

class Visualizer(object):
	
	@staticmethod
	def get_statistics(X, y):

		print("Features: ")
		print(X.head(5))

		print("Labels: ")
		print(y.head(5))

		print(X.describe())

		print("Last Features of the Dataset")
		print(X.tail(5))

		return