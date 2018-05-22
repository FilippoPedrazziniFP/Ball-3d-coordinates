### Main file for the baseline training and prediction.

import argparse
import shutil
import numpy as np

from ball_3d_coordinates.baseline.preprocessing.data_preprocessing import BaselinePreprocessor
from ball_3d_coordinates.baseline.preprocessing.data_loader import Loader
from ball_3d_coordinates.baseline.models.parabola_model import ParabolaModel

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--input_trace', type=int, default=25, 
    help='dimension of the trace to take into account.')
parser.add_argument('--noise', type=bool, default=False, 
    help='if True, a noise is applied to the features.')

args = parser.parse_args()

def main():

    # Fix the seed
    np.random.seed(0)
    
    # Load the data
    loader = Loader()
    X, y, camera_parameters = loader.load_data()

    # Preprocess the data
    preprocessor = BaselinePreprocessor(input_trace=args.input_trace)
    X_test, y_test = preprocessor.fit_transform(X, y, noise=args.noise)

    # Define the Model
    model = ParabolaModel(camera_parameters)

    # Test the model
    predictions = []
    for i, sample in enumerate(X_test):
        model.fit(sample, 0)
        pred = model.predict(12)
        predictions.append(pred)

    # Reshaping y_test
    y_test = np.squeeze(y_test)

    mae = model.evaluate(predictions, y_test)
    print("MAE: ", mae)

    return

main()