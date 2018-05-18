#### Main file for the training/predcition of the Image net

import argparse
import shutil
import numpy as np

from ball_3d_coordinates.obj_detection.preprocessing.data_preprocessing import ConvPreprocessor
from ball_3d_coordinates.obj_detection.preprocessing.data_loader import Loader
from ball_3d_coordinates.obj_detection.model.conv_net import ConvNet

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--debug', type=bool, default=True, 
    help='if True debug the model.')
parser.add_argument('--restore', type=bool, default=True, 
    help='if True restore the model from --model_path.')
parser.add_argument('--test', type=bool, default=False, 
    help='if True test the model.')
parser.add_argument('--train', type=bool, default=False, 
    help='if True train the model.')
parser.add_argument('--tune', type=bool, default=False, 
    help='if True tune the model.')
parser.add_argument('--create_df', type=bool, default=False, 
    help='if True creates a dataframe with the prediction of the model.')

""" Model Parametes """
parser.add_argument('--log_dir', type=str, default='./tensorbaord', 
    help='directory where to store tensorbaord values.')
parser.add_argument('--model_path', type=str, default='./ball_3d_coordinates/obj_detection/weights/img_net', 
    help='model checkpoints directory.')
parser.add_argument('--epochs', type=int, default=1000, 
    help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=32, 
    help='number of samples in the training batch.')
parser.add_argument('--number_of_samples', type=int, default=50, 
    help='how many frames you want to load for the prediction using the convnet.')
parser.add_argument('--number_of_samples_train', type=int, default=50, 
    help='how many frames you want to use for training.')

args = parser.parse_args()

def main():

    # Remove Tensorboard Folder
    try:
        shutil.rmtree('./tensorbaord')
    except FileNotFoundError:
        pass
    
    # Fix the seed
    np.random.seed(0)

    # Load the data
    loader = Loader(number_of_samples=args.number_of_samples)
    X, y = loader.load_data()
    print("Loaded the data...")

    # Preprocess the data
    preprocessor = ConvPreprocessor(number_of_samples_train=args.number_of_samples_train)
    X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.fit_transform(X, y)

    # Define the Model
    model = ConvNet(
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_dir=args.log_dir,
        model_path=args.model_path
        )

    # Restore the model
    if args.restore == True:
        model.restore()

    # Train the model
    if args.train == True:
        history = model.fit(X_train, y_train, X_test, y_test)

    # Tune the model
    if args.tune == True:
        model.tune(X_train, y_train)

    # Test the model
    if args.test == True:
        model.evaluate(X_test, y_test)

    # Debug the model
    if args.debug == True:
        model.debug(X_test, y_test)

    # Create the DF for the next model
    if args.create_df == True:
        model.create_df(X)
    
    return

main()