import argparse
import shutil
import numpy as np

from ball_3d_coordinates.traj_flying_experiments.preprocessing.data_preprocessing_traj import TmpTrajPreprocessor
from ball_3d_coordinates.traj_flying_experiments.preprocessing.exploration import Visualizer
from ball_3d_coordinates.traj_flying_experiments.preprocessing.data_loader_traj import Loader
from ball_3d_coordinates.traj_flying_experiments.models.tmp_net_traj import TmpTrajNet
from ball_3d_coordinates.traj_flying_experiments.models.lstm_traj import LSTMTrajNet

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--restore', type=bool, default=True, 
    help='if True restore the model from --model_path.')
parser.add_argument('--train', type=bool, default=True, 
    help='if True train the model.')
parser.add_argument('--tune', type=bool, default=False, 
    help='if True tune the model.')
parser.add_argument('--test', type=bool, default=True, 
    help='if True test the model.')

""" Model Parameters """
parser.add_argument('--log_dir', type=str, default='./tensorbaord', 
    help='directory where to store tensorbaord values.')
parser.add_argument('--epochs', type=int, default=1000, 
    help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=100, 
    help='number of samples in the training batch.')
parser.add_argument('--input_trace', type=int, default=25, 
    help='dimension of the trace to take into account.')
parser.add_argument('--noise', type=bool, default=False, 
    help='if True, a noise is applied to the features.')

args = parser.parse_args()

def main():

    # Remove tensorboard folder.
    try:
        shutil.rmtree('./tensorbaord')
    except FileNotFoundError:
        pass

    # Fix the seed
    np.random.seed(0)
    
    # Load the data
    loader = Loader()
    X, y = loader.load_data()

    # Explore data
    Visualizer.get_statistics(X, y)

    # Preprocess the data
    preprocessor = TmpTrajPreprocessor(input_trace=args.input_trace)
    X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.fit_transform(X, y, noise=args.noise)

    # Define the Model
    model = LSTMTrajNet(
        input_trace=args.input_trace,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_dir=args.log_dir
        )

    # Restore the model
    if args.restore == True:
        model.restore()

    # Train the model
    if args.train == True:
        print("Starting training...")
        history = model.fit(X_train, y_train, X_test, y_test)
        print("Finished training...")

    # Tune the model
    if args.tune == True:
        print("Starting tuning...")
        model.tune(X_train, y_train)
        print("Finished tuning...")

    # Test the model
    if args.test == True:
        print("Starting testing...")
        loss, metric = model.evaluate(X_test, y_test)
        print("Finished testing...")
        print("LOSS: ", loss)
        print("MAE: ", metric)

    return

main()