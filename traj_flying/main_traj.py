import numpy as np
import sys
import tensorflow as tf
import argparse
import cv2
import os
import time
from numpy.linalg import pinv
import shutil

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras import backend as K

from preprocessing.data_preprocessing import Preprocessing
from models.fnn_tmp_model import ConvTemp
from models.lstm_tmp_model import LSTMTemp

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--model_path', type=str, default='./model/conv_2_layer_fixed', help='model checkpoints directory.')
parser.add_argument('--restore', type=bool, default=False, help='if True restore the model from --model_path.')
parser.add_argument('--log_dir', type=str, default='./tensorbaord', help='directory where to store tensorbaord values.')

""" Dataset Parameters """
parser.add_argument('--dataset', type=str, default='flex_traj_fixed_camera_bigger', choices=['flex_traj_fixed_camera','flex_traj_moving_camera', 'flex_traj_fixed_camera_bigger'], help='type of dataset that you want to use.')
parser.add_argument('--noise', type=float, default=(0, 2.5), nargs=2, help='if you want to add a gaussian noise to the data.')
parser.add_argument('--size_noise', type=bool, default=True, help='if true applies noise just to the size of the ball.')
parser.add_argument('--constraint_noise', type=bool, default=True, help='if true the noise is constraint to the values in the tuple.')

""" Preprocessing steps """
parser.add_argument('--standardize_features', type=bool, default=True, help='if True, applies a standard scaler to the features.')
parser.add_argument('--input_dimension', type=int, default=3, help='input dimension for feeding the network (in case of apply_calibration==True, must be set to 3.')
parser.add_argument('--output_dimension', type=int, default=3, help='output dimension of the network.')
parser.add_argument('--calibration_method', type=str, default=None, choices=['linear', 'opencv', 'ffnn'], help='Which method to use to perform the camera calibration.')
parser.add_argument('--ground_calibration', type=bool, default=True, help='If you want to simulate the standard ground calibration.')

""" How much you want to use the temporal information """
parser.add_argument('--input_trace', type=int, default=25, help='how many frames used for training.')
parser.add_argument('--output_trace', type=int, default=1, help='how many frames used as labels.')

""" Model parameters """
parser.add_argument('--epochs', type=int, default=1000, help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for the training (number of traces to take).')
parser.add_argument('--learning_rate_decay', type=float, default=0.0, help='how to decay the learning rate at each epoch.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate.')
parser.add_argument('--number_of_samples', type=int, default=10000000, help='how many videos in the folder you want to use to train the model.')

args = parser.parse_args()

_PROCESSORS = 8

def train_test_model(X_train, X_val, X_test, y_train, y_val, y_test):

    checkpointer = ModelCheckpoint(filepath=FLAGS.model_path + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, period=10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=10, patience=1000)
    tensorbaord = TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=0, 
        batch_size=32, write_graph=True, write_grads=False, 
        write_images=False)
    
    model = ConvTemp(
        input_trace=FLAGS.input_trace, 
        output_trace=FLAGS.output_trace,
        input_dimension=FLAGS.input_dimension, 
        output_dimension=FLAGS.output_dimension,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay=FLAGS.learning_rate_decay)
    net = model.build_net()

    if FLAGS.restore == True:
        net.load_weights(FLAGS.model_path + ".hdf5")
        print("Loaded model from disk")

    history = net.fit(
          X_train, y_train,
          epochs=FLAGS.epochs,
          batch_size=FLAGS.batch_size,
          validation_data=(X_val, y_val),
          verbose=1,
          shuffle=True,
          callbacks=[tensorbaord, checkpointer, early_stopping]
          )

    test_loss, test_acc = net.evaluate(X_test, y_test)
    print('test acc:', test_acc)
    print('test loss:', test_loss)

    return

def main(argv):

    """ Removing current tensorboard folder. """
    try:
        shutil.rmtree('./tensorbaord')
    except FileNotFoundError:
        pass

    config = tf.ConfigProto(intra_op_parallelism_threads=_PROCESSORS, inter_op_parallelism_threads=_PROCESSORS)
    sess = tf.Session(config=config)
    K.set_session(sess)

    """ Importing the dataset and do the preprocessing required. """
    X_train, X_test, X_val, y_train, y_test, y_val = Preprocessing.traj_preprocessing(
        data_path=FLAGS.dataset, 
        calibration_method=FLAGS.calibration_method,
        number_of_samples=FLAGS.number_of_samples, 
        standardize=FLAGS.standardize_features,
        noise=FLAGS.noise,
        ground_calibration=FLAGS.ground_calibration,
        size_noise=FLAGS.size_noise,
        constraint_noise=FLAGS.constraint_noise,
        input_trace=FLAGS.input_trace,
        input_dim=FLAGS.input_dimension
        )

    train_test_model(X_train, X_val, X_test, y_train, y_val, y_test)
    return

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






