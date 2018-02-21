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
from models.ssd_keras import SSDKeras

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--model_path', type=str, default='./model/ssd_fixed', help='model checkpoints directory.')
parser.add_argument('--restore', type=bool, default=False, help='if True restore the model from --model_path.')
parser.add_argument('--log_dir', type=str, default='./tensorbaord', help='directory where to store tensorbaord values.')

""" Model parameters """
parser.add_argument('--epochs', type=int, default=10, help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for the training (number of traces to take).')
parser.add_argument('--learning_rate_decay', type=float, default=0.0, help='how to decay the learning rate at each epoch.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate.')
parser.add_argument('--number_of_samples', type=int, default=10, help='how many videos in the folder you want to use to train the model.')

args = parser.parse_args()

_IMG_HEIGHT_REAL = 814
_IMG_WIDTH_REAL = 1360

_IMG_HEIGHT = 200
_IMG_WIDTH = 400
_INPUT_CHANNELS = 3

def train_test_model(train_generator, validation_generator, test_generator ):

    checkpointer = ModelCheckpoint(filepath=FLAGS.model_path + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, period=10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=10, patience=1000)
    tensorbaord = TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=0, 
        batch_size=32, write_graph=True, write_grads=False, 
        write_images=False)
    
    model = SSDKeras(
        img_height=_IMG_HEIGHT_REAL,
        img_width=_IMG_WIDTH_REAL,
        input_channels=_INPUT_CHANNELS,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay=FLAGS.learning_rate_decay)
    net = model.build_vgg_net()

    if FLAGS.restore == True:
        net.load_weights(FLAGS.model_path + ".hdf5")
        print("Loaded model from disk")

    history = net.fit_generator(
          train_generator,
          epochs=FLAGS.epochs,
          validation_data=validation_generator,
          verbose=1,
          steps_per_epoch=5,
          validation_steps=1,
          callbacks=[tensorbaord, checkpointer, early_stopping]
          )

    test_loss, test_acc = net.evaluate_generator(test_generator)
    print('test acc:', test_acc)
    print('test loss:', test_loss)

    return

def main(argv):

    """ Removing current tensorboard folder. """
    try:
        shutil.rmtree('./tensorbaord')
    except FileNotFoundError:
        pass

    """ Importing the dataset and do the preprocessing required. """
    train_generator, validation_generator, test_generator = Preprocessing.preprocessing(
        number_of_samples=FLAGS.number_of_samples,
        img_height=_IMG_HEIGHT_REAL,
        img_width=_IMG_WIDTH_REAL,
        input_channels=_INPUT_CHANNELS)
    train_test_model(train_generator, validation_generator, test_generator )
    return

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






