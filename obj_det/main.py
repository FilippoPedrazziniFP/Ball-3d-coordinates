import numpy as np
import sys
import tensorflow as tf
import argparse
import cv2
import os
import time
from numpy.linalg import pinv
import shutil
#import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras import backend as K

from preprocessing.data_preprocessing import Preprocessing
from models.ssd_keras import SSDKeras

parser = argparse.ArgumentParser()

""" General Parameters """
parser.add_argument('--model_path', type=str, default='./model/ssd_vgg_fixed', help='model checkpoints directory.')
parser.add_argument('--restore', type=bool, default=False, help='if True restore the model from --model_path.')
parser.add_argument('--fine_tuning', type=bool, default=True, help='if True unlocks some layers for tuning.')
parser.add_argument('--test', type=bool, default=False, help='if True it skips the training process and goes directly to test.')
parser.add_argument('--log_dir', type=str, default='./tensorbaord', help='directory where to store tensorbaord values.')

""" Model parameters """
parser.add_argument('--epochs', type=int, default=100, help='number of batch iterations.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for the training (number of traces to take).')
parser.add_argument('--learning_rate_decay', type=float, default=1e-3, help='how to decay the learning rate at each epoch.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate.')
parser.add_argument('--number_of_samples', type=int, default=10, help='how many videos in the folder you want to use to train the model.')

args = parser.parse_args()

##### Real values: (814, 1360)

_IMG_HEIGHT = 814
_IMG_WIDTH = 1360
_INPUT_CHANNELS = 3

def train_test_model(train_generator, validation_generator, test_generator):

    checkpointer = ModelCheckpoint(filepath=FLAGS.model_path + '.h5', monitor='val_loss', verbose=1, save_best_only=True, period=10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=10, patience=1000)
    tensorbaord = TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=0, 
        batch_size=32, write_graph=True, write_grads=False, 
        write_images=False)
    
    model = SSDKeras(
        img_height=_IMG_HEIGHT,
        img_width=_IMG_WIDTH,
        input_channels=_INPUT_CHANNELS,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay=FLAGS.learning_rate_decay)
    net = model.build_smaller_vgg_net()

    if FLAGS.restore == True:
        net.load_weights(FLAGS.model_path + ".h5")
        print("Loaded model from disk")
    
    if FLAGS.test != True:
        history = net.fit_generator(
            train_generator,
            epochs=FLAGS.epochs,
            validation_data=validation_generator,
            verbose=1,
            steps_per_epoch=5,
            validation_steps=1,
            callbacks=[tensorbaord, checkpointer, early_stopping]
            )

    """ Second training for fine tuning """
    if FLAGS.fine_tuning == True:
        net = model.fine_tuning(net)

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

    # net.save(FLAGS.model_path + '.h5')

    """ Check visually the prediction """
    for x, y in train_generator:
        img_tmp = np.reshape(x[0], (1, _IMG_HEIGHT, _IMG_WIDTH, _INPUT_CHANNELS))
        prediction = net.predict(img_tmp).flatten()
        width = int(prediction[0]*_IMG_WIDTH)
        height_ = int(prediction[1]*_IMG_HEIGHT)
        label = y[0].flatten()
        print("PREDICTION: (%s, %s)" %(width, height_))
        height = abs(height_ - _IMG_HEIGHT)
        print("LABEL: (%s, %s)" %(int(label[0]*_IMG_WIDTH), int(label[1]*_IMG_HEIGHT)))
        original_img = x[0]*255.0
        img = cv2.circle(original_img, (width, height), 6, (255, 0, 0), -1)
        cv2.imwrite('test.png',img)
        break

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
        img_height=_IMG_HEIGHT,
        img_width=_IMG_WIDTH,
        input_channels=_INPUT_CHANNELS)
    train_test_model(train_generator, validation_generator, test_generator )
    return

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






