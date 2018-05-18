from itertools import product
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import TimeDistributed, Conv1D, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import mean_squared_error
from sklearn.model_selection import KFold

import ball_3d_coordinates.util.util as util

class TmpNet(object):
    def __init__(self, input_trace, batch_size, epochs, 
            log_dir=None, model_path=None):
        super(TmpNet, self).__init__()
        self.input_trace = input_trace
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir
        self.model_path = model_path
        self.checkpointers = self.build_checkpointers()
        self.model = self.build_model()

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def build_model(self):

        model = Sequential()

        model.add(Conv1D(filters=256, kernel_size=self.input_trace, padding="valid", 
            activation='relu', input_shape=(self.input_trace, 3)))
        model.add(TimeDistributed(Dense(3)))
        model.summary()

        model.compile(
            optimizer=Adam(),
            loss=TmpNet.root_mean_squared_error, 
            metrics=['mae'])

        return model

    def build_checkpointers(self):

        checkpointers = []

        if self.model_path is not None:
            checkpointer = ModelCheckpoint(filepath=self.model_path + '.h5', 
                monitor='val_loss', verbose=1, save_best_only=True, period=10)
            checkpointers.append(checkpointer)
        
        if self.log_dir is not None:
            tensorboard = TensorBoard(log_dir=self.log_dir, histogram_freq=0, 
                batch_size=32, write_graph=True, write_grads=False, write_images=False)
            checkpointers.append(tensorboard)

        return checkpointers

    def fit(self, X_train, y_train, X_val, y_val):

        history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    batch_size=self.batch_size,
                    verbose=1,
                    callbacks=self.checkpointers)

        return history

    def evaluate(self, X_test, y_test):
        
        loss, metric = self.model.evaluate(X_test, y_test, verbose=0)
        
        return loss, metric

    def predict(self, X):
        
        y_pred = self.model.predict(X)
        
        return y_pred

    def restore(self):
        try:
            self.model.load_weights(self.model_path + ".h5")
            print("Loaded model from disk")
        except OSError as e:
            print("Model not loaded... for: ", e)
        except TypeError as e:
            print("Model not loaded... for: ", e)
        
        return
    
    def tune(self, X_train, y_train):
        raise NotImplementedError
        learning_rate = [0.01, 0.1, 0.0001]
        decay_rate = [1e-4, 1e-8, 1e-6, 1e-10]

        results = []
        for lr, dc in product(learning_rate, decay_rate):
            print("Training with lr: %s, and decay: %s" %(lr, dc))
            k_fold = KFold(n_splits=3, random_state=0)
            min_loss = []
            min_val_loss = []
            i = 0
            for train_idx, test_idx in k_fold.split(X_train):
                i += 1
                print("fold: ", i)
                
                """ Opening the session manually """
                sess = tf.InteractiveSession()
                
                net = SSDKeras.build_net_for_tuning(lr, dc)
                tmp_x_train, tmp_x_val = X_train[train_idx], X_train[test_idx]
                tmp_y_train, tmp_y_val = y_train[train_idx], y_train[test_idx]
                history = net.fit(tmp_x_train, tmp_y_train, epochs=epochs, 
                    validation_data=(tmp_x_val, tmp_y_val), batch_size=batch_size, verbose=0)
                min_loss.append(np.min(history.history['loss']))
                min_val_loss.append(np.min(history.history['val_loss']))
                
                """ Closing the session manually """
                del net
                K.clear_session()
                sess.close()
            min_loss = np.asarray(min_loss)
            min_val_loss = np.asarray(min_val_loss)
            print("LOSS: %s, VAL_LOSS: %s" %(min_loss, min_val_loss))
            results.append([lr, dc, np.mean(min_loss), np.mean(min_val_loss)])

        for line in results:
            print("LR: %s, DECAY: %s, LOSS: %s, VAL_LOSS: %s" %(line[0], line[1], line[2], line[3]))   
        return

		