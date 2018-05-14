from tensorflow.python import keras
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import CSVLogger
from keras.utils import np_utils
import numpy as np
import pickle
from keras.models import load_model
import tensorflow as tf

DIM_ENCODER = 1024
IMAGE_SIZE = (64, 64, 3)

class DeepModel():

    def __init__(self):
        self.Encoder = self.NewEncoder()
        for layer in self.Encoder.layers:
            print(layer.get_output_at(0).get_shape().as_list())
        self.Decoder = self.Decoder()
        print('New Network')
        for layer in self.Decoder.layers:
            print(layer.get_output_at(0).get_shape().as_list())


    # Block definition of the cnn layer
    def ConvLayer(self, filters):
        def cnn_layer(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            return x
        return cnn_layer

    def NewEncoder(self):
        input = Input(shape=IMAGE_SIZE)
        # CNN layers
        L1 = Conv2D(filters=128, kernel_size=5, activation='relu')(input)
        L2 = Conv2D(filters=256, kernel_size=5, activation='relu')(L1)

        # Fully connected layers
        L3 = Dense(DIM_ENCODER)(Flatten()(L2))
        L4 = Dense(4*4*1024)(L3)
        L4 = Reshape((4,4,1024))(L4)

        # Start to upscale
        L5 = Conv2DTranspose(filters=3, kernel_size=9)(L4)

        return Model(input, L5)

    def Decoder(self):
        input = Input(shape=(8,8,512))
        L1 = Conv2DTranspose(filters=128, kernel_size=2, strides=(2,2))(input)
        L2 = Conv2DTranspose(filters=64, kernel_size=6, strides=(1, 1))(L1)

        return Model(input, L2)

if __name__ == '__main__':
    run = DeepModel()