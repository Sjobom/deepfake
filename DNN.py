from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
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
        self.decoder = self.Decoder()

        print('New Network')
        for layer in self.decoder.layers:
            print(layer.get_output_at(0).get_shape().as_list())

        # Create the shared encoder
        self.encoder = self.NewEncoder()

        # The different decoders
        self.decoder_1 = self.Decoder()
        self.decoder_2 = self.Decoder()

        input = Input(shape=IMAGE_SIZE)

        # Create the to autoencoders with shared encoder
        self.autoencoder_1 = Model(input, self.decoder_1(self.encoder(input)))
        self.autoencoder_2 = Model(input, self.decoder_2(self.encoder(input)))

        # Compile the autoencoders
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        self.autoencoder_1.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_2.compile(optimizer=optimizer, loss='mean_absolute_error')




        print('New Network')
        for layer in self.autoencoder_1.layers:
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
        L1 = Conv2D(filters=128, kernel_size=5, data_format="channels_last", activation='relu')(input)
        L2 = Conv2D(filters=256, kernel_size=5, activation='relu')(L1)

        # Fully connected layers
        L3 = Dense(DIM_ENCODER)(Flatten()(L2))
        L4 = Dense(4*4*1024)(L3)
        L4 = Reshape((4,4,1024))(L4)

        # Start to upscale
        L5 = Conv2DTranspose(filters=3, kernel_size=9)(L4)

        return Model(input, L5)

    def EncoderLayers(self):
        input = Input(shape=IMAGE_SIZE)
        # CNN layers
        L1 = Conv2D(filters=128, kernel_size=5, data_format="channels_last", activation='relu')(input)
        L2 = Conv2D(filters=256, kernel_size=5, activation='relu')(L1)

        # Fully connected layers
        L3 = Dense(DIM_ENCODER)(Flatten()(L2))
        L4 = Dense(4*4*1024)(L3)
        L4 = Reshape((4,4,1024))(L4)

        # Start to upscale
        L5 = Conv2DTranspose(filters=3, kernel_size=9)(L4)

        return Model(input, L5)

    def Decoder(self):
        input = Input(shape=(12, 12, 3))
        L1 = Conv2DTranspose(filters=128, kernel_size=3, strides=(1,1))(input)
        L2 = Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2))(L1)
        L3 = Conv2DTranspose(filters=3, kernel_size=4, strides=(2, 2))(L2)
        return Model(input, L3)

if __name__ == '__main__':
    run = DeepModel()