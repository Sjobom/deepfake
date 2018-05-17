from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
import numpy as np
import pickle
from keras.models import load_model
import tensorflow as tf

#DIM_ENCODER = 1024
#DIM_ENCODER = 64
IMAGE_SIZE = (64, 64, 3)

class DeepModel():

    def __init__(self, DIM_ENCODER=1024, lr=5e-5):
        self.DIM_ENCODER = DIM_ENCODER
        self.lr = lr
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
        optimizer = Adam(lr=lr, beta_1=0.5, beta_2=0.999)
        self.autoencoder_1.compile(optimizer=optimizer, loss='mean_absolute_error')
        self.autoencoder_2.compile(optimizer=optimizer, loss='mean_absolute_error')

        print('New Network')
        for layer in self.autoencoder_1.layers:
            print(layer.get_output_at(0).get_shape().as_list())


    # Block definition of the cnn layer
    def ConvLayer(self, filters):
        def cnn_layer(layer):
            layer = Conv2D(filters, kernel_size=5, strides=2, data_format="channels_last", padding='same')(layer)
            layer = LeakyReLU(0.1)(layer)
            return layer
        return cnn_layer

    def conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x

        return block

    def Encoder(self):
        input = Input(shape=IMAGE_SIZE)
        # CNN layers
        convLayer = self.conv(128)(input)
        convLayer = self.conv(256)(convLayer)
        convLayer = self.conv(512)(convLayer)
        convLayer = self.conv(1024)(convLayer)

        # Fully connected layers
        denseLayer = Dense(self.DIM_ENCODER)(Flatten()(convLayer))
        denseLayer = Dense(4 * 4 * 1024)(denseLayer)
        denseLayer = Reshape((4, 4, 1024))(denseLayer)

        # Start to upscale
        upscale = Conv2DTranspose(filters=512, kernel_size=5)(denseLayer)

        return Model(input, upscale)

    def NewEncoder(self):
        input = Input(shape=IMAGE_SIZE)
        # CNN layers
        convLayer = Conv2D(filters=128, kernel_size=5, strides=2, activation='relu', padding='same')(input)
        convLayer = Conv2D(filters=256, kernel_size=5, strides=2, activation='relu', padding='same')(convLayer)
        convLayer = Conv2D(filters=512, kernel_size=5, strides=2, activation='relu', padding='same')(convLayer)
        convLayer = Conv2D(filters=1024, kernel_size=5, strides=2, activation='relu', padding='same')(convLayer)

        # Fully connected layers
        denseLayer = Dense(self.DIM_ENCODER)(Flatten()(convLayer))
        denseLayer = Dense(4*4*1024)(denseLayer)
        denseLayer = Reshape((4,4,1024))(denseLayer)

        # Start to upscale
        upscale = Conv2DTranspose(filters=512, kernel_size=5)(denseLayer)

        return Model(input, upscale)

    def Decoder(self):
        input = Input(shape=(8, 8, 512))
        upConv = Conv2DTranspose(filters=256, kernel_size=7, strides=(2,2), activation='relu', padding='same')(input)
        upConv = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(upConv)
        upConv = Conv2DTranspose(filters=64, kernel_size=6, strides=(2, 2), activation='relu', padding='same')(upConv)
        convLayer = Conv2D(filters=3, kernel_size=5, activation='sigmoid', padding='same')(upConv)
        return Model(input, convLayer)

if __name__ == '__main__':
    run = DeepModel()
