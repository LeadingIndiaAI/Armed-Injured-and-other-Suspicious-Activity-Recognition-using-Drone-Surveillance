import numpy as np

from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, MaxPooling2D)

from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense
import keras
import math

def _3dcnn(input_shape, nb_classes = 101):
	# Define model
    input_x = Input(shape = input_shape)

    initial_conv = Conv3D(16, kernel_size= (3, 3, 3), padding='same')(input_x)
    initial_conv = LeakyReLU(alpha=.001)(initial_conv)

    initial_conv = Conv3D(32, kernel_size= (3, 3, 3), padding='same')(initial_conv)
    initial_conv = LeakyReLU(alpha=.001)(initial_conv)

    ###########################
    # PARALLEL 1

    conv1 = Conv3D(16, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    conv1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)

    conv1 = Conv3D(16, kernel_size=(3, 3, 3),padding='same')(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    
    conv1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv1)
    conv1 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv1)
    conv1 = LeakyReLU(alpha=.001)(conv1)
    ##############################

    #Parallel 2

    conv2 = Conv3D(8, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv2 = LeakyReLU(alpha=.001)(conv2)

    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(8, kernel_size=(3, 3, 3),padding='same')(conv2)
    conv2 = LeakyReLU(alpha=.001)(conv2)
    
    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv2)
    conv2 = LeakyReLU(alpha=.001)(conv2)
    ##############################

    #Parallel 3

    conv3 = Conv3D(4, kernel_size=(1, 1, 1),padding='same')(initial_conv)
    conv3 = LeakyReLU(alpha=.001)(conv3)
    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)

    conv3 = Conv3D(4, kernel_size=(3, 3, 3),padding='same')(conv3)
    conv3 = LeakyReLU(alpha=.001)(conv3)

    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
    conv3 = Conv3D(1, kernel_size=(1, 1, 1),padding='same')(conv3)
    conv3 = LeakyReLU(alpha=.001)(conv3)
    ###################################

    added = keras.layers.Add()([conv1, conv2, conv3])
    added = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(added)
    
    added = Flatten()(added)

    dense_1 = Dense(784)(added)
    dense_2 = Dense(nb_classes)(dense_1)

    model = Model(input_x, dense_2)

    return model