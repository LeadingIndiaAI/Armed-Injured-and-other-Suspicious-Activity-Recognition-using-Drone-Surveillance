import numpy as np
import keras
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten, 
                          MaxPooling3D, BatchNormalization, AveragePooling3D, 
                          Reshape, Lambda, GlobalAveragePooling3D, Concatenate,
                          ReLU, Add)
from keras.models import Sequential, Model, load_model
import tensorflow as tf

## Slow Fast network model ##

def Conv_BN_ReLU(planes, kernel_size, strides=(1, 1, 1), padding='same', use_bias=False):
    return Sequential([
        Conv3D(planes, kernel_size, strides=strides, padding=padding, use_bias=use_bias),
        BatchNormalization(),
        ReLU()
    ])


def bottleneck(x, planes, stride=1, downsample=None, head_conv=1, use_bias=False):
    residual = x
    if head_conv == 1:
        x = Conv_BN_ReLU(planes, kernel_size=1, use_bias=use_bias)(x)
    elif head_conv == 3:
        x = Conv_BN_ReLU(planes, kernel_size=(3, 1, 1), use_bias=use_bias)(x)
    else:
        raise ValueError('Unsupported head_conv!!!')
    x = Conv_BN_ReLU(planes, kernel_size=(1, 3, 3), strides=(1, stride, stride), use_bias=use_bias)(x)
    x = Conv3D(planes*4, kernel_size=1, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    if downsample is not None:
        residual = downsample(residual)
    x = Add()([x, residual])
    x = ReLU()(x)
    return x

def datalayer(x, stride):
    return x[:, ::stride, :, :, :]

def SlowFast_body(inputs, layers, block, num_classes, dropout=0.5):
    inputs_fast = Lambda(datalayer, name='data_fast', arguments={'stride':2})(inputs)
    inputs_slow = Lambda(datalayer, name='data_slow', arguments={'stride':16})(inputs)
    fast, lateral = Fast_body(inputs_fast, layers, block)
    slow = Slow_body(inputs_slow, lateral, layers, block)
    x = Concatenate()([slow, fast])
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, out)



def Fast_body(x, layers, block):
    fast_inplanes = 8
    lateral = []
    x = Conv_BN_ReLU(8, kernel_size=(5, 7, 7), strides=(1, 2, 2))(x)
    x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    lateral_p1 = Conv3D(8*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_p1)
    x, fast_inplanes = make_layer_fast(x, block, 8, layers[0], head_conv=3, fast_inplanes=fast_inplanes)
    lateral_res2 = Conv3D(32*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_res2)
    x, fast_inplanes = make_layer_fast(x, block, 16, layers[1], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
    lateral_res3 = Conv3D(64*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_res3)
    x, fast_inplanes = make_layer_fast(x, block, 32, layers[2], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
    lateral_res4 = Conv3D(128*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
    lateral.append(lateral_res4)
    x, fast_inplanes = make_layer_fast(x, block, 64, layers[3], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
    x = GlobalAveragePooling3D()(x)
    return x, lateral

def Slow_body(x, lateral, layers, block):
    slow_inplanes = 64 + 64//8*2
    x = Conv_BN_ReLU(64, kernel_size=(1, 7, 7), strides=(1, 2, 2))(x)
    x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = Concatenate()([x, lateral[0]])
    x, slow_inplanes = make_layer_slow(x, block, 64, layers[0], head_conv=1, slow_inplanes=slow_inplanes)
    x = Concatenate()([x, lateral[1]])
    x, slow_inplanes = make_layer_slow(x, block, 128, layers[1], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
    x = Concatenate()([x, lateral[2]])
    x, slow_inplanes = make_layer_slow(x, block, 256, layers[2], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
    x = Concatenate()([x, lateral[3]])
    x, slow_inplanes = make_layer_slow(x, block, 512, layers[3], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
    x = GlobalAveragePooling3D()(x)
    return x


def make_layer_fast(x, block, planes, blocks, stride=1, head_conv=1, fast_inplanes=8, block_expansion=4):
    downsample = None
    if stride != 1 or fast_inplanes != planes * block_expansion:
        downsample = Sequential([
            Conv3D(planes*block_expansion, kernel_size=1, strides=(1, stride, stride), use_bias=False),
            BatchNormalization()
        ])
    fast_inplanes = planes * block_expansion
    x = block(x, planes, stride, downsample=downsample, head_conv=head_conv)
    for _ in range(1, blocks):
        x = block(x, planes, head_conv=head_conv)
    return x, fast_inplanes

def make_layer_slow(x, block, planes, blocks, stride=1, head_conv=1, slow_inplanes=80, block_expansion=4):
    downsample = None
    if stride != 1 or slow_inplanes != planes * block_expansion:
        downsample = Sequential([
            Conv3D(planes*block_expansion, kernel_size=1, strides = (1, stride, stride), use_bias=False),
            BatchNormalization()
        ])
    x = block(x, planes, stride, downsample, head_conv=head_conv)
    for _ in range(1, blocks):
        x = block(x, planes, head_conv=head_conv)
    slow_inplanes = planes * block_expansion + planes * block_expansion//8*2
    return x, slow_inplanes