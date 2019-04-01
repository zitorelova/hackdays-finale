from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv3D, MaxPooling3D, Input, Activation, Dropout, BatchNormalization, Reshape
from keras import Sequential
from keras.models import load_model, Model
import keras.backend as K
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image
from resnet_blocks import identity_block, convolution_block
from model_utils import *

BATCH_SIZE = 1
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
SEQ_LEN = 5
INPUT_SHAPE = (SEQ_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

data_dir = './data'
image_paths = sorted([i for i in os.listdir(data_dir)])
steering_angles = [get_steering_angle(i) for i in image_paths]

tr_gen = batch_generator(data_dir, image_paths, steering_angles, batch_size=BATCH_SIZE, is_training=True, is_3d=True)

bn_name_base = 'bn'

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

X_in = Input(INPUT_SHAPE)
X = Conv3D(filters=3, kernel_size=(3,3,3))(X_in)
X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
X = Activation('relu')(X)
X = MaxPooling3D(pool_size=(2,2,2))(X)
X = Conv3D(filters=64, kernel_size=(3,3,3))(X)
X = BatchNormalization(axis=-1, name=bn_name_base + '3a')(X)
X = Activation('relu')(X)
X = identity_block(X, 3, filters=64, stage=1, block=3)
X = Conv3D(filters=64, kernel_size=(3,3,3))(X)
X = BatchNormalization(axis=-1, name=bn_name_base + '4a')(X)
X = Activation('relu')(X)
X = identity_block(X, 3, filters=64, stage=2, block=4)
X = Conv3D(filters=8, kernel_size=(3,3,3))(X)
X = BatchNormalization(axis=-1, name=bn_name_base + '5a')(X)
X = Activation('relu')(X)
X = Conv3D(filters=8, kernel_size=(3,3,3))(X)
X = BatchNormalization(axis=-1, name=bn_name_base + '6a')(X)
X = Activation('relu')(X)
X = Conv3D(filters=8, kernel_size=(3,3,3))(X)
X = BatchNormalization(axis=-1, name=bn_name_base + '7a')(X)
X = Activation('relu')(X)
X = Flatten()(X)
X = LSTM(64, return_sequences=True)(X)
X = LSTM(16, input_shape=X.shape[1:])(X)
X = Dropout(0.8)(X) # hit or miss with this one

X = Dense(512, activation='relu')(X)
X = Dense(128, activation='relu')(X)
X = Dense(64, activation='relu')(X)
X = Dense(16, activation='relu')(X)
X = Dense(1)(X)

model = Model(inputs=X_in, outputs=X)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

model.fit_generator(generator=tr_gen, steps_per_epoch=1000, verbose=2, shuffle=False)

model.save('./models/3d_conv_lstm.h5')


