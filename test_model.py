from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras import Sequential
from akmtdfgen import generator_from_df
from akmtdfgen import get_demo_data
import keras.backend as K
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image

print("Loading in dataframe...")
label_file = pd.read_csv('./data/final_example.csv')
img_dir = './data'
base_path = './data'
f_paths = [os.path.join(base_path, str(i) + '.jpg') for i in label_file['frame_id']]
label_file['path'] = f_paths
label_file = label_file[['path', 'steering_angle']]
label_file.columns = ['imgpath', 'target']
img_width, img_height = 120, 160
batch_size = 8
target_size = (img_width, img_height)

test = Sequential()
test.add(Conv2D(filters=8, kernel_size=(3,3), input_shape=(img_width, img_height, 3)))
test.add(Conv2D(filters=8, kernel_size=(3,3)))
test.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
test.add(Flatten())
test.add(Dense(64, activation='relu'))
test.add(Dense(1))

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

test.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

train_df = label_file.iloc[:int(np.floor(label_file.shape[0] * 0.7)),:]
valid_df = label_file.iloc[int(np.floor(label_file.shape[0] * 0.7)):,:]

train_gen = generator_from_df(train_df, batch_size, target_size)
valid_gen = generator_from_df(valid_df, batch_size, target_size)
nbatches_train, mod = divmod(train_df.shape[0], batch_size)
nbatches_valid, mod = divmod(valid_df.shape[0], batch_size)
nworkers=8

print("Training model...")
test.fit_generator(train_gen, 
                        steps_per_epoch=nbatches_train,
                       epochs=epochs,
                       validation_data=valid_gen,
                       validation_steps=nbatches_valid,
                       workers=8)