from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras import Sequential
from keras.models import load_model
from akmtdfgen import generator_from_df
from akmtdfgen import get_demo_data
import keras.backend as K
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Change directories accordingly
print("Loading in dataframe...")
label_file = pd.read_csv('./data/final_example.csv')
img_dir = './data'
base_path = img_dir
f_paths = [os.path.join(base_path, str(i) + '.jpg') for i in label_file['frame_id']]
label_file['path'] = f_paths
label_file = label_file[['path', 'steering_angle']]
label_file.columns = ['imgpath', 'target']
img_width, img_height = 640, 480
epochs = 5
batch_size = 8
target_size = (img_width, img_height)


# Define loss function
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# Pretrained resnet50 with first 25 layers frozen
model = ResNet50(weights='imagenet', include_top=False,
                input_shape=(img_width, img_height, 3))
for layer in model.layers[:25]:
    layer.trainable=False

top_model = Sequential()
top_model.add(model)
top_model.add(Flatten())
top_model.add(Dense(512, activation='relu'))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(64, activation='relu'))
top_model.add(Dense(1))

top_model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

filepath = ('resnet50-freeze-{epoch:02d}-{val_loss:.2f}.h5')
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,
	verbose=1, period=2)

train_df = label_file.iloc[:int(np.floor(label_file.shape[0] * 0.7)),:]
valid_df = label_file.iloc[int(np.floor(label_file.shape[0] * 0.7)):,:]

train_gen = generator_from_df(train_df, batch_size, target_size)
valid_gen = generator_from_df(valid_df, batch_size, target_size)
nbatches_train, mod = divmod(train_df.shape[0], batch_size)
nbatches_valid, mod = divmod(valid_df.shape[0], batch_size)
nworkers=8

print("Training model...")
top_model.fit_generator(train_gen, 
                        steps_per_epoch=nbatches_train,
                       epochs=epochs,
                       validation_data=valid_gen,
                       validation_steps=nbatches_valid,
                       workers=nworkers, 
                       verbose=2)
