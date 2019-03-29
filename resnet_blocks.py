import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Conv3D, MaxPooling3D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
#from keras.applications.imagenet_utils import preprocess_input
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
	"""
	Identity block for resnet based architecture

	"""
	conv_name_base = 'res' + str(stage) + str(block) + '_branch'
	bn_name_base = 'bn' + str(stage) + str(block) + '_branch'


	# Save the input value. You'll need this later to add back to the main path. 
	X_shortcut = X

	# First component of main path
	X = Conv3D(filters=filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
	X = BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
	X = Activation('relu')(X)

	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)

	return X

def residual_conv_block(X, f, filters, stage, block):
	# not sure if we need a residual conv block if we are only skipping one layer anyway
	# it will be practically the same thing
	pass

def convolution_block(X, f, filters, stage, block):

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name = 'bn' + str(stage) + block + '_branch'

	X_shortcut = X

	return X

