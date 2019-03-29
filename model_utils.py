import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
SEQ_LEN = 5 # number of images in input

data_dir = './data'

def load_image(data_dir, image_file):
	"""
	Load RGB image from a file
	"""
	return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def get_steering_angle(image_file):
	"""
	Get steering angle from the filename
	"""
	return float(image_file.split('_')[5])

def crop(image):
	"""
	Crop the image (remove horizon)
	"""
	return image[50:-25, :, :]

def resize(image):
	"""
	Resize the image for input into model
	"""
	return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
	"""
	Convert image from RGB to YUV (This is what the NVIDIA model does)
	"""
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
	"""
	Combine all preprocess functions into one
	"""
	image = crop(image)
	image = resize(image)
	image = rgb2yuv(image)
	return image

def random_flip(image, steering_angle):
	"""
	Randomly flip the image and adjust the steering angle.
	"""
	if np.random.rand() < 0.5:
	   image = cv2.flip(image, 1)
	   steering_angle = -steering_angle
	return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly translate the image
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

# THIS FUNCTION IS SO HARD TO FIX ARGHHHH

def random_shadow(image):
    """
    Generate and add random shadow
    """
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augment(data_dir, image_file, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    """
    image = load_image(data_dir, image_file)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    # image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training, is_3d=False):
    """
    Generate training image given image paths and steering angles respectively
    """
    if is_3d:
        images = np.zeros([batch_size, SEQ_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        steers = np.empty(batch_size)
        i = 0
        while True:
            im_temp = np.empty([SEQ_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            for ix in range(i, i+SEQ_LEN):
                im_temp[ix % SEQ_LEN] = preprocess(load_image(data_dir, image_paths[ix]))
                images[0] = im_temp
                steers[0] = steering_angles[ix]
            yield images, steers
            i+=1

# Some tests
# image_paths = sorted([i for i in os.listdir(data_dir)])
# steering_angles = [get_steering_angle(i) for i in image_paths]
# temp = batch_generator(data_dir, image_paths, steering_angles, batch_size=1, is_training=True, is_3d=True)
