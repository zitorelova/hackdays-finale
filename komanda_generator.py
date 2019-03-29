import numpy as np 
import tensorflow as tf 
import os
slim = tf.contrib.slim

# MODEL PARAMS
SEQ_LEN = 10 
BATCH_SIZE = 4
LEFT_CONTEXT = 5

# IMAGE PARAMS
IMG_HEIGHT = 120
IMG_WIDTH = 160
CHANNELS = 3

# LSTM PARAMS
RNN_SIZE = 32
RNN_PROJ = 32

class BatchGenerator(object):
	def __init__(self, sequence, seq_len, batch_size): 
		self.sequence = sequence
		self.seq_len = seq_len
		self.batch_size = batch_size
		chunk_size = 1 + (len(sequence) - 1) / batch_size
		self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]

	def next(self):
		while True:
			output = []
			for i in range(self.batch_size):
				idx = self.indices[i]
				left_pad = self.sequence[idx - LEFT_CONTEXT:idx]
				if len(left_pad) == LEFT_CONTEXT:
					left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
				assert len(left_pad) == LEFT_CONTEXT


test = BatchGenerator(sequence=[i for i in range(5)], seq_len=10, batch_size=250)
