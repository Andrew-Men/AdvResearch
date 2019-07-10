#!/usr/local/anaconda3/envs/kaggle/bin python
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import glob
import numpy as np
import imageio
import keras
# import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization, Dropout


# FIX CRASH ON MAC #
import matplotlib
import platform
if platform.system() == 'Darwin':
	matplotlib.use("TkAgg")
# --------- #
from matplotlib import pyplot as plt

# for reproductive
# np.random.seed(42)
# tf.set_random_seed(42)
# sess = tf.Session(graph=tf.get_default_graph())
# keras.backend.set_session(sess)

def loadData():
	working_dir = os.path.split(os.path.realpath(__file__))[0]
	pos_data = np.load(os.path.join(working_dir,'positive_data.npy'))
	neg_data = np.load(os.path.join(working_dir,'negitive_data.npy'))
	data = np.concatenate((pos_data,neg_data),axis=0)
	labels = np.concatenate((np.ones((pos_data.shape[0],), dtype=int),np.zeros((neg_data.shape[0],), dtype=int)))

	return (data, labels)


# 绘图用
class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = {'batch':[], 'epoch':[]}
		self.accuracy = {'batch':[], 'epoch':[]}
		self.val_loss = {'batch':[], 'epoch':[]}
		self.val_acc = {'batch':[], 'epoch':[]}

	def on_batch_end(self, batch, logs={}):
		self.losses['batch'].append(logs.get('loss'))
		self.accuracy['batch'].append(logs.get('acc'))
		self.val_loss['batch'].append(logs.get('val_loss'))
		self.val_acc['batch'].append(logs.get('val_acc'))

	def on_epoch_end(self, batch, logs={}):
		self.losses['epoch'].append(logs.get('loss'))
		self.accuracy['epoch'].append(logs.get('acc'))
		self.val_loss['epoch'].append(logs.get('val_loss'))
		self.val_acc['epoch'].append(logs.get('val_acc'))

	def loss_plot(self):
		loss_type = 'epoch'
		iters = range(len(self.losses[loss_type]))

		plt.figure()
		# acc
		plt.subplot(2,1,1)
		plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
		plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
		plt.xlabel('epoch')
		plt.ylabel('acc-rate')
		plt.grid(True)
		plt.legend(loc="upper right")

		# loss
		plt.subplot(2,1,2)
		plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
		plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.grid(True)
		plt.legend(loc="upper right")

		plt.show()

data, labels = loadData()
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.3, random_state=42)


model = Sequential()

model.add(Conv2D(10, (5, 5), strides=1, padding='valid',data_format='channels_first', input_shape=(3,50,50)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(MaxPool2D((2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(10, (3, 3), strides=1, padding='valid'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(activation="sigmoid", units=1))

adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True)
# datagen.fit(x_train)

history = LossHistory()
model.fit(x_train, y_train, steps_per_epoch=15, epochs=100, validation_data=(x_val,y_val), validation_steps=10, callbacks=[history])
history.loss_plot()

# for reproductive
keras.backend.clear_session()

# imageio.help('PNG')