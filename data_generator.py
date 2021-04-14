import tensorflow as tf
import numpy as np


def get_data(net_info):
	if net_info['dataset'] == 'cifar-10':
		cifar = tf.keras.datasets.cifar10
		(x_train, y_train), (x_test, y_test) = cifar.load_data()

	elif net_info['dataset'] == 'mnist':
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train[..., tf.newaxis]
		x_test = x_test[..., tf.newaxis]

	x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
	mean = np.mean(x_train, axis=0)
	std = np.std(x_train, axis=0)
	x_train -= mean
	x_test -= mean
	if net_info['dataset'] == 'cifar-10':
		x_train /= std
		x_test /= std
	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return x_train, y_train, x_test, y_test
