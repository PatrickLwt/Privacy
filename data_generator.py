import tensorflow as tf
import numpy as np
import scipy.io as sio


def get_data(net_info):
	### Load Dataset
	if net_info['dataset'] == 'cifar-10':
		cifar = tf.keras.datasets.cifar10
		(x_train, y_train), (x_test, y_test) = cifar.load_data()

	elif net_info['dataset'] == 'mnist':
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train[..., tf.newaxis]
		x_test = x_test[..., tf.newaxis]

	elif net_info['dataset'] == 'svhn':
		train_loaded = sio.loadmat('data/train_32x32.mat')
		test_loaded = sio.loadmat('data/test_32x32.mat')

		x_train, y_train = train_loaded["X"].astype(np.float32), train_loaded["y"].astype(np.int32)
		### data is in the shape of (32,32,3,73257)
		x_train = x_train.transpose(3, 0, 1, 2)
		y_train[y_train == 10] = 0
		### label is in the shape of (73257,1)
		y_train = y_train.reshape(len(y_train))

		x_test, y_test = test_loaded["X"].astype(np.float32), test_loaded["y"].astype(np.int32)
		x_test = x_test.transpose(3, 0, 1, 2)
		y_test[y_test == 10] = 0
		y_test = y_test.reshape(len(y_test))

	### Preprocessing
	x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
	mean = np.mean(x_train, axis=0)
	std = np.std(x_train, axis=0)
	std[std==0] = 0.00000001
	x_train -= mean
	x_test -= mean
	x_train /= std
	x_test /= std
	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return x_train, y_train, x_test, y_test

if __name__ == '__main__':
	net_info = {'dataset': 'svhn'}
	x_train, y_train, x_test, y_test = get_data(net_info)
