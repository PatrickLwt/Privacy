import numpy as np
import tensorflow as tf 

def randomWsigma(size):
	a = np.random.rand(size,3)
	q, r = np.linalg.qr(a,'complete')
	return q
	
def calculateBound(query_shape, l2_sensitivity, epsilon, delta, print_log=False):
	shape_mul = 1
	if len(query_shape)!=0:
		for i in range(len(query_shape)):
			shape_mul = shape_mul * query_shape[i]
	zeta = tf.math.sqrt(2 * tf.math.sqrt(- shape_mul * tf.math.log(delta)) - 2 * tf.math.log(delta) + shape_mul)
	alpha = l2_sensitivity ** 2
	alpha = tf.cast(alpha, tf.float64)
	beta = 2 * zeta * l2_sensitivity
	beta = tf.cast(beta, tf.float64)
	bd = (tf.math.square(- beta + tf.math.sqrt(beta ** 2 + 8 * alpha * epsilon))) / (4 * (alpha ** 2))
	# non_zero = tf.not_equal(bd, 0.0)
	# small = tf.constant(1.e-7)
	# bound = tf.where(non_zero, bd, small)
	bound = tf.cast(bd, tf.float32)
	if print_log:
		print('our bound: alpha={} beta={} zeta={} bound={}'.format(alpha, beta, zeta, bound))
	return bound

def mvg_sampler(sigma, query_shape):
	noise = tf.random.normal(query_shape)
	if len(query_shape)<3:
		b_sigma = tf.math.sqrt(sigma)*tf.eye(query_shape[0])
	else:
		b_sigma = tf.math.sqrt(sigma)*tf.eye(query_shape[-1])
	try:
		Z = tf.linalg.matmul(b_sigma, noise)
	except:
		Z = tf.linalg.matmul(b_sigma, tf.expand_dims(noise, -1))
		Z = tf.squeeze(Z, axis=-1)
	return Z

def getNoise(feature, l2_sensitivity, epsilon, delta, print_log=False):
	query_shape = feature.get_shape().as_list()
	bound = calculateBound(query_shape, l2_sensitivity, epsilon, delta, print_log)

	sigma = 1/bound
	# ssigma = tf.transpose(sigma)[0]
	# noise = mvg_sampler(sigma, query_shape)
	noise = tf.math.sqrt(sigma) * tf.random.normal(query_shape)
	return noise

def calculateIDNBound(query_shape, l2_sensitivity, epsilon, delta, print_log=False):
	shape_mul = 1
	I_1 = 1
	if len(query_shape)!=0:
		I_1 = query_shape[0]
		for i in range(len(query_shape)):
			shape_mul = shape_mul * query_shape[i]
	zeta = tf.math.sqrt(2 * tf.math.sqrt(- shape_mul * tf.math.log(delta)) - 2 * tf.math.log(delta) + shape_mul)
	alpha = l2_sensitivity * l2_sensitivity
	alpha = tf.cast(alpha, tf.float32)
	beta = 2 * zeta * l2_sensitivity
	beta = tf.cast(beta, tf.float64)
	bd = I_1 * ((- zeta + tf.math.sqrt(zeta * zeta + 2 * epsilon)) ** 2) / (alpha)
	# non_zero = tf.not_equal(bd, 0.0)
	# small = tf.constant(1.e-7)
	# bound = tf.where(non_zero, bd, small)
	bound = tf.cast(bd, tf.float32)
	if print_log:
		print('our bound: alpha={} beta={} zeta={} bound={}'.format(alpha, beta, zeta, bound))
	return bound

def getIDNNoise(feature, l2_sensitivity, epsilon, delta, print_log=False):
	query_shape = feature.get_shape().as_list()
	bound = calculateIDNBound(query_shape, l2_sensitivity, epsilon, delta, print_log)

	sigma = 1/bound
	# ssigma = tf.transpose(sigma)[0]
	# noise = mvg_sampler(sigma, query_shape)
	# tf.print('sigma=', sigma)
	noise = tf.math.sqrt(sigma) * tf.random.normal(query_shape)
	return noise

