import numpy as np
import tensorflow as tf

## Make the w all 1 
c1 = tf.constant((2 ** (1 / 3)) + (2 ** (-2 / 3)))

## update the markov bound to (2epsilon+(2ln(delta))/t)/(1+t)
def get_markov_bound(epsilon, delta, noise_iter):
    t = (4 * tf.math.log(delta) - tf.math.sqrt(16*tf.math.log(delta)*(tf.math.log(delta)-epsilon))) / (-4 * epsilon)
    eps = tf.math.multiply(tf.constant(2.0), epsilon)
    lamd = tf.math.divide(tf.constant(2.0), t)
    lamd2 = tf.math.multiply(tf.math.add(tf.constant(1.0), t), noise_iter)
    det = tf.math.log(delta)
    markov_bound_raw = tf.math.add(eps, tf.math.multiply(lamd, det), name="y_bound")
    markov_bound = tf.math.divide(markov_bound_raw, lamd2)
    return markov_bound


def generate_noise(grad_array, clip_value, epsilon, delta, noise_iter):
    alpha = tf.cast(clip_value, tf.float32)
    y_bound = get_markov_bound(epsilon, delta, noise_iter)
    const = tf.math.pow(tf.constant(2.0), tf.constant(1 / 3))

    alpha1 = tf.math.multiply(alpha, const)
    middle1 = tf.math.divide(tf.math.multiply(tf.constant(3.0), y_bound), c1)
    middle = tf.math.pow(middle1, tf.constant(-1 / 2))
    sigma_value = tf.math.multiply(alpha1, middle)
    sigma = tf.ones_like(grad_array)
    sigma = sigma_value * sigma


    return sigma


def py_opt_noise(sigmas):
    centers = np.zeros(np.shape(sigmas))
    noise_np = np.random.normal(centers, sigmas)
    noise_np = np.float32(noise_np)
    return noise_np
