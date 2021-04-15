import numpy as np
import tensorflow as tf

# llambda = tf.constant(3000.0)
c1 = tf.constant((2 ** (1 / 3)) + (2 ** (-2 / 3)))


def get_alpha(clip_value, grad_array):
    clip_value = tf.cast(clip_value, tf.float32)
    grad_array = tf.cast(grad_array, tf.float32)
    alpha = tf.minimum(clip_value, tf.norm(grad_array), name="alpha")
    return alpha


## update the markov bound to (2epsilon+(2ln(delta))/t)/(1+t)
def get_markov_bound(epsilon, delta, noise_iter):
    t = (4 * tf.math.log(delta) - tf.math.sqrt(16*tf.math.log(delta)*(tf.math.log(delta)-epsilon))) / (-4 * epsilon)
    eps = tf.math.multiply(tf.constant(2.0), epsilon)
    lamd = tf.math.divide(tf.constant(2.0), t)
    lamd2 = tf.math.multiply(tf.add(tf.constant(1.0), t), noise_iter)
    det = tf.math.log(delta)
    markov_bound_raw = tf.add(eps, tf.math.multiply(lamd, det), name="y_bound")
    markov_bound = tf.math.divide(markov_bound_raw, lamd2)
    return markov_bound


def generate_noise(grad_array, clip_value, epsilon, delta, noise_iter):
    # grad_array = tf.cast(grad_array, tf.float64)
    # clip_value = tf.cast(clip_value, tf.float64)
    # epsilon = tf.cast(epsilon, tf.float64)
    # delta = tf.cast(delta, tf.float64)

    alpha = get_alpha(clip_value, grad_array)
    y_bound = get_markov_bound(epsilon, delta, noise_iter)

    # grad_num = grad_array.get_shape().as_list()[0]

    Delta_star = tf.math.divide(tf.math.multiply(alpha, grad_array), tf.norm(grad_array))

    ldeno1 = tf.math.multiply(Delta_star, grad_array)
    ldeno2 = tf.math.pow(ldeno1, tf.constant(2 / 3))
    ldeno3 = tf.reduce_sum(ldeno2)
    ldeno = tf.math.multiply(c1, ldeno3)
    lnume = tf.math.multiply(tf.constant(3.0), y_bound)
    lambda_star = tf.math.pow(tf.math.divide(lnume, ldeno), tf.constant(-3 / 2))

    sdeno = tf.abs(grad_array)
    not_equal = tf.not_equal(sdeno, 0)
    replace = tf.ones_like(sdeno) / 1000000000000
    sdeno = tf.where(not_equal, sdeno, replace)
    sdeno = tf.cast(sdeno, tf.float64)
    lambda_star = tf.cast(lambda_star, tf.float64)
    Delta_star = tf.cast(Delta_star, tf.float64)
    snume1 = tf.math.multiply(lambda_star, tf.square(Delta_star))
    snume = tf.math.multiply(tf.constant(2.0, dtype=tf.float64), snume1)
    sigma1 = tf.math.divide(snume, sdeno)
    sigma = tf.math.pow(sigma1, tf.constant(1 / 3, dtype=tf.float64))
    # sigma = tf.divide(sigma2, tf.constant(10.0 ** 4, dtype=tf.float64))
    # sigma = tf.cast(sigma, tf.float32)

    return sigma, tf.norm(grad_array)


def py_opt_noise(sigmas):
    centers = np.zeros(np.shape(sigmas))
    noise_np = np.random.normal(centers, sigmas)
    noise_np = np.float32(noise_np)
    return noise_np
