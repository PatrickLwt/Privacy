import tensorflow as tf
import numpy as np 

def generalized_harmonic_number(order, exponent=1):
    sum = tf.constant(0.)
    i = tf.constant(1)
    while_condition = lambda i, sum: tf.less(i, order+1)
    def body(i, sum):
        sum = sum + (1.0 / (tf.cast(i, dtype=tf.float32) ** exponent))
        return [tf.add(i, 1), sum]

    r, sum2 = tf.while_loop(while_condition, body, [i, sum])
        
    return sum2

def mvg_sampler(ssigma, query_shape):
    noise = tf.random.normal(query_shape)
    b_sigma = ssigma * tf.eye(query_shape[0])
    try:
        Z = tf.linalg.matmul(b_sigma, noise)
    except:
        Z = tf.linalg.matmul(b_sigma, tf.reshape(noise, (query_shape[0], -1)))
        # Z = tf.squeeze(Z, axis=-1)
        Z = tf.reshape(Z, query_shape)
    return Z

def calculateBound(m, n, l2_sensitivity, epsilon, delta):
    zeta = tf.sqrt(2 * tf.sqrt(- m * n * tf.math.log(delta)) - 2 * tf.math.log(delta) + m * n)
    gamma = l2_sensitivity / 2.0
    r = tf.minimum(m, n)
    r = tf.cast(r, tf.int32)
    H_r = generalized_harmonic_number(r)
    H_r_sqrt = generalized_harmonic_number(r, 0.5)
    alpha = (H_r + H_r_sqrt) * (gamma ** 2) + 2 * H_r * gamma * l2_sensitivity
    alpha = tf.cast(alpha, tf.float64)
    beta = 2 * ((m * n) ** 0.25) * H_r * l2_sensitivity * zeta
    beta = tf.cast(beta, tf.float64)
    bd = (tf.square(- beta + tf.sqrt(beta ** 2 + 8 * alpha * epsilon))) / (4 * (alpha ** 2))
    bound = tf.cast(bd, tf.float32)
    return bound

def getTheta(wsigma):
    columnsum = tf.reduce_sum(tf.abs(wsigma), axis=0)
    theta = columnsum / tf.reduce_sum(columnsum)
    return theta


def getNoise(grads, epsilon, delta, l2_sensitivity):
    query_shape = grads.get_shape().as_list()
    # print(query_shape)
    try:
        m, n = query_shape[0], query_shape[1]
    except:
        m = query_shape[0]
        n = 1

    bound = calculateBound(m, n, l2_sensitivity, epsilon, delta)
    budget = bound * bound / n
    # theta = getTheta(wsigma)
    # p = theta * budget
    # not_equal = tf.not_equal(p, 0)
    # replace = tf.ones_like(p)/1000000
    # p2 = tf.where(not_equal, p, replace)
    # sigma = 1 / tf.sqrt(p2)
    directional = False
    directional_tao = 0.8
    theta = np.ones(m)
    if directional:
        if m > 2:
            for i in range(m):
                theta[i] = (1 - directional_tao) / (m-2)
            theta[0] = directional_tao / 2
            theta[1] = directional_tao / 2
        else:
            if m == 1:
                theta[0] = directional_tao
            else:
                theta[0] = directional_tao / 2
                theta[1] = directional_tao / 2
    else:
        theta = theta / m  
    p = theta * budget
    sigma = 1 / tf.sqrt(p)
    noise = mvg_sampler(sigma, query_shape)
    return noise

def getNoisebias(grads, epsilon, delta, l2_sensitivity):
    query_shape = grads.get_shape().as_list()
    m = query_shape[0]
    n = 1
    bound = calculateBound(m, n, l2_sensitivity, epsilon, delta)
    budget = bound * bound / n
    sigma = 1 / tf.sqrt(budget)
    noise = tf.random_normal(query_shape)
    noise = noise * tf.sqrt(sigma)
    return noise

# grads = tf.ones((5,3,4,5))
# noise = getNoise(grads, 1, 1e-5, 1)
# print(noise.shape)
