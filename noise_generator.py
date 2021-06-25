# noise_generator.py
# This contains Gaussian, IMGM, UDN, MVG mechanism for generating noise, and
# the unperturbed.
# input: grad = [] --- gradients of one layer
#        epsilon, delta --- the privacy parameters to achieve at the end of training
#        clip_value --- l2 norm clipping value of the gradients, delta_f / batch_size
#        sample_rate_q --- lot size / total number of examples, q = L/ N
#        total_epochs --- the total number of training epochs
# output: noise = [] --- random noise generated, has the same shape as grad
from __future__ import division

import numpy as np
import tensorflow as tf
import special_printer as sp
import noise_utils.udnnoise_generator as udn
import noise_utils.mvgnoise_generator as mvg
import noise_utils.imgm_bound as gb 
import noise_utils.optver2_noise_gen as opt2
import noise_utils.optver3_noise_gen as opt3
from noise_utils.cal_aplus import getAplus
from scipy import optimize
# for sig=6
# eps20 -> 0.72
# eps60 -> 1.24
# eps200 -> 2.26

#for sig=30

# Let's define some default values, in case we come up with other mechanisms
_epsilon = 1.0  # 2.0   0.5
_delta = 1e-5

_batch_size = 100.0   # 2000.0  # 600.0
_lot_size = 500.0
_l2_clip_value = 0.5 / _batch_size   # 3.0 / 2000 # 4.0 / 600.0      # !! TODO !! change l2_clip_value for every gradient

_total_num_examples = 50000.0
_total_epochs = 10.0   # 50.0    # 100.0


class GaussianMechanism(object):

    def __init__(self, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value, batch_size=_batch_size, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        # input sampling rate q
        self._sample_rate_q = lot_size / total_num_examples

        # total number of noise addition iterations
        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        # clip value per example
        self._clip_value = l2_clip_value

        self._batch_size = batch_size
        sp.info('Gaussian Noise Mechanism')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('clip', self._clip_value) +
                sp.var_print('noise_iter', self._noise_iter) +
                sp.var_print('sample_rate', self._sample_rate_q, 3)
                )

    # Return noise of the Gaussian mechanism to be added to the gradients.
    # Usage: called each time of generating noise.
    # parameters: grad --- tensor, gradients of a layer, does not need to be flattened.
    # return: noise --- tensor, the same shape as grad

    def generate_noise(self, grad):
        sigma = 2.0 * self._sample_rate_q / self._epsilon * np.sqrt(self._noise_iter * np.log(1.0 / self._delta))
        sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        noise = tf.random.normal(grad.get_shape(), 0.0, sigma * self._clip_value)
        return noise


class IMGMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value, batch_size=_batch_size, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        # input sampling rate q
        self._sample_rate_q = lot_size / total_num_examples

        # total number of noise addition iterations
        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        # clip value per example
        # self._clip_value = l2_clip_value / lot_size
        self._clip_value = l2_clip_value

        self._batch_size = batch_size

        sp.info('IMGM Noise Mechanism')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('clip', self._clip_value) +
                sp.var_print('noise_iter', self._noise_iter) +
                sp.var_print('sample_rate', self._sample_rate_q, 3)
                )

    # Return noise of the Gaussian mechanism to be added to the gradients.
    # Usage: called each time of generating noise.
    # parameters: grad --- tensor, gradients of a layer, does not need to be flattened.
    # return: noise --- tensor, the same shape as grad

    def generate_noise(self, grad):
        sigma = gb.calculate_gaussian_noise_stddev(self._epsilon, self._delta, self._clip_value, self._noise_iter, self._sample_rate_q) / np.sqrt(self._batch_size)
        noise = tf.random.normal(grad.get_shape(), 0.0, sigma)
        return noise


class UDNMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta, batch_size=_batch_size,
                 l2_clip_value=_l2_clip_value, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        self._sample_rate_q = lot_size / total_num_examples

        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        self._eps_per_iter = self._epsilon / (self._sample_rate_q) / 2.0 /\
                             np.sqrt(self._noise_iter * np.log(np.e + self._epsilon / self._delta))

        self._delta_per_iter = self._delta / (2 * self._noise_iter * self._sample_rate_q)

        self._clip_value = l2_clip_value 

        sp.info('UDN Noise Mechanism')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('sample_rate', self._sample_rate_q, 3) +
                sp.var_print('eps_per_iter', self._eps_per_iter) +
                sp.var_print('delta_per_iter', self._delta_per_iter)
                )

    def generate_noise(self, grad):
        noise = udn.getNoise(grad, self._clip_value, self._eps_per_iter , self._delta_per_iter)


        return noise

class IDNMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta, batch_size=_batch_size,
                 l2_clip_value=_l2_clip_value, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        self._sample_rate_q = lot_size / total_num_examples

        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        def calculate_eps_per(eps_total, delta=1e-6):

            def f1(epsilon):
                eps = (np.exp(epsilon) - 1) * epsilon * self._noise_iter / (np.exp(epsilon) + 1) + epsilon * np.sqrt(2 * self._noise_iter * np.log(np.e + np.sqrt(self._noise_iter) * epsilon / delta))
                return eps-eps_total

            def f2(epsilon):
                eps = (np.exp(epsilon) - 1) * epsilon * self._noise_iter / (np.exp(epsilon) + 1) + epsilon * np.sqrt(2 * self._noise_iter * np.log(1 / delta))
                return eps-eps_total
            eps1 = optimize.root_scalar(f1, bracket=[0, 10], method='brentq').root
            eps2 = optimize.root_scalar(f2, bracket=[0, 10], method='brentq').root

            return np.maximum(eps1, eps2)

        # a = self._noise_iter * self._epsilon
        # b = (np.exp(self._epsilon) - 1) * a / (np.exp(self._epsilon) + 1) + self._epsilon * np.sqrt(2 * self._noise_iter * np.log(np.e + np.sqrt(self._noise_iter) * self._epsilon / self._delta))
        # c = (np.exp(self._epsilon) - 1) * a / (np.exp(self._epsilon) + 1) + self._epsilon * np.sqrt(2 * self._noise_iter * np.log(1 / self._delta))
        # self._eps_per_iter = np.minimum(a, np.minimum(b, c))
        # self._eps_per_iter = self._epsilon / (self._sample_rate_q) / 2.0 /\
        #                      np.sqrt(self._noise_iter * np.log(np.e + self._epsilon / self._delta))
        self._eps_per_iter = calculate_eps_per(self._epsilon) / self._sample_rate_q
        # self._delta_per_iter = self._delta / (2 * self._noise_iter * self._sample_rate_q)
        self._delta_per_iter = (1 - ((1 - self._delta) / (1 - 1e-6))**(1/self._noise_iter)) / self._sample_rate_q

        self._clip_value = l2_clip_value 

        sp.info('IDN Noise Mechanism')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('sample_rate', self._sample_rate_q, 3) +
                sp.var_print('eps_per_iter', self._eps_per_iter) +
                sp.var_print('delta_per_iter', self._delta_per_iter)
                )

    def generate_noise(self, grad):
        shape = grad.get_shape().as_list()
        shape_mul = 1
        if len(shape)!=0:
            for i in range(len(shape)):
                shape_mul = shape_mul * shape[i]
        l2_sen = self._clip_value * np.sqrt(shape_mul)
        noise = udn.getIDNNoise(grad, l2_sen, self._eps_per_iter , self._delta_per_iter)
        
        return noise


class MVGMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta, batch_size=_batch_size,
                 l2_clip_value=_l2_clip_value, lot_size=_lot_size,
                 total_epochs=_total_epochs, total_num_examples=_total_num_examples):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        self._sample_rate_q = lot_size / total_num_examples

        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        self._eps_per_iter = self._epsilon / (self._sample_rate_q) / 2.0 /\
                             np.sqrt(self._noise_iter * np.log(np.e + self._epsilon / self._delta))

        self._delta_per_iter = self._delta / (2 * self._noise_iter * self._sample_rate_q)

        self._clip_value = l2_clip_value


    def generate_noise(self, grad):
        shape = grad.get_shape().as_list()
        shape_mul = 1
        if len(shape)!=0:
            for i in range(len(shape)):
                shape_mul = shape_mul * shape[i]
        l2_sen = self._clip_value * np.sqrt(shape_mul)
        noise = mvg.getNoise(grad, self._eps_per_iter, self._delta_per_iter, l2_sen)
        return noise



class MMMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value, batch_size=_batch_size, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        self._batch_size = batch_size

        # input sampling rate q
        self._sample_rate_q = lot_size / total_num_examples

        # total number of noise addition iterations
        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        # clip value per example
        self._clip_value = l2_clip_value

        # epsilon value for each iteration
        self._eps_per_iter = self._epsilon

        # delta value for each iteration
        self._delta_per_iter = self._delta / (2 * self._noise_iter)

        # Get Aplus, frankly speaking we should generate aplus according to m,
        # but it is so slow.So here we use the hard coding specificly serving
        # our FCNet model, if applying to other networks, it should be adjusted.
        # You should generate aplus in generate_noise function
        # use aplus = tf.cast(getAplus(m))
        self.aplus1 = tf.cast(getAplus(105), tf.float32)   ### Corresponding to the input dim, for Adult dataset, input dim = 105
        self.aplus2 = tf.cast(getAplus(12), tf.float32)   ### Corresponding to the middle dim, for this network, midlle dim =12


    def generate_noise(self, grad):
        try:
            m , n =grad.get_shape().as_list()
        except:
            m = grad.get_shape().as_list()[0]
            n = 1
        self._eps_per_iter = self._epsilon / (m  * n) / (2.0 * self._sample_rate_q) / 2.0 /\
                             np.sqrt(self._noise_iter * np.log(np.e + self._epsilon / self._delta))
        l2_sen = self._clip_value * np.sqrt(m*n)
        sigma = l2_sen / self._eps_per_iter * np.sqrt(2*np.log(1.25/self._delta_per_iter))
        noise = tf.random.normal(grad.get_shape(), 0.0, sigma)

        if m==105:
            aplus = self.aplus1
        elif m==12:
            aplus = self.aplus2

        try: ### for weight
            noise = tf.linalg.matmul(aplus, noise)
        except: ### for bias shape is [m]
            pass
        
        return noise



class UnperturbedMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta, \
                 l2_clip_value=_l2_clip_value, batch_size=_batch_size, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        # input sampling rate q
        self._sample_rate_q = batch_size / total_num_examples

        # total number of noise addition iterations
        # self._noise_iter = total_epochs / self._sample_rate_q

        # clip value per example
        # self._clip_value = l2_clip_value / lot_size
        self._clip_value = l2_clip_value

        sp.info('Unperturbed No Noise')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('clip', self._clip_value) +
                sp.var_print('sample_rate', self._sample_rate_q, 3)
                )
    # Return noise of the optimized mechanism to be added to the gradients.
    # Usage: called each time of generating noise.
    # parameters: grad --- tensor, gradients of a layer, does not need to be flattened.
    # return noise --- zero tensor, same shape as grad

    def generate_noise(self, grad):
        return tf.zeros_like(grad)

### Optimized Mechanism (w=1)
class OptimizedVer3Mechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value, batch_size=_batch_size, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        # input sampling rate q
        self._sample_rate_q = lot_size / total_num_examples

        # total number of noise addition iterations
        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        # clip value per example
        # self._clip_value = l2_clip_value / lot_size
        self._clip_value = l2_clip_value

        # epsilon value for each iteration
        self._eps_per_iter = self._epsilon / (2.0 * self._sample_rate_q) /\
                             np.sqrt(self._noise_iter * np.log(np.e + self._epsilon / self._delta))

        # delta value for each iteration
        self._delta_per_iter = self._delta / (2 * self._noise_iter * self._sample_rate_q) 

        # without composition
        self._epsilon_wc = self._epsilon / self._sample_rate_q
        self._delta_wc = self._delta / self._sample_rate_q

        sp.info('Optimized Special Mechanism')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('clip', self._clip_value) +
                sp.var_print('sample_rate', self._sample_rate_q, 3) +
                sp.var_print('eps_per_iter', self._eps_per_iter) +
                sp.var_print('delta_per_iter', self._delta_per_iter)
                )
    # Return noise of the optimized mechanism to be added to the gradients.
    # Usage: called each time of generating noise.
    # parameters: grad --- tensor, gradients of a layer, does not need to be flattened.
    # return noise --- tensor, the same shape as grad

    def generate_noise(self, grad):
        ## changed clip value
        '''shape = grad.get_shape().as_list()
        shape_mul = 1
        if len(shape) != 0:
            for i in range(len(shape)):
                shape_mul = shape_mul * shape[i]
        l2_sen = self._clip_value * np.sqrt(shape_mul)'''
        # flatten the gradients
        grad_array = tf.reshape(grad, [-1])
        sigma_array = opt3.generate_noise(grad_array,
                                    tf.constant(self._clip_value, dtype=tf.float32), # l2_sen
                                    tf.constant(self._epsilon_wc, dtype=tf.float32),
                                    tf.constant(self._delta_wc, dtype=tf.float32),
                                    tf.constant(self._noise_iter, dtype=tf.float32))
        # sigma_array = tf.reshape(sigma_array, tf.shape(grad))
        noise1 = tf.py_function(opt3.py_opt_noise, inp=[sigma_array], Tout=[tf.float32])
        noise = tf.reshape(noise1, tf.shape(grad))
        return noise


class OptimizedVer2Mechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value, batch_size=_batch_size, lot_size=_lot_size,
                 total_num_examples=_total_num_examples, total_epochs=_total_epochs):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta

        # input sampling rate q
        self._sample_rate_q = lot_size / total_num_examples

        # total number of noise addition iterations
        self._noise_iter = int(total_epochs * total_num_examples / lot_size)

        # clip value per example
        # self._clip_value = l2_clip_value / lot_size
        self._clip_value = l2_clip_value

        # epsilon value for each iteration
        self._eps_per_iter = self._epsilon / (2.0 * self._sample_rate_q) /\
                             np.sqrt(self._noise_iter * np.log(np.e + self._epsilon / self._delta))

        # delta value for each iteration
        self._delta_per_iter = self._delta / (2 * self._noise_iter * self._sample_rate_q) 

        # without composition
        self._epsilon_wc = self._epsilon / self._sample_rate_q
        self._delta_wc = self._delta / self._sample_rate_q

        sp.info('Optimized Normal Mechanism')
        sp.info(sp.var_print('epsilon', self._epsilon) +
                sp.var_print('delta', self._delta) +
                sp.var_print('clip', self._clip_value) +
                sp.var_print('sample_rate', self._sample_rate_q, 3) +
                sp.var_print('eps_per_iter', self._eps_per_iter) +
                sp.var_print('delta_per_iter', self._delta_per_iter)
                )
    # Return noise of the optimized mechanism to be added to the gradients.
    # Usage: called each time of generating noise.
    # parameters: grad --- tensor, gradients of a layer, does not need to be flattened.
    # return noise --- tensor, the same shape as grad

    def generate_noise(self, grad):

        # flatten the gradients
        grad_array = tf.reshape(grad, [-1])

        # sigma_array, _ = nog2.generate_noise(grad_array,
        #                             tf.constant(self._clip_value, dtype=tf.float32),
        #                             tf.constant(self._eps_per_iter, dtype=tf.float32),
        #                             tf.constant(self._delta_per_iter, dtype=tf.float32))
        sigma_array, _ = opt2.generate_noise(grad_array,
                                    tf.constant(self._clip_value, dtype=tf.float32),
                                    tf.constant(self._epsilon_wc, dtype=tf.float32),
                                    tf.constant(self._delta_wc, dtype=tf.float32),
                                    tf.constant(self._noise_iter, dtype=tf.float32))
        # sigma_array = tf.reshape(sigma_array, tf.shape(grad))
        noise1 = tf.py_function(opt2.py_opt_noise, inp=[sigma_array], Tout=[tf.float32])
        noise = tf.reshape(noise1, tf.shape(grad))
        return noise

if __name__ == '__main__':
    gau_mech = GaussianMechanism()
    unperturbed = UnperturbedMechanism()

