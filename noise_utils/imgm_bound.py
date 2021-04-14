# gaussian.py
# calculate mu for gaussian noise N(0,mu^2)

import numpy as np
from math import erf


def calculate_our_mu(epsilon, delta, l2_sensitivity):
    return l2_sensitivity * (np.sqrt(-2 * np.log(delta / 2)) + np.sqrt(-2 * np.log(delta / 2) - 2 * epsilon)) / \
           (2 * epsilon)


def calculate_gaussian_noise_stddev(epsilon, delta, l2_sensitivity, T, q):
    '''
    calculate mu for gaussian noise N(0,mu^2) under (eps,delta)-dp
    according to https://github.com/IBM/differential-privacy-library/blob/master/diffprivlib/mechanisms/gaussian.py
    Paper link: https://arxiv.org/pdf/1805.06530.pdf https://arxiv.org/pdf/1907.02444v1.pdf
    :param eps: dp parameter
    :param delta: dp parameter
    :param l2_sensitivity:
    :param T the numbers of training epochs
    :param q the sample rate
    :return: mu
    '''

    def phi(x):
        return (1 + erf(x / np.sqrt(2))) / 2

    def b_plus(x):
        return phi(np.sqrt(epsilon * x)) - np.exp(epsilon) * phi(- np.sqrt(epsilon * (x + 2))) - delta

    def b_minus(x):
        return phi(- np.sqrt(epsilon * x)) - np.exp(epsilon) * phi(- np.sqrt(epsilon * (x + 2))) - delta

    delta_0 = b_plus(0)

    if delta_0 == 0:
        alpha = 1
    else:
        if delta_0 < 0:
            target_func = b_plus
        else:
            target_func = b_minus

        # Find the starting interval by doubling the initial size until the target_func sign changes, as suggested
        # in the paper
        left = 0
        right = 1

        while target_func(left) * target_func(right) > 0:
            left = right
            right *= 2

        # Binary search code copied from mechanisms.LaplaceBoundedDomain
        old_interval_size = (right - left) * 2

        while old_interval_size > right - left:
            old_interval_size = right - left
            middle = (right + left) / 2

            if target_func(middle) * target_func(left) <= 0:
                right = middle
            if target_func(middle) * target_func(right) <= 0:
                left = middle

        alpha = np.sqrt(1 + (left + right) / 4) + (-1 if delta_0 < 0 else 1) * np.sqrt((left + right) / 4)

    return alpha * q * np.sqrt(T) * l2_sensitivity / np.sqrt(2 * epsilon)


def calsigma_Raw(epsilon, delta, l2_sensitivity, T, q):
    sigma = 2.0 * q * l2_sensitivity / epsilon * np.sqrt(T * np.log(1.0 / delta))
    return sigma

def strong_composition1(epsilon, delta, l2_sensitivity, T, q):
    eps_per_iter = epsilon / (q * 2.0 * np.sqrt(T * np.log(np.e + epsilon / delta)))
    delta_per_iter = delta / (2 * T * q)
    sigma = calculate_gaussian_noise_stddev(eps_per_iter, delta_per_iter, l2_sensitivity, 1, q) / q
    return sigma

def strong_composition2(epsilon, delta, l2_sensitivity, T, q):
    eps_per_iter = epsilon / (q * 2.0 * np.sqrt(T * np.log(np.e + epsilon / delta)))
    delta_per_iter = delta / (2 * T * q)
    sigma = np.sqrt(2*np.log(1.25/delta_per_iter))*l2_sensitivity/eps_per_iter
    return sigma
