# noise_gen_wrapper.py
# The interface between the caller and noise_generator.

import os, math, time, sys
import tensorflow as tf
import numpy as np

import special_printer as sp
import noise_generator as ng


def define_noiseGen(net_info):
    ### decide by the nosie type
    if net_info['noise_type'] == 'Gaussian':
        noiseGen = ng.GaussianMechanism(epsilon=net_info['epsilon'],
                                        delta=net_info['delta'],
                                        total_epochs=net_info['train_epochs'],
                                        batch_size=net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'IMGM':
        noiseGen = ng.IMGMechanism(epsilon=net_info['epsilon'],
                                        delta=net_info['delta'],
                                        total_epochs=net_info['train_epochs'],
                                        batch_size=net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'Unperturbed':
        noiseGen = ng.UnperturbedMechanism(epsilon=net_info['epsilon'],
                                        delta=net_info['delta'],
                                        total_epochs=net_info['train_epochs'],
                                        batch_size=net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'UDN':
        noiseGen = ng.UDNMechanism(epsilon = net_info['epsilon'],
                                        delta = net_info['delta'],
                                        total_epochs = net_info['train_epochs'],
                                        batch_size = net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'MVG':
        noiseGen = ng.MVGMechanism(epsilon = net_info['epsilon'],
                                        delta = net_info['delta'],
                                        total_epochs = net_info['train_epochs'],
                                        batch_size = net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'MM':
        noiseGen = ng.MMMechanism(epsilon = net_info['epsilon'],
                                        delta = net_info['delta'],
                                        total_epochs = net_info['train_epochs'],
                                        batch_size = net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'Optimized_Special':
        noiseGen = ng.OptimizedVer3Mechanism(epsilon = net_info['epsilon'],
                                        delta = net_info['delta'],
                                        total_epochs = net_info['train_epochs'],
                                        batch_size = net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    elif net_info['noise_type'] == 'Optimized_Normal':
        noiseGen = ng.OptimizedVer2Mechanism(epsilon = net_info['epsilon'],
                                        delta = net_info['delta'],
                                        total_epochs = net_info['train_epochs'],
                                        batch_size = net_info['batch_size'],
                                        lot_size=net_info['lot_size'],
                                        l2_clip_value=net_info['clip_value'],
                                        total_num_examples = net_info['total_num_examples'])
    else:
        sp.warning('UNRECOGNIZED NOISE TYPE' + str(net_info['noise_type']))
        exit(-1)
    return noiseGen

