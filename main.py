import tensorflow as tf
import numpy as np
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import noise_gen_wrapper as ngw
import generate_argparse as ga
import data_generator as dg 
import special_printer as sp 
from network import VGG16, LeNet

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PARAMS = {
    'mnist':{'noise_type': 'Unperturbed', 'epsilon': 10.0, 'delta': 0.00001, 'clip_value': 1.0,
            'train_epochs': 60, 'batch_size': 128, 'lot_size': 128, 'total_num_examples': 60000, 'load_model': False,
            'load_path': 'pretrain_model/pretrained_lenet.h5',
            'net_name': 'LeNet', 'augmentation': False, 'init_lr': 0.1, 'lr_decay_step': 215, 'lr_decay_rate': 0.99,},
    'cifar-10':{'noise_type': 'Gaussian', 'epsilon': 10.0, 'delta': 0.00001, 'clip_value': 5.0,
            'train_epochs': 60, 'batch_size': 64, 'lot_size': 256, 'total_num_examples': 50000, 'load_model': True,
            'load_path': 'pretrain_model/pretrained_vgg16.h5',
            'net_name': 'VGG16', 'augmentation': True, 'init_lr': 0.1, 'lr_decay_step': 4000, 'lr_decay_rate': 0.5}
}

dataset_name = 'mnist'
net_info = {
    'dataset': dataset_name,  # 'mnist', 'cifar-10'
    'noise_type': PARAMS[dataset_name]['noise_type'],  # Gaussian, UDN, Optimized, 
    'epsilon': PARAMS[dataset_name]['epsilon'],
    'delta': PARAMS[dataset_name]['delta'],
    'clip_value': PARAMS[dataset_name]['clip_value'],  # Clip by global norm
    'train_epochs': PARAMS[dataset_name]['train_epochs'],
    'batch_size': PARAMS[dataset_name]['batch_size'],
    'lot_size': PARAMS[dataset_name]['lot_size'],
    'total_num_examples': PARAMS[dataset_name]['total_num_examples'],
    'load_model': PARAMS[dataset_name]['load_model'],  # Whether to load pretrain model
    'load_path': PARAMS[dataset_name]['load_path'],
    'net_name': PARAMS[dataset_name]['net_name'],
    'augmentation': PARAMS[dataset_name]['augmentation'],  # Whether to do data augmentation
    'init_lr': PARAMS[dataset_name]['init_lr'],
    'lr_decay_step': PARAMS[dataset_name]['lr_decay_step'],
    'lr_decay_rate': PARAMS[dataset_name]['lr_decay_rate']
}

ga.generate_then_parse_arguments(net_info)


### Way to Accelerate using vectorized_map
# def clip_gradients_vmap(g, l2_norm_clip):
#     ### Clips gradients in a way that is compatible with vectorized_map.
#     grads_flat = tf.nest.flatten(g)
#     squared_l2_norms = [tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat]
#     global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
#     div = tf.maximum(global_norm / l2_norm_clip, 1.)
#     clipped_flat = [g / div for g in grads_flat]
#     clipped_grads = tf.nest.pack_sequence_as(g, clipped_flat)
#     return clipped_grads


class OurModel(globals()[net_info['net_name']]):
    def __init__(self, net_info):
        super(OurModel, self).__init__()
        self.clip_value = net_info['clip_value']
        self.batch_size = net_info['batch_size']
        self.batch_per_lot = int(net_info['lot_size'] / net_info['batch_size'])
        self.noise_gen = ngw.define_noiseGen(net_info)
        self.apply_flag = tf.Variable(False, dtype=tf.bool, trainable=False)


    # def special_compile(self, custom_optimizer, custom_loss, custom_metric):
    #     self.custom_optimizer = custom_optimizer
    #     self.custom_loss = custom_loss
    #     self.custom_metric = custom_metric
    #     self.custom_loss_mean = tf.keras.metrics.Mean(custom_loss.name)
    #     self.custom_metric_mean = tf.keras.metrics.Mean(custom_metric.name)
    #     super().compile()

    # def compile(self, **kwargs):
    #     raise NotImplementedError("Please use special_compile()")

    def clip_gradients(self, g):
        return tf.clip_by_global_norm(g, self.clip_value)[0]

    def reduce_noise_normalize_batch(self, g):
        summed_gradient = tf.reduce_sum(g, axis=0)
        noise = self.noise_gen.generate_noise(summed_gradient)
        noised_gradient = tf.add(summed_gradient, noise)

        return tf.truediv(noised_gradient, self.batch_size)


    def apply(self):
        gradients = [g / self.batch_per_lot for g in self.accumulated_grads]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        for g in self.accumulated_grads:
            g.assign(tf.zeros_like(g))
        return

    def not_apply(self):
        return

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.compiled_loss(labels, predictions)
            # add the regularization of loss
            regularization = sum(self.losses)
            loss += regularization

        jacobian = tape.jacobian(loss, self.trainable_variables)

        ### Accelerating
        # clipped_gradients = tf.vectorized_map(
        #     lambda g: clip_gradients_vmap(g, self.clip_value), jacobian)

        clipped_gradients = tf.map_fn(self.clip_gradients, jacobian)
        noised_gradients = tf.nest.map_structure(self.reduce_noise_normalize_batch, clipped_gradients)

        for g, new_grad in zip(self.accumulated_grads, noised_gradients):
            g.assign_add(new_grad)
        
        tf.cond(self.apply_flag, lambda: self.apply(), lambda: self.not_apply())
        # gradients = tape.gradient(loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self(images, training=False)
        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.x_test, self.y_test
        loss, acc = self.model.evaluate(x, y, batch_size=256, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    def on_train_batch_begin(self, batch, logs=None):
        if (batch+1) % self.model.batch_per_lot == 0:
            self.model.apply_flag.assign(True)
        else:
            self.model.apply_flag.assign(False)
    #     print('\nStep: {}, Apply Flag: {}\n'.format(batch, self.model.apply_flag))


def main(net_info):
    x_train, y_train, x_test, y_test = dg.get_data(net_info)
    ### for cifar
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(net_info['init_lr'], net_info['lr_decay_step'], net_info['lr_decay_rate'], staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    loss_custom = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    # loss_custom = tf.keras.losses.CategoricalCrossentropy()

    model = OurModel(net_info)
    m = model.build_model()
    m.summary()

    # Load pretrain model
    model(x_test)
    if net_info['load_model']:
        print("Do Loading Model!!!")
        initial_weights = [layer.get_weights() for layer in model.layers]
        model.load_weights(net_info['load_path'], by_name=True)

        # Check whether loading the model successfully
        for layer, initial in zip(model.layers, initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'Checkpoint contained no weights for layer {layer.name}!')
            else:
                print(f'Loading weights for layer {layer.name}!')


    model.compile(optimizer=optimizer, loss=loss_custom, metrics=['accuracy'], run_eagerly = False)

    ### sampling lot with replacement
    # idx = np.array([])
    # # sample lots without replacement
    # for i in range(int(net_info['total_num_examples'] / net_info['lot_size'])):
    #     sample_idx = np.random.choice(
    #         net_info['total_num_examples'], net_info['lot_size'], replace=False)
    #     idx = np.hstack((idx, sample_idx))
    # idx = idx.astype(np.int16)

    if net_info['augmentation']:
        print("Do Augmentation!!!")
        datagen = ImageDataGenerator(
            rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        model.fit(datagen.flow(x_train, y_train, batch_size=net_info['batch_size'], shuffle=True), steps_per_epoch=int(len(x_train)
                / net_info['batch_size']), epochs=net_info['train_epochs'], callbacks=[TestCallback(x_test, y_test)])
    else:
        print("No Augmentation!!!")
        model.fit(x_train, y_train, batch_size=net_info['batch_size'], shuffle=True, steps_per_epoch=int(len(x_train)
                / net_info['batch_size']), epochs=net_info['train_epochs'], callbacks=[TestCallback(x_test, y_test)])


if __name__ == "__main__":
    main(net_info)