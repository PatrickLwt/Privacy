import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras import Model, regularizers, Sequential


class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.weight_decay = 0.0005

        # Block 1
        self.conv1_1 = Conv2D(64, 3, activation='relu', padding='same', input_shape=[32, 32, 3],
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block1_conv1', trainable=False)
        self.bn1_1 = BatchNormalization()
        self.drop1 = Dropout(0.3)
        self.conv1_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(
                              self.weight_decay), name='block1_conv2', trainable=False)
        self.bn1_2 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(2, 2))

        # Block 2
        self.conv2_1 = Conv2D(128, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block2_conv1', trainable=False)
        self.bn2_1 = BatchNormalization()
        self.drop2 = Dropout(0.4)
        self.conv2_2 = Conv2D(128, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block2_conv2', trainable=False)
        self.bn2_2 = BatchNormalization()
        self.pool2 = MaxPool2D(pool_size=(2, 2))

        # Block 3
        self.conv3_1 = Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv1', trainable=False)
        self.bn3_1 = BatchNormalization()
        self.drop3 = Dropout(0.4)
        self.conv3_2 = Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv2', trainable=False)
        self.bn3_2 = BatchNormalization()
        self.drop4 = Dropout(0.4)
        self.conv3_3 = Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv3', trainable=False)
        self.bn3_3 = BatchNormalization()
        self.pool3 = MaxPool2D(pool_size=(2, 2))

        # Block 4
        self.conv4_1 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv1', trainable=False)
        self.bn4_1 = BatchNormalization()
        self.drop5 = Dropout(0.4)
        self.conv4_2 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv2', trainable=False)
        self.bn4_2 = BatchNormalization()
        self.drop6 = Dropout(0.4)
        self.conv4_3 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv3', trainable=False)
        self.bn4_3 = BatchNormalization()
        self.pool4 = MaxPool2D(pool_size=(2, 2))

        # Block 5
        self.conv5_1 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv1')
        self.bn5_1 = BatchNormalization()
        self.drop7 = Dropout(0.4)
        self.conv5_2 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv2')
        self.bn5_2 = BatchNormalization()
        self.drop8 = Dropout(0.4)
        self.conv5_3 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv3')
        self.bn5_3 = BatchNormalization()
        self.pool5 = MaxPool2D(pool_size=(2, 2))
        self.drop9 = Dropout(0.5)

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(
            self.weight_decay), name='fc1')
        self.bn1 = BatchNormalization()
        self.drop10 = Dropout(0.5)
        self.d2 = Dense(10, activation='softmax', name='fc2')

    def call(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.drop1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.drop2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.drop3(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.drop4(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.drop5(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.drop6(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.drop7(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.drop8(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.pool5(x)
        x = self.drop9(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn1(x)
        x = self.drop10(x)
        x = self.d2(x)
        return x

    def build_model(self):
        x = Input(shape=(32, 32, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        self.accumulated_grads = [tf.Variable(tf.zeros_like(
            var), trainable=False) for var in self.trainable_variables]
        return model



class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(32, 5, activation='relu', padding='same', input_shape=[28, 28, 1], name='conv1')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(2, 2))

        self.conv2 = Conv2D(64, 5, activation='relu', padding='same', name='conv2')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPool2D(pool_size=(2,2))

        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu', name='fc1')
        self.bn3 = BatchNormalization()
        self.d2 = Dense(10, activation='softmax', name='fc2')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn3(x)
        x = self.d2(x)
        return x

    def build_model(self):
        x = Input(shape=(28, 28, 1))
        model = Model(inputs=[x], outputs=self.call(x))
        self.accumulated_grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
        return model 

