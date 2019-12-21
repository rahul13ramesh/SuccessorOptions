import os
import tensorflow as tf

from .layers import conv2d, linear, initializers
from .network import Network


class CNN(Network):
    def __init__(self, sess,
                 data_format,
                 channels,
                 observation_dims,
                 output_size,
                 trainable=True,
                 hidden_activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.constant_initializer(0.1),
                 value_hidden_sizes=[512],
                 advantage_hidden_sizes=[512],
                 network_output_type='dueling',
                 name='CNN',
                 successor=False):
        super(CNN, self).__init__(sess, name)

        if data_format == 'NHWC':
            self.inputs = tf.placeholder(
                'float32',
                [None] + observation_dims[0:2] + [channels],
                name='inputs')
        elif data_format == 'NCHW':
            self.inputs = tf.placeholder(
                'float32',
                [None, channels] + observation_dims[0:2],
                name='inputs')
        else:
            raise ValueError("unknown data_format : %s" % data_format)

        self.var = {}
        self.l0 = tf.div(self.inputs, 255.)

        with tf.variable_scope(name):
            self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(
                self.l0, 32, [8, 8],
                [4, 4], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l1_conv')

            self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(
                self.l1, 64, [4, 4],
                [4, 4], weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l2_conv')

            self.l3, self.var['l3_w'], self.var['l3_b'] = conv2d(
                self.l2, 64, [4, 4], [2, 2],
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l3_conv')

            self.l4, self.var['l4_w'], self.var['l4_b'] = conv2d(
                self.l3, 64, [3, 3], [1, 1],
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l4_conv')

            self.l5, self.var['l5_w'], self.var['l5_b'] = linear(
                self.l4, 512,
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l5_conv')

            if successor:
                self.phiVal, self.var['phi_w'], self.var['phi_b'] = linear(
                    self.l5, 128,
                    weights_initializer, biases_initializer,
                    hidden_activation_fn, data_format, name='phi_lin')

                self.l6, self.var['l6_w'], self.var['l6_b'] = linear(
                    self.phiVal, 128,
                    weights_initializer, biases_initializer,
                    hidden_activation_fn, data_format, name='l6_lin')

                self.l7, self.var['l7_w'], self.var['l7_b'] = linear(
                    self.l6, 128,
                    weights_initializer, biases_initializer,
                    hidden_activation_fn, data_format, name='l7_lin')

                #  Phi Reconstruction
                self.l62, self.var['l62_w'], self.var['l62_b'] = linear(
                    self.phiVal, 64,
                    weights_initializer, biases_initializer,
                    hidden_activation_fn, data_format, name='l62_lin')

                self.l63, self.var['l63_w'], self.var['l63_b'] = linear(
                    self.l62, 64,
                    weights_initializer, biases_initializer,
                    hidden_activation_fn, data_format, name='l63_lin')

                self.phiValre, self.var['l64_w'], self.var['l64_b'] = linear(
                    self.l63, 128,
                    weights_initializer, biases_initializer,
                    hidden_activation_fn, data_format, name='l64_lin')

                layer = self.l7
                output_size = 128
                network_output_type == 'normal'

            else:
                layer = self.l5

            self.build_output_ops(
                layer, network_output_type,
                value_hidden_sizes, advantage_hidden_sizes,
                output_size, weights_initializer,
                biases_initializer, hidden_activation_fn,
                output_activation_fn, trainable)

    def calc_phiVal(self, observation):
        return self.phiVal.eval({self.inputs: observation}, session=self.sess)
