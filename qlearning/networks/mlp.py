import tensorflow as tf

from .layers import initializers, linear
from .network import Network


class MLPSmall(Network):
    def __init__(self, sess,
                 trainable=True,
                 batch_size=None,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.zeros_initializer,
                 output_size=10, hidden_activation_fn=tf.tanh,
                 output_activation_fn=None, hidden_sizes=[30, 10],
                 network_output_type='dueling', inputSize=10,
                 name='MLPSmall'):
        super(MLPSmall, self).__init__(sess, name)

        with tf.variable_scope(name):
            layer = self.inputs = tf.placeholder('float32', [batch_size, inputSize])
            #  layer = tf.reshape(layer, [-1] + layer.get_shape().as_list()[2:])
            print(layer.get_shape())

            for idx, hidden_size in enumerate(hidden_sizes):
                w_name, b_name = 'w_%d' % idx, 'b_%d' % idx

                layer, self.var[w_name], self.var[b_name] = \
                    linear(layer, hidden_size, weights_initializer,
                           biases_initializer, hidden_activation_fn,
                           trainable, name='lin_%d' % idx)

            self.phi = layer
            self.build_output_ops(layer, network_output_type,
                                  output_size, weights_initializer,
                                  biases_initializer, hidden_activation_fn,
                                  output_activation_fn, trainable)
