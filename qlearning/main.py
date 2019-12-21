#!/bin/bash python3
import random
import tensorflow as tf 
from networks.mlp import MLPSmall 
from agents.deep_q import DeepSuccessor
from env import FetchEnvironment

flags = tf.app.flags

# GPU
flags.DEFINE_boolean(
    'use_gpu', False,
    'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')

#  DQN
flags.DEFINE_boolean(
    'successor_learn', False,
    'Whether to do training or testing')

flags.DEFINE_boolean(
    'successor_collect', False,
    'Whether to do training or testing')

# Environment
flags.DEFINE_string(
    'observation_dims', "[10]",
    'Observation dimension')
flags.DEFINE_integer(
    'max_r', +10,
    'The maximum value of clipped reward')
flags.DEFINE_integer(
    'min_r', -10,
    'The minimum value of clipped reward')
flags.DEFINE_boolean(
    'random_start', True,
    'Whether to start with random state')
flags.DEFINE_boolean(
    'use_cumulated_reward', False,
    'Whether to use cumulated reward or not')

# Training
flags.DEFINE_integer(
    'max_delta', None,
    'The maximum value of delta')
flags.DEFINE_integer(
    'min_delta', None,
    'The minimum value of delta')
flags.DEFINE_float(
    'ep_start', 1.,
    'The value of epsilon at start in e-greedy')
flags.DEFINE_float(
    'ep_end', 0.01,
    'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer(
    'batch_size', 8,
    'The size of batch for minibatch training')
flags.DEFINE_integer(
    'max_grad_norm', None,
    'The maximum norm of gradient while updating')
flags.DEFINE_float('discount_r', 0.99, 'The discount factor for reward')

# Timer
flags.DEFINE_integer(
    't_train_freq', 4, '')

# Below numbers will be multiplied by scale
flags.DEFINE_integer(
    'scale', 1000,
    'The scale for big numbers')
flags.DEFINE_integer(
    'memory_size', 20,
    'The size of experience memory (*= scale)')
flags.DEFINE_integer(
    't_target_q_update_freq', 1,
    'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer(
    't_test', 1,
    'The maximum number of t while training (*= scale)')
flags.DEFINE_integer(
    't_ep_end', 100,
    'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer(
    't_train_max', 5000,
    'The maximum number of t while training (*= scale)')
flags.DEFINE_float(
    't_learn_start', 5,
    'The time when to begin training (*= scale)')
flags.DEFINE_float(
    'learning_rate_decay_step', 5,
    'The learning rate of training (*= scale)')

# Optimizer
flags.DEFINE_float(
    'learning_rate', 0.001,
    'The learning rate of training')
flags.DEFINE_float(
    'learning_rate_minimum', 0.00025,
    'The minimum learning rate of training')
flags.DEFINE_float(
    'learning_rate_decay', 0.96,
    'The decay of learning rate of training')
flags.DEFINE_float(
    'decay', 0.99,
    'Decay of RMSProp optimizer')
flags.DEFINE_float(
    'momentum', 0.0,
    'Momentum of RMSProp optimizer')
flags.DEFINE_float(
    'gamma', 0.99,
    'Discount factor of return')
flags.DEFINE_float(
    'beta', 0.01,
    'Beta of RMSProp optimizer')

# Debug
flags.DEFINE_boolean(
    'display', False,
    'Whether to do display the game screen or not')
flags.DEFINE_integer(
    'random_seed', 123,
    'Value of random seed')
flags.DEFINE_string(
    'tag', '',
    'The name of tag for a model, only for debugging')
flags.DEFINE_boolean(
    'allow_soft_placement', True,
    'Whether to use part or all of a GPU')

# Internal
conf = flags.FLAGS

# set random seed
tf.set_random_seed(conf.random_seed)
random.seed(conf.random_seed)


def main(_):
    numAction = 3
    for flag in ['memory_size', 't_target_q_update_freq', 't_test',
                 't_ep_end', 't_train_max', 't_learn_start',
                 'learning_rate_decay_step']:
        setattr(conf, flag, getattr(conf, flag) * conf.scale)

    #  Allow soft placement to occupy as much GPU as needed
    sess_config = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.intra_op_parallelism_threads = 8
    conf.observation_dims = eval(conf.observation_dims)

    with tf.Session(config=sess_config) as sess:
        env = FetchEnvironment()
        testEnv = FetchEnvironment()

        pred_network = MLPSmall(sess=sess, output_size=10,
                                name='pred_network', trainable=True,
                                inputSize=10)
        target_network = MLPSmall(sess=sess, output_size=10,
                                  name='target_network', trainable=False,
                                  inputSize=10)

        if conf.successor_learn:
            agent = DeepSuccessor(sess, pred_network, env, testEnv,
                                  conf, target_network=target_network)
            agent.train(conf.t_train_max)
        elif conf.successor_collect:
            agent = DeepSuccessor(sess, pred_network, env, testEnv,
                                  conf, target_network=target_network)
            agent.collectSR(300000, "checkpoints/4999999_model.ckpt")


if __name__ == '__main__':
    tf.app.run()
