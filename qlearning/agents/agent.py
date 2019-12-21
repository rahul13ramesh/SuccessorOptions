import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import logging
from copy import deepcopy

from .experience import Experience

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s]: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class Agent(object):
    def __init__(self, sess, pred_network, env, testEnv,
                 conf, target_network=None):
        self.sess = sess

        self.ep_start = conf.ep_start
        self.ep_end = conf.ep_end
        self.history_length = 1
        self.t_ep_end = conf.t_ep_end
        self.t_learn_start = conf.t_learn_start
        self.t_train_freq = conf.t_train_freq
        self.t_target_q_update_freq = conf.t_target_q_update_freq

        self.discount_r = conf.discount_r
        self.min_r = conf.min_r
        self.max_r = conf.max_r
        self.min_delta = conf.min_delta
        self.max_delta = conf.max_delta
        self.max_grad_norm = conf.max_grad_norm
        self.observation_dims = conf.observation_dims

        self.learning_rate = conf.learning_rate
        self.learning_rate_minimum = conf.learning_rate_minimum
        self.learning_rate_decay = conf.learning_rate_decay
        self.learning_rate_decay_step = conf.learning_rate_decay_step

        # network
        self.pred_network = pred_network
        self.target_network = target_network
        self.target_network.create_copy_op(self.pred_network)

        self.env = env
        self.testEnv = testEnv
        self.experience = Experience(conf.batch_size, 4,
                                     conf.memory_size, conf.observation_dims)

        self.saver = tf.train.Saver(max_to_keep=5)

        with tf.variable_scope('t'):
            self.t_op = tf.Variable(0, trainable=False, name='t')
            self.t_add_op = self.t_op.assign_add(1)

        if conf.random_start:
            self.new_game = self.env.new_random_game
        else:
            self.new_game = self.env.new_game

    def train(self, t_max):
        tf.global_variables_initializer().run()

        self.target_network.run_copy()

        start_t = self.t_op.eval(session=self.sess)
        observation, reward, terminal = self.new_game()

        startTime = time.time()

        STEPLOGSIZE = 10000
        for self.t in range(start_t, t_max):
            ep = (self.ep_end +
                  max(0., (self.ep_start - self.ep_end) * (self.t_ep_end - max(0., self.t - self.t_learn_start)) / self.t_ep_end))

            self.t_add_op.eval(session=self.sess)
            if self.t % STEPLOGSIZE == 0:
                diff = time.time() - startTime
                logger.info("At Step " + str(self.t) +
                            " : "+str("%.4f" % (float(STEPLOGSIZE)/(diff))) +
                            " steps/sec")
                startTime = time.time()

            if self.t % 100000 == 0 or self.t == t_max - 1:

                logger.info("SAVING MODEL")
                self.saver.save(self.sess, "./checkpoints/" +
                                str(self.t) + "_model.ckpt")
                logger.info("MODEL SAVED")
                totReturn = 0
                allEps = []


            # 1. predict
            action = self.predict(observation, ep)
            # 2. act
            observation, reward, terminal, info = self.env.step(
                action, is_training=True)
            # 3. observe
            q, loss, is_update = self.observe(
                observation, reward, action, terminal)

            if terminal:
                observation, reward, terminal = self.new_game()

    def predict(self, s_t, ep):
        if random.random() < ep:
            action = self.env.env.action_space.sample()
        else:
            action = self.pred_network.calc_actions([s_t])[0]
        return action

    def q_learning_minibatch_test(self):
        s_t = np.array([[[0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [1., 0., 0., 0.]]], dtype=np.uint8)
        s_t_plus_1 = np.array([[[0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [1., 0., 0., 0.],
                                [0., 0., 0., 0.]]], dtype=np.uint8)
        s_t = s_t.reshape([1, 1] + self.observation_dims)
        s_t_plus_1 = s_t_plus_1.reshape([1, 1] + self.observation_dims)

        action = [3]
        reward = [1]
        terminal = [0]

        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
        target_q_t = (1. - terminal) * self.discount_r * \
            max_q_t_plus_1 + reward

        _, q_t, a, loss = self.sess.run([
            self.optim, self.pred_network.outputs, self.pred_network.actions, self.loss
        ], {
            self.targets: target_q_t,
            self.actions: action,
            self.pred_network.inputs: s_t,
        })

        logger.info("q: %s, a: %d, l: %.2f" % (q_t, a, loss))

    def update_target_q_network(self):
        assert self.target_network is not None
        self.target_network.run_copy()
