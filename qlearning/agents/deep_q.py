import os
import time
import numpy as np
import tensorflow as tf
import pickle
from logging import getLogger

from .agent import Agent

logger = getLogger(__name__)


class DeepSuccessor(Agent):
    def __init__(self, sess, pred_network, env, testEnv,
                 conf, target_network=None):
        super(DeepSuccessor, self).__init__(sess, pred_network, env, testEnv,
                                            conf, target_network=target_network)

        # Optimizer
        with tf.variable_scope('optimizer'):
            self.targets = tf.placeholder(
                'float32', [None, 10], name='target_sr_t')

            self.delta = tf.reduce_mean(
                tf.square(self.targets - self.pred_network.outputs),
                axis=1)

            self.clipped_error = tf.where(self.delta < 1.0,
                                          self.delta,
                                          tf.sqrt(self.delta) - 0.5, name='clipped_error')

            self.loss = tf.reduce_mean(self.clipped_error, name='loss') 

            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.t_op,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))

            optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01)

            self.optim = optimizer.minimize(self.loss)

    def observe(self, observation, reward, action, terminal):
        reward = max(self.min_r, min(self.max_r, reward))

        self.experience.add(observation, reward, action, terminal)
        result = [], 0, False

        if self.t > self.t_learn_start:
            if self.t % self.t_train_freq == 0:
                result = self.q_learning_minibatch()

            if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
                self.update_target_q_network()

        return result

    def q_learning_minibatch(self):
        if self.experience.count < 4:
            return [], 0, False
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

        terminal = np.array(terminal) + 0.

        srOutputs = self.target_network.calc_outputs(s_t_plus_1)
        phiVal = self.target_network.calc_phi(s_t_plus_1)
        terminal = np.expand_dims(terminal, 1)

        target_sr = (1. - terminal) * self.discount_r * srOutputs + phiVal

        _, srVal, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
            self.targets: target_sr,
            self.pred_network.inputs: s_t,
        })

        return srVal, loss, True

    def train(self, t_max):
        tf.global_variables_initializer().run()

        start_t = self.t_op.eval(session=self.sess)
        observation, reward, terminal = self.new_game()

        startTime = time.time()

        STEPLOGSIZE = 10000
        for self.t in range(start_t, t_max):
            ep = 1

            self.t_add_op.eval(session=self.sess)

            if self.t % STEPLOGSIZE == 0:
                diff = time.time() - startTime
                logger.info("At Step " + str(self.t) +
                            " : " + str("%.4f" % (float(STEPLOGSIZE) / (diff))) +
                            " steps/sec")
                startTime = time.time()

            if self.t % 100000 == 0 or self.t == t_max - 1:

                logger.info("SAVING MODEL")
                self.saver.save(self.sess, "./checkpoints/" +
                                str(self.t) + "_model.ckpt")
                logger.info("MODEL SAVED")

            # 1. predict
            action = self.predict(observation, ep)
            # 2. act
            observation, reward, terminal, info = self.env.step(
                action, is_training=True)
            # 3. observe
            sr, loss, is_update = self.observe(
                observation, reward, action, terminal)

            if terminal:
                observation, reward, terminal = self.new_game()

    def restoreCkpt(self, restorePath):
        tf.global_variables_initializer().run()
        self.saver.restore(self.sess, restorePath)

    def collectSR(self, numSR, restorePath):
        tf.global_variables_initializer().run()

        self.saver.restore(self.sess, restorePath)
        countSR = 0
        observation, reward, terminal = self.new_game()

        srMatrix = []
        stateMatrix = []
        steps = 0
        while countSR < numSR:
            print(countSR)
            ep = 1
            steps += 1
            countSR += 1
            print(countSR)

            self.t_add_op.eval(session=self.sess)

            action = self.predict(observation, ep)
            # 2. act
            observation, reward, terminal, info = self.env.step(
                action, is_training=False)

            ot = self.pred_network.calc_outputs([observation])
            srMatrix.append(ot[0])
            stateMatrix.append(observation)

            if terminal or steps >= 3000:
                steps = 0
                observation, reward, terminal = self.new_game()

        f = open("data/srVals3.pkl", "wb")
        pickle.dump((srMatrix, stateMatrix), f)
        f.close()
