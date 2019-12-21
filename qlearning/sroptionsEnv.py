import gym
import pickle
import numpy as np
import tensorflow as tf

from networks.mlp import MLPSmall
from agents.deep_q import DeepSuccessor
from env import FetchEnvironment

from gym.envs.registration import register


class SrOptionsWrapper_v0(gym.Env):
    """
    Wrapper for modified reward using SR-options
    """
    def __init__(self, visualize=False, optionNum=1):

        GAME = 'FetchReach-v1'
        self.steps = 0
        self.fenv = gym.make(GAME)

        path = "../data/srCenter.pkl"
        f = open(path, "rb")
        centerSR = pickle.load(f)
        self.SR = np.array(centerSR[optionNum])
        f.close()

        self.sess = tf.Session()

        self.pred_network = MLPSmall(
            sess=self.sess, output_size=10, name='pred_network',
            trainable=True, inputSize=10)
        self.saver = tf.train.Saver(max_to_keep=5)

        #  target_network = MLPSmall(sess=self.sess, output_size=10,
                                  #  name='target_network', trainable=False,
                                  #  inputSize=10)
        #  env = FetchEnvironment()
        #  testEnv = FetchEnvironment()

        #  self.agent = DeepSuccessor(self.sess, pred_network, env, testEnv,
                                   #  None, target_network=target_network)
        restorePath = "../checkpoints/4999999_model.ckpt"
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, restorePath)

    @property
    def observation_space(self):
        #  return self.fenv.observation_space
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=([10]), dtype='uint8')

    @property
    def action_space(self):
        return self.fenv.action_space

    def step(self, action):
        ob, _, _, _ = self.fenv.step(action)
        self.steps += 1

        done = False
        ob = ob['observation']
        if self.steps >= 1000:
            done = True
        elif ob[0] <= 1.05 or ob[0] >= 1.510:
            done = True
        elif ob[1] <= 0.398 or ob[1] >= 1.1:
            done = True
        elif ob[2] >= 0.875:
            done = True
        curPhi = self.pred_network.calc_phi([ob])[0]
        reward = np.dot(self.SR, (curPhi - self.prevPhi))
        reward = reward/10.0
        reward = np.clip(reward, -1.0, 1.0)
        self.prevPhi = np.array(curPhi)
        return ob, reward, done, {}

    def reset(self):
        self.steps = 0
        obs = self.fenv.reset()
        obs = obs['observation']
        self.prevPhi = self.pred_network.calc_phi([obs])[0]
        return np.array(obs)

    def render(self):
        self.fenv.render()


register(
    id='SrOption-v0',
    entry_point='sroptionsEnv:SrOptionsWrapper_v0',
    kwargs={'optionNum': 0}
)

register(
    id='SrOption-v1',
    entry_point='sroptionsEnv:SrOptionsWrapper_v0',
    kwargs={'optionNum': 1}
)

register(
    id='SrOption-v2',
    entry_point='sroptionsEnv:SrOptionsWrapper_v0',
    kwargs={'optionNum': 2}
)

register(
    id='SrOption-v3',
    entry_point='sroptionsEnv:SrOptionsWrapper_v0',
    kwargs={'optionNum': 3}
)

register(
    id='SrOption-v4',
    entry_point='sroptionsEnv:SrOptionsWrapper_v0',
    kwargs={'optionNum': 4}
)
