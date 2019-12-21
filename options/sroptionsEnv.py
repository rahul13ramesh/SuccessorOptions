import gym
import pickle
import numpy as np
import tensorflow as tf
from qlearning.networks.mlp import MLPSmall
from qlearning.agents.deep_q import DeepSuccessor
from qlearning.env import FetchEnvironment

from gym.envs.registration import register


class SrOptionsWrapper_v0(gym.Env):
    """
    Wrapper for modified reward using SR-options
    """
    def __init__(self, visualize=False):
        optionNum = 1

        GAME = 'FetchReach-v1'
        self.fenv = gym.make(GAME)

        path = "data/srCenter.pkl"
        f = open(path, "wb")
        centerSR = pickle.load(f)
        self.SR = np.array(centerSR[optionNum])
        f.close()
        self.sess = tf.Session()

        pred_network = MLPSmall(sess=self.sess, output_size=10,
                                name='pred_network', trainable=True,
                                inputSize=10)

        target_network = MLPSmall(sess=self.sess, output_size=10,
                                  name='target_network', trainable=False,
                                  inputSize=10)
        env = FetchEnvironment()
        testEnv = FetchEnvironment()

        self.agent = DeepSuccessor(self.sess, pred_network, env, testEnv,
                                   None, target_network=target_network)
        self.agent.restoreCkpt("checkpoints/4999999_model.ckpt")

    @property
    def observation_space(self):
        return self.fenv.observation_space

    @property
    def action_space(self):
        return self.fenv.action_space

    def step(self, action):
        ob, _, _, _ = self.fenv.step(action)
        self.steps += 1

        done = False
        ob = ob['observation']
        if self.steps >= 1000:
            done = False
        curPhi = self.agent.pred_network.calc_phi([ob])[0]
        reward = np.dot(self.SR, (curPhi - self.prevPhi))
        self.prevPhi = np.array(curPhi)
        return ob, reward, done, {}

    def reset(self):
        self.steps = 0
        obs = self.fenv.reset()
        obs = obs['observation']
        self.prevPhi = self.agent.pred_network.calc_phi([obs])[0]
        return np.array(obs)

    def render(self):
        self.fenv.render()


register(
    id='SrOption-v0',
    entry_point='sroptionsEnv:SrOptionsWrapper_v0',
)
