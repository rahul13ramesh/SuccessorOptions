from sroptionsEnv import SrOptionsWrapper_v0
import gym
import time
import argparse


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import DDPG, PPO2
from stable_baselines.common.vec_env import DummyVecEnv

parser = argparse.ArgumentParser()
parser.add_argument("option", help="option to use", type=int)
args = parser.parse_args()

option = int(args.option)
TOTAL = 10000
env = gym.make("SrOption-v" + str(option))
env = DummyVecEnv([lambda: env])

#  model = DDPG(MlpPolicy, env)

model = PPO2(
    MlpPolicy, env, learning_rate=3e-4, ent_coef=0.0,
    n_steps=2048, gamma=0.99, noptepochs=10,
    nminibatches=int(2048/64), verbose=1)

model = PPO2(MlpPolicy, env, learning_rate=3e-4, ent_coef=0.0)


model.learn(total_timesteps=int(TOTAL), log_interval=200, seed=100)


print("start recording")
done = False
obs = env.reset()[0]
env.render()
time.sleep(2)
while True:
    action = model.predict(obs)
    obs, reward, done, _ = env.step(action[0])
    env.render()
    time.sleep(0.1)
