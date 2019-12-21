#  from sroptionsEnv import SrOptionsWrapper_v0
import gym

env = gym.make("SrOption-v0")
state = env.reset()
print(state)

while True:
    done = False
    steps = 0
    while not done:
        steps += 1
        obs, rew, done, _ = env.step(env.action_space.sample())
        env.render()
        print(rew, done, obs)
    print("steps" + str(steps))
    env.reset()
