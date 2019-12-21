
import random
import numpy as np


class Experience(object):
    def __init__(self, batch_size, history_length, memory_size, observation_dims):
        self.batch_size = batch_size
        self.history_length = history_length
        self.memory_size = memory_size
        self.observation_dims = observation_dims

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.observations = np.empty(
            [self.memory_size] + observation_dims, dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty(
            [self.batch_size] + observation_dims, dtype=np.float16)
        self.poststates = np.empty(
            [self.batch_size] + observation_dims, dtype=np.float16)

        self.count = 0
        self.current = 0

    def add(self, observation, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.observations[self.current, ...] = observation
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if self.terminals[index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.retreive(index - 1)
            self.poststates[len(indexes), ...] = self.retreive(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals

    def retreive(self, index):
        index = index % self.count
        retObs = self.observations[index, ...]
        return retObs
