import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

from env.roomDim import roomArr
from env.roomSmallDim import roomSmallArr
from env.roomPend import roomPend
from env.roomDeepmind1 import roomDeep


class gridWorld1(object):

    def __init__(self, size, incremental=False):
        if size == 1:
            self.room = roomPend
        elif size == 2:
            self.room = roomSmallArr
        elif size == 3:
            self.room = roomArr
        else:
            self.room = roomDeep

        self.incremental = incremental
        self.start = [1, 1]
        self.steps = 0

        self.done = False
        self.count = 0
        self.actionSpace()
        self.height = len(self.room)
        self.width = len(self.room[0])

        self.reset()
        self.resetGoal()

    def reset(self):
        if self.incremental:
            self.pos = [1, 1]
            self.steps = 0
        else:
            self.pos = self.start
        self.count = 0
        self.done = False
        return self.pos

    def resetGoal(self):
        while True:
            x = np.random.randint(self.height-1) + 1
            y = np.random.randint(self.width-1) + 1
            if self.room[x][y] == 0 and self.start != [x, y]:
                break

        self.goal = [x, y]

    def resetStart(self, leftMost=False):
        while True:
            x = np.random.randint(self.height-1) + 1
            y = np.random.randint(self.width-1) + 1
            if self.room[x][y] == 0 and self.goal != [x, y]:
                break

        if leftMost:
            x, y = 1, 1

        self.start = [x, y]

    def actionSpace(self):
        self.NOOP = 0
        self.LEFT = 1
        self.RIGHT = 2
        self.UP = 3
        self.DOWN = 4

    def step(self, action):
        x = self.pos[0]
        y = self.pos[1]

        if action == self.LEFT:
            if self.room[x - 1][y] == 0:
                self.pos = [x - 1, y]
        elif action == self.RIGHT:
            if self.room[x + 1][y] == 0:
                self.pos = [x + 1, y]
        elif action == self.UP:
            if self.room[x][y - 1] == 0:
                self.pos = [x, y - 1]
        elif action == self.DOWN:
            if self.room[x][y + 1] == 0:
                self.pos = [x, y + 1]

        rew = 0
        if tuple(self.pos) == tuple(self.goal):
            rew = 10
            self.done = True

        if self.incremental:
            self.steps += 1
            if self.steps >= 100:
                self.done = True
        return self.pos, rew

    def isDone(self):
        return self.done

    def plotState(self, path=None):
        self.count += 1
        tmp = np.asarray(self.room)
        tmp = tmp * 2
        tmp[self.goal[0], self.goal[1]] = 3  # Goal
        tmp[self.pos[0], self.pos[1]] = 4  # Current
        tmp[self.start[0], self.start[1]] = 5  # Current

        plt.imshow(tmp, interpolation=None, cmap=plt.get_cmap("viridis"), origin='lower')
        plt.axis('off')
        if path:
            pth = path + "img" + str(self.count) + ".png"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(pth)
            plt.close()
        else:
            plt.show()
