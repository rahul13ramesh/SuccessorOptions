import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pickle

plt.switch_backend('agg')

class SimpleQLearner(object):
    def __init__(self, env, rewardVec, indMap, actSize=5, gamma=0.9999):
        """
        Q-learner that trains on primitive actions
        Use to train the option policies
        """
        self.env = env
        self.rewVec = rewardVec
        self.stateMap = indMap
        self.gamma = gamma
        self.actionSize = actSize
        self.stateSize = self.rewVec.size
        self.Q = np.zeros((self.stateSize, self.actionSize))

    def train(self, iters=int(10e6)):
        """
        Run iterations to obtain optimal policy using Q-learning
        """

        EPS_START = 1
        EPS_END = 0
        alpha = 0.9
        state = self.stateMap[tuple(self.env.reset())]
        for i in tqdm(range(iters)):

            eps = max(0, EPS_START - (EPS_START - EPS_END) * float((4/5) * i/iters))

            prevState = int(state)
            if np.random.rand() < eps:
                #  unformly random
                act = np.random.randint(5)
                state, _ = self.env.step(act)
            else:
                act = self.getGreedyQ(prevState)
                state, _ = self.env.step(act)
                #  Greedy
            state = self.stateMap[tuple(state)]
            rew = self.reward(prevState, state)
            actNext = self.getGreedyQ(state)

            self.Q[prevState, act] = alpha * self.Q[prevState, act] + \
                (1 - alpha) * (rew + self.gamma * self.Q[state, actNext])

        return self.Q

    def reward(self, s1, s2):
        """
        The reward for the given eigenvector/SR(self.rewVec)
        """
        r1 = self.rewVec[s1]
        r2 = self.rewVec[s2]
        return (r2 - r1)

    def getGreedyQ(self, state):
        """
        Get action that maximizes Q-value
        """
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def plotPolicy(self, path):
        """
        Plot policy with arrows
        """
        height = len(self.env.room)
        width = len(self.env.room[0])

        X = np.zeros((height, width))
        Y = np.zeros((height, width))

        U = np.zeros((height, width))
        V = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                X[i, j] = i + 0.5
                Y[i, j] = j + 0.5

        for i in range(height):
            for j in range(width):
                if (i, j) not in self.stateMap:
                    a = 0
                else:
                    a = self.getGreedyQ(self.stateMap[(i, j)])
                if a == 0 or (self.Q[self.stateMap[(i, j)], a] <= 0):
                    U[i, j] = 0.0
                    V[i, j] = 0.0
                elif a == 1:
                    U[i, j] = -1
                    V[i, j] = 0
                elif a == 2:
                    U[i, j] = 1
                    V[i, j] = 0
                elif a == 3:
                    U[i, j] = 0
                    V[i, j] = -1
                elif a == 4:
                    U[i, j] = 0
                    V[i, j] = 1

        major_ticksx = list(range(height))
        major_ticksy = list(range(width))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(color='gray', linestyle='-', linewidth=2)
        ax.grid(True)
        ax.set_xticks(major_ticksx)
        ax.set_yticks(major_ticksy)
        plt.xlim(xmin=0, xmax=height)
        plt.ylim(ymin=0, ymax=width)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.title('Policy Visualization')
        plt.quiver(X, Y, U, V, units='width', headwidth=5,
                   minlength=0.2, minshaft=0.2, linewidths=0.1, scale=100.3)
        cmap = plt.get_cmap('viridis')
        xval = 0

        for x in self.env.room:
            yval = 0
            for y in x:
                if y == 1:
                    plt.plot([xval + 0.5], [yval + 0.5],
                             marker='s',
                             markersize=14,
                             color=cmap(0))

                yval += 1
            xval += 1

        xval = 0
        for x in self.env.room:
            yval = 0
            for y in x:
                if y == 0:
                    if (xval, yval) in self.stateMap or True:
                        a = self.getGreedyQ(self.stateMap[(xval, yval)])
                        if a == 0 or (self.Q[self.stateMap[(xval, yval)], a] <= 0):
                            plt.plot([xval + 0.5], [yval + 0.5],
                                     marker='s',
                                     markersize=14,
                                     color=cmap(200))

                yval += 1
            xval += 1

        plt.savefig(path)
        plt.close()
        #  plt.show()

    def savePolicy(self, path):
        """
        Save the Q-values of the policy into pickle file
        """
        fp = open(path, "wb")
        pickle.dump((self.Q, self.stateMap), fp)
        fp.close()

    def loadPolicy(self, path):
        """
        Load the Q-values of the policy from pickle file
        """

        fp = open(path, "rb")
        self.Q, self.stateMap = pickle.load(fp)
        fp.close()

    def plotPolicy2(self):
        """
        Plot the subgoals/terminal points of the policy
        """
        room = deepcopy(self.env.room)

        xval = 0
        for x in self.env.room:
            yval = 0
            for y in x:
                if y == 0:
                    if (xval, yval) in self.stateMap:
                        s = self.stateMap[(xval, yval)]
                        if self.Q[s, self.getGreedyQ(s)] <= 0:
                                room[xval][yval] = 3
                yval += 1
            xval += 1

        plt.imshow(room, cmap=plt.get_cmap("viridis"), origin='lower')
        plt.axis('off')
        plt.show()


