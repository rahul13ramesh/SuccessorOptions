"""
Defines class for Successor Options and Incremental Successor options
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from bokeh import palettes

from support.medoids import medoidCluster
from support.medoids import clusteringHeuristic
from support.medoids import kmeansCluster
from support.medoids import kmeansClusterInc
from support.plot3d import plot3d


class Successor(object):
    def __init__(self, env):
        """
        Wrapper for obtaining Successor Options
        """

        self.env = env
        self.height = len(env.room)
        self.width = len(env.room[0])

        #  Create adjacency matrix
        self.keyInd = {}
        self.revMap = {}
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                self.revMap[count] = (i, j)
                self.keyInd[(i, j)] = count
                count += 1

        self.keys = self.keyInd.keys()
        self.successor = np.zeros((len(self.keys), len(self.keys)))

    def getSuccessor(self, alpha=0.1, gamma=1, iters=int(10e6), skewed=False):
        """
        Returns the sucessor representations after obtaining samples from
        uniformly random policy
        """
        self.env.reset()
        state = self.env.pos
        for i in tqdm(range(iters)):

            if skewed:
                if i % 1000 == 999:
                    self.env.resetStart()
                    self.env.reset()
                    state = self.env.pos

                act = np.random.choice(
                    [0, 1, 2, 3, 4],
                    p=[0.0, 0.3, 0.1, 0.3, 0.3])

            else:
                act = np.random.randint(5)

            prevState = list(state)
            state = self.env.step(act)[0]

            rowNum1 = self.keyInd[tuple(prevState)]
            rowNum2 = self.keyInd[tuple(state)]

            oneHot = np.zeros((len(self.keys)))
            oneHot[rowNum1] = 1

            self.successor[rowNum1] = (1 - alpha) * self.successor[rowNum1] + \
                alpha * (oneHot + gamma * self.successor[rowNum2])

        #  self.successor = self.successor/np.max(self.successor)

        return self.successor

    def saveSuccessor(self, path):
        np.savetxt(path, self.successor, delimiter=",")

    def saveLabels(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.clusterLabel, f)

    def loadLabels(self, path):
        with open(path, "rb") as f:
            self.clusterLabel = pickle.load(f)

    def loadSuccessor(self, path):
        self.successor = np.loadtxt(path, delimiter=",")

    def clusterRepresentation(
            self, numClusters, path,
            clusterType=0, normMethod=0):
        """
        Cluster the successor representations into clusters using k-medoids

        normMethod:  0-No normalization is done
                     1-Each column of SR matrix is normalized

        clusterType: 0-K-medoids
                     1-Greedy heursitic clustering
                     2-K-means                      (best model)
        """
        #  Get set of states which are not boundaries
        validState = 0
        for i in range(self.height):
            for j in range(self.width):
                validState += 1 - self.env.room[i][j]

        validSuccessor = np.zeros((validState, len(self.keys)))
        self.validMap = {}
        ct = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.env.room[i][j] == 0:
                    validSuccessor[ct] = self.successor[self.keyInd[(i, j)]]
                    self.validMap[ct] = (i, j)
                    ct += 1

        if normMethod == 0:
            pass

        if normMethod == 1:
            #  In order to avoid divide by 0 error(for constant columns)
            t = np.var(validSuccessor, axis=0)
            for i in range(t.size):
                if t[i] == 0:
                    t[i] = 1
            #  Normalize the columns for effective clustering
            validSuccessor = (validSuccessor - np.mean(validSuccessor,
                                                       axis=0)) / t

        #  Cluster the representations of only filtered valid states
        distances = np.zeros((validState, validState))
        for i in range(validState):
            for j in range(validState):
                distances[i, j] = self.dist(validSuccessor[i],
                                            validSuccessor[j])

        self.clusterLabel = None
        if clusterType == 0:
            points, self.medoids = medoidCluster(distances, k=numClusters)
        elif clusterType == 1:
            self.medoids = clusteringHeuristic(distances, k=numClusters)
        elif clusterType == 2:
            self.medoids, self.clusterLabel = kmeansCluster(validSuccessor,
                                                            k=numClusters)

        self.plotAllSubGoals(self.medoids, path)

        return self.medoids

    def plotAllSubGoals(self, points, path):
        newMap = np.array(self.env.room)
        room2 = 255 * (1 - np.array(self.env.room))
        room2 = np.transpose(np.array([room2, room2, room2]), [1, 2, 0])
        newMap = np.transpose(
            np.array([10*newMap, 10*newMap, 20*newMap]), [1, 2, 0])
        newMap = newMap + room2

        coords = []
        for pt in points:
            coords.append(self.validMap[pt])

        gap = int(float(250 - 110)/(len(points) + 1))
        pal = [palettes.Viridis256[x] for x in range(250, 110, -gap)]
        for j, pt in enumerate(coords):
            h = pal[j].lstrip('#')
            rgb = tuple(int(h[k:k+2], 16) for k in (0, 2, 4))
            newMap[pt[0], pt[1]] = np.array(rgb)

        def rect(pos):
            r = plt.Rectangle(pos-0.5, 1, 1, facecolor="none",
                              edgecolor="k", linewidth=1.3)
            plt.gca().add_patch(r)

        # Plot the medoid centers
        plt.imshow(newMap, origin='lower')
        plt.axis('off')
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        #  for i in range(height):
        #  for j in range(width):
        #  rect(np.array((j, i)))

        plt.savefig(path)
        plt.close()

    def plotSuccessor(self, numPlots):
        """
        Plot successor representation of random states
        """
        for j in range(numPlots):
            k = np.random.randint(len(self.keys))
            mat = np.reshape(self.successor[k], (self.height, self.width))
            plot3d(mat)
            plt.imshow(mat, cmap=plt.get_cmap("viridis"), origin='lower')
            plt.axis('off')
            plt.show()

    def plotSuccessorState(self, state, path1, path2):
        """
        Plot successor representation of specified state
        """
        k = state
        mat = np.reshape(self.successor[k], (self.height, self.width))
        plot3d(mat, path2)
        plt.imshow(mat, cmap=plt.get_cmap("viridis"), origin='lower')
        plt.axis('off')
        plt.savefig(path1)
        plt.close()

    @staticmethod
    def dist(ar1, ar2):
        return np.sum(np.square(ar1 - ar2))


class IncSuccessor(object):
    def __init__(self, env):
        """
        Wrapper or obtaining options using Inc. SR
        """

        self.env = env
        #  self.env.setRender()
        self.height = len(env.room)
        self.width = len(env.room[0])
        self.actionSize = 5

        #  Create adjacency matrix
        self.keyInd = {}
        self.revMap = {}
        count = 0
        validCount = 0
        for i in range(self.height):
            for j in range(self.width):
                self.revMap[count] = (i, j)
                self.keyInd[(i, j)] = count
                count += 1
                if env.room[i][j] == 0:
                    validCount += 1

        print("valid states ", validCount)
        self.keys = self.keyInd.keys()
        self.successor = np.zeros((len(self.keys), len(self.keys)))
        self.reachedStates = []

    def getSuccessor(self, alpha=0.1, gamma=1.0, iters=int(10e6),
                     optionsAvailable=False, render=False, itersteps=30000):
        """
        Returns the sucessor representations after obtaining samples from
        uniformly random policy
        """
        #  Reinitializing reachedStates and SR for each iteration
        #  self.reachedStates = []
        #  self.keys = self.keyInd.keys()
        #  self.successor = np.zeros((len(self.keys), len(self.keys)))
        self.loadOptions("data/dat4/policies/", optionsAvailable)
        self.optSize = len(self.Qopt)

        for i in tqdm(range(itersteps)):
            steps = 0
            #  rat = 50.0
            self.env.reset()
            state = self.env.pos
            if self.keyInd[tuple(state)] not in self.reachedStates:
                self.reachedStates.append(self.keyInd[tuple(state)])

            while True:
                if self.env.isDone():
                    break
                act = np.random.randint(self.actionSize)
                if optionsAvailable and steps == 0:
                    act = np.random.randint(self.optSize) + self.actionSize
                #  if np.random.rand() >= (1.0/rat) or self.optSize == 0:
                    #  act = np.random.randint(self.actionSize)
                #  else:
                    #  act = np.random.randint(self.optSize) + self.actionSize

                option_sampled = act
                # sampled an option, run it till it terminates
                if option_sampled >= self.actionSize:
                    stateOption = self.stateMap[tuple(state)]
                    while True:
                        prevState = list(state)
                        prevStateOption = int(stateOption)
                        act = self.getGreedyQOption(
                            prevStateOption, option_sampled - self.actionSize)

                        # option needs to be terminated
                        if act == 0 or (self.Qopt[option_sampled - self.actionSize][prevStateOption, act] <= 0):
                            break
                        else:
                            state = self.env.step(act)[0]
                            steps += 1
                            if render:
                                self.env.render()
                            stateOption = self.stateMap[tuple(state)]

                            if self.keyInd[tuple(state)] not in self.reachedStates:
                                self.reachedStates.append(
                                    self.keyInd[tuple(state)])

                            """rowNum1 = self.keyInd[tuple(prevState)]
                            rowNum2 = self.keyInd[tuple(state)]

                            oneHot = np.zeros((len(self.keys)))
                            oneHot[rowNum1] = 1

                            self.successor[rowNum1] = (1 - alpha) * self.successor[rowNum1] +  \
                                    alpha * (oneHot + gamma * self.successor[rowNum2])"""

                            if self.env.isDone():
                                break

                else:
                    act = option_sampled
                    prevState = list(state)
                    state = self.env.step(act)[0]
                    steps += 1
                    if render:
                        self.env.render()

                    if self.keyInd[tuple(state)] not in self.reachedStates:
                        self.reachedStates.append(self.keyInd[tuple(state)])

                    rowNum1 = self.keyInd[tuple(prevState)]
                    rowNum2 = self.keyInd[tuple(state)]

                    oneHot = np.zeros((len(self.keys)))
                    oneHot[rowNum1] = 1

                    self.successor[rowNum1] = (1 - alpha) * self.successor[rowNum1] +  \
                        alpha * (oneHot + gamma * self.successor[rowNum2])

        return self.successor

    def computeEigenvectors(self, ):
        self.eigenvals, self.eigenvecs = np.linalg.eigh(self.validSuccessor)

    def getValidSuccessor(self, ):

        self.validStates = 0
        for i in range(self.height):
            for j in range(self.width):
                self.validStates += 1 - self.env.room[i][j]

        validState = 0
        # validMap = []
        self.validKeyInd = {}
        self.validRevMap = {}
        self.validSuccessor = np.zeros((self.validStates, len(self.keys)))
        for i in range(self.height):
            for j in range(self.width):
                # validState += 1 - self.env.room[i][j]
                if self.env.room[i][j] == 0:
                    # validMap.append(self.keyInd[(i, j)])
                    self.validSuccessor[validState] = self.successor[self.keyInd[(
                        i, j)]]
                    self.validKeyInd[(i, j)] = validState
                    self.validRevMap[validState] = (i, j)
                    validState += 1

        # validSuccessor = self.successor[validMap][:, validMap]
        # self.validSuccessor = validSuccessor

    def getReachedSuccessor(self, ):

        reachedState = 0
        self.reachedKeyInd = {}
        self.reachedRevMap = {}
        self.reachedSuccessor = np.zeros(
            (len(self.reachedStates), len(self.keys)))
        for i in range(self.height):
            for j in range(self.width):
                if self.keyInd[(i, j)] in self.reachedStates:
                    self.reachedSuccessor[reachedState] = self.successor[self.keyInd[(
                        i, j)]]
                    self.reachedKeyInd[(i, j)] = reachedState
                    self.reachedRevMap[reachedState] = (i, j)
                    reachedState += 1

        print("number of reached states", len(self.reachedStates))
        # reachedSuccessor = self.successor[self.reachedStates]#[:, self.reachedStates]

        #self.reachedSuccessor = reachedSuccessor

    def saveSuccessor(self, path):
        np.savetxt(path, self.successor, delimiter=",")

    def loadSuccessor(self, path):
        self.successor = np.loadtxt(path, delimiter=",")

    def getGreedyQOption(self, state, id):
        qvals = self.Qopt[id][state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def loadOptions(self, path, optionsAvailable):
        """
        Load the Q-values(option policy) of options into self.Qopt
        """
        self.Qopt = []
        optSize = 5

        if optionsAvailable:
            for i in range(optSize):

                filePath = path + "explorepolicy" + str(i) + ".csv"
                fp = open(filePath, "rb")
                curOptQvals, stateMap = pickle.load(fp)
                fp.close()

                self.Qopt.append(deepcopy(curOptQvals))

            if optSize != 0:
                self.stateMap = deepcopy(stateMap)

    def getRareStateSuccessor(self, final=False):

        numLow = 5
        numHigh = 30
        if final:
            numLow = 0
            numHigh = 100
        srSum = np.sum(self.reachedSuccessor, axis=1)
        srLowerQuartile = np.percentile(srSum, numLow)
        srUpperQuartile = np.percentile(srSum, numHigh)
        self.rareStates = np.asarray(
            np.where((srSum < srUpperQuartile) & (srSum > srLowerQuartile))).squeeze()

        # self.rareStates = np.argsort(np.sum(self.reachedSuccessor, axis=1))#[:20]

        self.rareStatesSuccessor = []
        for j in range(len(self.rareStates)):
            self.rareStatesSuccessor.append(
                self.validSuccessor[self.validKeyInd[self.reachedRevMap[self.rareStates[j]]]])

        self.rareStatesSuccessor = np.array(self.rareStatesSuccessor)

    def clusterRepresentation(
            self, numClusters, path,
            clusterType=2, normMethod=0):
        """
        Cluster the successor representations into clusters using k-medoids
        """
        #  Get set of states which are not boundaries
        # validState = 0
        # for i in range(self.height):
        #     for j in range(self.width):
        #         validState += 1 - self.env.room[i][j]

        validSuccessor = self.rareStatesSuccessor
        validState = validSuccessor.shape[0]
        # validSuccessor = np.zeros((validState, len(self.keys)))
        # self.validMap = {}
        # ct = 0
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if self.env.room[i][j] == 0:
        #             self.validMap[ct] = (i, j)
        #             ct += 1

        if normMethod == 0:
            pass

        if normMethod == 1:
            #  In order to avoid divide by 0 error(for constant columns)
            t = np.var(validSuccessor, axis=0)
            for i in range(t.size):
                if t[i] == 0:
                    t[i] = 1
            #  Normalize the columns for effective clustering
            validSuccessor = (
                validSuccessor - np.mean(validSuccessor, axis=0)) / t

        #  Cluster the representations of only filtered valid states
        distances = np.zeros((validState, validState))
        for i in range(validState):
            for j in range(validState):
                distances[i, j] = self.dist(
                    validSuccessor[i], validSuccessor[j])

        if clusterType == 0:
            points, self.medoids = medoidCluster(distances, k=numClusters)
        elif clusterType == 1:
            self.medoids = clusteringHeuristic(distances, k=numClusters)
            print(self.medoids)
        elif clusterType == 2:
            self.medoids = kmeansClusterInc(
                validSuccessor, self.rareStates, k=numClusters)

        self.plotAllSubGoals(self.medoids, path)
        return self.medoids

    def plotAllSubGoals(self, points, path):
        newMap = np.array(self.env.room)
        room2 = 255 * (1 - np.array(self.env.room))
        room2 = np.transpose(np.array([room2, room2, room2]), [1, 2, 0])
        newMap = np.transpose(
            np.array([10*newMap, 10*newMap, 20*newMap]), [1, 2, 0])
        newMap = newMap + room2

        coords = []
        for pt in points:
            coords.append(self.reachedRevMap[pt])

        for j, pt in enumerate(coords):
            rgb = (33, 113, 181)
            newMap[pt[0], pt[1]] = np.array(rgb)

        def rect(pos):
            r = plt.Rectangle(pos-0.5, 1, 1, facecolor="none",
                              edgecolor="k", linewidth=1.3)
            plt.gca().add_patch(r)

        # Plot the medoid centers
        plt.imshow(newMap, origin='lower')
        plt.axis('off')
        plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
        #  for i in range(height):
        #  for j in range(width):
        #  rect(np.array((j, i)))

        plt.savefig(path)
        plt.close()

    def plotSuccessor(self, numPlots):
        """
        Plot successor representation of random states
        """
        for j in range(numPlots):
            k = np.random.randint(len(self.keys))
            mat = np.reshape(self.successor[k], (self.height, self.width))
            plot3d(mat)
            plt.imshow(mat, cmap=plt.get_cmap("viridis"), origin='lower')
            plt.axis('off')
            plt.show()

    def plotSuccessorState(self, state, path1, path2):
        """
        Plot successor representation of random states
        """
        k = state
        mat = np.reshape(self.successor[k], (self.height, self.width))
        plot3d(mat, path2)
        plt.imshow(mat, cmap=plt.get_cmap("viridis"), origin='lower')
        plt.axis('off')
        plt.savefig(path1)
        plt.close()

    def plotSingleEigen(self, eigenNum, path):
        sortedind = np.argsort(self.eigenvals)[::-1]
        height = len(self.env.room)
        width = len(self.env.room[0])

        i = eigenNum
        row = self.eigenvecs[:, sortedind[i]]
        eigenMat = np.zeros((height - 2, width - 2))
        eigenMat2 = eigenMat * np.nan
        for i, val in enumerate(row):
            node = self.validRevMap[i]
            eigenMat[node[0] - 1, node[1] - 1] = val
            eigenMat2[node[0] - 1, node[1] - 1] = val
        plot3d(eigenMat, path)
        # plt.imshow(eigenMat2, cmap=plt.get_cmap("viridis"), origin='lower')
        # plt.axis('off')
        # plt.savefig(path)
        # plt.close()

    def plotSuccessorMagnitudes(self, path=None):
        newMap = np.array(self.env.room)
        room2 = 255 * (1 - np.array(self.env.room))
        room2 = np.transpose(np.array([room2, room2, room2]), [1, 2, 0])
        newMap = np.transpose(np.array([10*newMap, 10*newMap, 20*newMap]),
                              [1, 2, 0])
        newMap = newMap + room2

        pal = palettes.Viridis256

        mat = np.zeros((self.height, self.width))
        sums = np.sum(self.reachedSuccessor, axis=1)
        print("magnitude max")
        print(np.max(sums))
        for k in self.reachedRevMap:
            i, j = self.reachedRevMap[k]
            sums = np.sum(self.reachedSuccessor, axis=1)
            mat[i, j] = sums[k]

            val = int(255 * (min(sums[k], 2000)/float(2000.0)))
            h = pal[val].lstrip('#')
            rgb = tuple(int(h[k:k+2], 16) for k in (0, 2, 4))

            newMap[i, j] = np.array(rgb)

        plt.imshow(newMap, origin='lower')
        plt.axis('off')
        plt.savefig(path)
        plt.close()

    @staticmethod
    def dist(ar1, ar2):
        return np.sum(np.square(ar1 - ar2))
