"""
Defines class for Successor Options
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

from support.medoids import medoidCluster
from support.medoids import clusteringHeuristic
from support.medoids import kmeansCluster
from support.plot3d import plot3d
from bokeh import palettes


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

            self.successor[rowNum1] = (1 - alpha) * self.successor[rowNum1] +  \
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
            validSuccessor = (validSuccessor - np.mean(validSuccessor, axis=0)) / t

        #  Cluster the representations of only filtered valid states
        distances = np.zeros((validState, validState))
        for i in range(validState):
            for j in range(validState):
                distances[i, j] = self.dist(validSuccessor[i], validSuccessor[j])

        self.clusterLabel = None
        if clusterType == 0:
            points, self.medoids = medoidCluster(distances, k=numClusters)
        elif clusterType == 1:
            self.medoids = clusteringHeuristic(distances, k=numClusters)
        elif clusterType == 2:
            self.medoids, self.clusterLabel = kmeansCluster(validSuccessor, k=numClusters)

        self.plotAllSubGoals(self.medoids, path)

        return self.medoids
    
    def plotAllSubGoals(self, points, path):
        newMap = np.array(self.env.room)
        height, width = len(newMap), len(newMap[0])
        room2 = 255 * (1 - np.array(self.env.room))
        room2 = np.transpose(np.array([room2, room2, room2]), [1, 2, 0])
        newMap = np.transpose(np.array([10*newMap, 10*newMap, 20*newMap]), [1, 2, 0])
        newMap = newMap + room2

        coords = []
        for pt in points:
            coords.append(self.validMap[pt])

        gap = int(float(250 - 110)/(len(points) + 1)) 
        pal = [palettes.Viridis256[x] for x in range(250, 110, -gap)]
        for j, pt in enumerate(coords):
            h = pal[j].lstrip('#')
            rgb = tuple(int(h[k:k+2], 16) for k in (0, 2 ,4))
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

