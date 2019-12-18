"""
Class that defines Eigen-options
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from support.plot3d import plot3d


class Laplacian(object):
    def __init__(self, env):
        """
        Class for defining object for Eigen-options
        """
        self.env = env
        self.height = len(env.room)
        self.width = len(env.room[0])

    def getLaplacian(self, iters=int(10e6)):
        """
        Follow uniformly random policy and obtain the graph laplacian
        Returns the the eigenvalues, eigenvectors
        """

        mat = np.zeros((self.height, self.width), dtype=np.int32)
        state = self.env.pos

        graph = {}
        for i in tqdm(range(iters)):

            #  Uniformly random policy
            act = np.random.randint(5)
            prevState = list(state)

            state = self.env.step(act)[0]
            mat[state[0], state[1]] += 1

            prevState = tuple(prevState)
            if prevState not in graph:
                graph[prevState] = set()

            graph[prevState].add(tuple(state))

        #  Ensure every state has out-degree
        keys = graph.keys()
        for node in graph:
            for x in graph[node]:
                assert(x in keys)

        #  Create map of from matrix row to env state
        self.keyInd = {}
        self.revMap = {}
        for i, k in enumerate(keys):
            self.revMap[i] = k
            self.keyInd[k] = i

        #  Create adjacency matrix
        adjMat = np.zeros((len(keys), len(keys)))
        for node in graph:
            for x in graph[node]:
                adjMat[self.keyInd[node], self.keyInd[x]] += 1

        rSum = adjMat.sum(axis=1)
        D = np.diag(rSum)
        Dinv = np.diag(1 / rSum)
        lap = D - adjMat

        #  EVD
        self.eigenvals, self.eigenvecs = np.linalg.eig(np.matmul(Dinv, lap))

        return self.eigenvals, self.eigenvecs

    def plotLaplacianEigen(self, plotNum=50):
        """
        plot eigenvectors in increasing order of eigen-values
        """
        #  Sort smallest eigen-values
        sortedind = np.argsort(self.eigenvals)
        height = len(self.env.room)
        width = len(self.env.room[0])

        for i in range(plotNum):
            row = self.eigenvecs[:, sortedind[i]]
            eigenMat = np.zeros((height - 2, width - 2))
            eigenMat2 = eigenMat * np.nan
            for i, val in enumerate(row):
                node = self.revMap[i]
                eigenMat[node[0] - 1, node[1] - 1] = val
                eigenMat2[node[0] - 1, node[1] - 1] = val
            plot3d(eigenMat)
            plt.imshow(eigenMat2, cmap=plt.get_cmap("viridis"), origin='lower')
            plt.axis('off')

    def plotSingleEigen(self, eigenNum, path):
        """
        Plot the eigenvector corresponding to the eigenNum smallest eigenvalue
        """
        sortedind = np.argsort(self.eigenvals)
        height = len(self.env.room)
        width = len(self.env.room[0])

        i = eigenNum
        row = self.eigenvecs[:, sortedind[i]]
        eigenMat = np.zeros((height - 2, width - 2))
        eigenMat2 = eigenMat * np.nan
        for i, val in enumerate(row):
            node = self.revMap[i]
            eigenMat[node[0] - 1, node[1] - 1] = val
            eigenMat2[node[0] - 1, node[1] - 1] = val
        plt.imshow(eigenMat2, cmap=plt.get_cmap("viridis"), origin='lower')
        plt.axis('off')
        plt.savefig(path)
        plt.close()

    def saveEigenVec(self, path1, path2, path3):
        """
        Save the eigenvector, eigenvalues and reward structure
        """
        np.savetxt(path1, self.eigenvals, delimiter=",")
        np.savetxt(path2, self.eigenvecs, delimiter=",")
        f3 = open(path3, "wb")
        pickle.dump((self.keyInd, self.revMap), f3)
        f3.close()

    def loadEigenVecs(self, path1, path2, path3):
        """
        Load the eigenvector, eigenvalues and reward structure
        """
        self.eigenvals = np.loadtxt(path1, delimiter=",")
        self.eigenvecs = np.loadtxt(path2, delimiter=",")
        f3 = open(path3, "rb")
        self.keyInd, self.revMap = pickle.load(f3)
