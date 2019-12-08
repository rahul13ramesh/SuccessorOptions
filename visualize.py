#!/usr/bin/env python3
"""
Plot Eigen-policies and Successor policies
"""
import numpy as np
import matplotlib.pyplot as plt

from laplacian import Laplacian
from successor import Successor
from env.env import gridWorld1
from support.qlearner import SimpleQLearner


def main():
    iters = [int(50e4), int(50e4), int(20e7), int(10e7)]

    for i in range(4):
        num = i + 1
        env = gridWorld1(size=num)

        laplaceModel = Laplacian(env)
        successorModel = Successor(env)

        print("Laplacian framework")
        laplaceModel.loadEigenVecs(
            "data/dat" + str(num) + "/eigenval" + str(num) + ".csv",
            "data/dat" + str(num) + "/eigenvec" + str(num) + ".csv",
            "data/dat" + str(num) + "/eigenMap" + str(num) + ".pkl")

        for i in range(20):
            laplaceModel.plotSingleEigen(
                i, "images/eigenvec" + str(num) + "/eigen" + str(i) + ".png")

        print("Successor framework")
        successorModel.loadSuccessor(
            "data/dat" + str(num) + "/successor" + str(num) + ".csv")

        ln = len(successorModel.keys)
        for i in range(20):
            obj = 1
            while obj == 1:
                ind = np.random.randint(ln)
                pos = successorModel.revMap[ind]
                obj = env.room[pos[0]][pos[1]]

            successorModel.plotSuccessorState(
                ind,
                "images/successor" + str(num) + "/sr2d" + str(i) + ".png",
                "images/successor" + str(num) + "/sr3d" + str(i) + ".png")


if __name__ == '__main__':
    main()
