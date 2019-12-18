#!/usr/bin/env python3
"""
Generate the policies using the Laplacian Eigenoptions
"""
import numpy as np
from multiprocessing import Pool

from laplacian import Laplacian
from env.env import gridWorld1
from support.qlearner import SimpleQLearner


laplaceModel = None


def trainLaplace(i):
    global laplaceModel
    global globalNum, env
    print(globalNum, i)
    # Q-learning iterations
    titers = [int(30e5), int(30e5), int(50e6), int(50e6)]
    #  Use eigenvalues in descending order
    sortedind = np.argsort(laplaceModel.eigenvals)
    if i % 2 == 0:
        rew = laplaceModel.eigenvecs[:, sortedind[int(i/2)]]
    else:
        rew = -laplaceModel.eigenvecs[:, sortedind[int(i/2)]]
    #  ind = laplaceModel.revMap[np.argmax(rew)]
    laplaceModel.plotSingleEigen(
        int(i/2), "images/policies" + str(globalNum) + "/eigen"
        + str(i) + ".png")

    Qlearner = SimpleQLearner(env, rew, laplaceModel.keyInd)
    Qlearner.train(titers[globalNum-1])
    Qlearner.plotPolicy("images/policies" + str(globalNum)
                        + "/eigenpolicy" + str(i) + ".png")
    Qlearner.savePolicy("data/dat" + str(globalNum) + "/policies/eigenpolicy"
                        + str(i) + ".csv")


def main():
    iters = [int(50e4), int(50e4), int(10e6), int(10e6)]
    numMod = [5, 5, 10, 10]
    global globalNum, env
    NUM_CORES = 8

    # iterate over 4 envs
    for num in range(1, 5):
        print(num)
        globalNum = num
        env = gridWorld1(size=num)

        global laplaceModel
        laplaceModel = Laplacian(env)

        laplaceModel.getLaplacian(iters=iters[num-1])
        laplaceModel.saveEigenVec(
            "data/dat" + str(num) + "/eigenval" + str(num) + ".csv",
            "data/dat" + str(num) + "/eigenvec" + str(num) + ".csv",
            "data/dat" + str(num) + "/eigenMap" + str(num) + ".pkl")

        # Load laplacian model
        laplaceModel.loadEigenVecs(
            "data/dat" + str(num) + "/eigenval" + str(num) + ".csv",
            "data/dat" + str(num) + "/eigenvec" + str(num) + ".csv",
            "data/dat" + str(num) + "/eigenMap" + str(num) + ".pkl")

        #  Train policies for eigen-vectors (train in parallel)
        procPool = Pool(NUM_CORES)
        procPool.map(trainLaplace, range(numMod[num-1]))


if __name__ == '__main__':
    main()
