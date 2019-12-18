#!/usr/bin/env python3
"""
Get the policies for the sr-options
"""

import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool
from successor import Successor
from env.env import gridWorld1
from support.qlearner import SimpleQLearner


plt.switch_backend('agg')
srModel = None
global srModel, env, globNum


def buildSRPolicy(j):
    global srModel, env, globNum
    print(j, globNum)
    titers = [int(10e5), int(20e5), int(50e6), int(50e6)]
    md = srModel.medoids[j]
    pt = srModel.keyInd[srModel.validMap[md]]
    rew = srModel.successor[pt]
    Qlearner = SimpleQLearner(env, rew, srModel.keyInd)
    Qlearner.train(titers[globNum-1])
    Qlearner.plotPolicy("images/policies" + str(globNum) + "/srpolicy"
                        + str(j) + ".png")
    Qlearner.savePolicy("data/dat" + str(globNum) + "/policies/srpolicy"
                        + str(j) + ".csv")


def main():
    parser = argparse.ArgumentParser(description='Get SR policies')
    parser.add_argument('--cluster', type=int, default=2)
    parser.add_argument('--norm', type=int, default=0)
    args = parser.parse_args()

    #  iters = [int(50e4), int(50e4), int(10e6), int(10e6)]
    #  The iterations for training Q-learning
    numMod = [4, 5, 10, 10]
    for num in range(1, 5):
        print("Env : " + str(num))
        global srModel, env, globNum
        env = gridWorld1(size=num)
        globNum = num
        srModel = Successor(env)
        srModel.loadSuccessor(
            "data/dat" + str(num) + "/successor" + str(num) + ".csv")

        srModel.clusterRepresentation(
            numMod[num-1], "images/tmp/sr.png",
            args.cluster, args.norm)
        srModel.saveLabels('data/dat' + str(num) + "/successorLabels"
                           + str(num) + ".pkl")

        procPool = Pool(12)
        procPool.map(buildSRPolicy, range(len(srModel.medoids)))


if __name__ == '__main__':
    main()
