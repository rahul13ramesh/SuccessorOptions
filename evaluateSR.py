"""
Script for evaluating the Successor options/Eigen-options
built from the buildPolicies*.py files
"""
import argparse
import random
import numpy as np

from support.smdpqlearner import SmdpQlearner
from env.env import gridWorld1


def main():

    parser = argparse.ArgumentParser(description='Get SR policies')
    parser.add_argument('--env', type=int, default=3)
    parser.add_argument('--ratio', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    assert(args.env >= 1 and args.env <= 4)

    ITERS = [int(5e4), int(5e4), int(5e5), int(5e5)][args.env-1]
    OPTSIZE = [4, 5, 10, 10][args.env-1]
    EVALNUM = 100

    for i in range(200):
        env = gridWorld1(size=args.env)
        env.resetGoal()
        env.resetStart()
        env.reset()

        #  env.plotState(path=None)
        pth = "data/dat" + str(args.env) + "/policies/"

        print("Option0")
        qlearner1 = SmdpQlearner(env, optionPath=pth, optSize=0, gamma=0.99, plot=False)
        qlearner1.train(iters=ITERS, evalNum=EVALNUM, policyPath="images/tmp1")

        print("Option5-Eigen")
        qlearner2 = SmdpQlearner(env, optionPath=pth, optSize=OPTSIZE, gamma=0.99, plot=False)
        qlearner2.train(iters=ITERS, evalNum=EVALNUM, policyPath="images/tmp2")

        print("Option5-SR")
        qlearner3 = SmdpQlearner(env, optionPath=pth, optSize=OPTSIZE, gamma=0.99, plot=False, eigen=False)
        qlearner3.train(iters=ITERS, evalNum=EVALNUM, policyPath="images/tmp3")

        print("Option5-Eigen-NU")
        qlearner4 = SmdpQlearner(env, optionPath=pth, optSize=OPTSIZE, gamma=0.99, plot=False)
        qlearner4.train(iters=ITERS, evalNum=EVALNUM, policyPath="images/tmp2", uniform=False, rat=args.ratio)

        print("Option5-SR-NU")
        qlearner5 = SmdpQlearner(env, optionPath=pth, optSize=OPTSIZE, gamma=0.99, plot=False, eigen=False)
        qlearner5.train(iters=ITERS, evalNum=EVALNUM, policyPath="images/tmp3", uniform=False, rat=args.ratio)

        print("Option5-SR-AE")
        pklPath = "data/dat" + str(args.env) +  "/successorLabels" + str(args.env) + ".pkl"
        qlearner5 = SmdpQlearner(env, optionPath=pth, optSize=OPTSIZE, gamma=0.99, plot=False, eigen=False)
        qlearner5.train(iters=ITERS, evalNum=EVALNUM, policyPath="images/tmp3", uniform=False,
                        rat=args.ratio, adaptive=True, adaPath = pklPath)


if __name__ == '__main__':
    main()
