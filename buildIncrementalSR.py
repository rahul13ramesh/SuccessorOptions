import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


from laplacian import Laplacian
from successor import IncSuccessor
from env.env import gridWorld1
from support.qlearner import SimpleQLearner
from support.smdpqlearner import SimpleQLearnerOverOptions


def main():
    titers = [int(10e5), int(5e3), int(5e3), int(5e3)]

    numMod = [10, 4, 7, 5]

    env_num = 4
    assert(env_num >= 1 and env_num <= 4)

    print("Env : " + str(num))
    env = gridWorld1(size=num)
    print("goal ", env.goal)
    srModel = IncSuccessor(env)
    render = False

    #  options are unavailable in first round 
    optionsAvailable = False

    #  Each incremental iteration
    for itera in range(9):

        #  Build successor representation with current set of options
        srModel.getSuccessor(optionsAvailable=optionsAvailable, render=render)
        srModel.getValidSuccessor()
        srModel.getReachedSuccessor()
        srModel.plotSuccessorMagnitudes("images/magnitudes/" + str(itera) + ".png")

        room1 = deepcopy(env.room)

        randomState = []
        for i in range(20):
            while True:
                x = np.random.randint(env.height)
                y = np.random.randint(env.width)
                if env.room[x][y] == 0:
                    break

            randomState.append(srModel.validKeyInd[tuple([x, y])])

        # If final iteration, run full successor representatio
        if itera >= 8:
            srModel.getRareStateSuccessor(final=True)
        # Otherwise, cluster SR of states that are not used frequently
        else:
            srModel.getRareStateSuccessor()
        srModel.clusterRepresentation(numMod[num-1], "images/inc-goals/sr" + str(itera) + ".png", 2, 0)

        print("size", srModel.successor.shape)
        curMedoids = []

        #  Iterate over every cluster center and learn policy
        for j, md in enumerate(srModel.medoids):
            print(j)

            rew = srModel.validSuccessor[srModel.validKeyInd[srModel.reachedRevMap[md]]]

            Qlearner = SimpleQLearner(env, rew, srModel.keyInd)
            if itera >= 8:
                Qlearner.train(int(5e4))
            else:
                Qlearner.train(titers[num-1])
            Qlearner.plotPolicy("images/policies" + str(num) + "/explorepolicy" + str(j) + ".png")
            Qlearner.savePolicy("data/dat" + str(num) + "/policies/explorepolicy" + str(j) + ".csv")

            ind = srModel.reachedRevMap[md]

            plot_ind = srModel.keyInd[ind]
            srModel.plotSuccessorState(
                plot_ind, "images/sr" + str(num) + "/sr2d" + str(j) + ".png",
                "images/sr" + str(num) + "/sr3d" + str(j) + ".png")

            curMedoids.append(md)

        optionsAvailable = True


if __name__ == '__main__':
    main()
