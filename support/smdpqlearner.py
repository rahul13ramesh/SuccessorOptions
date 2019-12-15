import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pickle

plt.switch_backend('agg')


class SmdpQlearner(object):
    def __init__(self, env, actSize=5, optSize=10, plot=False, postfix="",
                 optionPath=None, eigen=True, gamma=0.9999):
        self.env = env
        self.gamma = gamma
        self.plot = plot
        self.actSize = actSize
        self.testNum = 0
        self.optSize = optSize
        self.postfix = postfix
        self.eigen = eigen
        self.stateSize = (self.env.height * self.env.width)
        self.Q = np.zeros((self.stateSize, self.actSize + self.optSize))

        self.env.reset()

        self.loadOptions(optionPath)

    def loadOptions(self, path):
        """
        Load the Q-values(option policy) of options into self.Qopt
        """
        self.Qopt = []
        if self.eigen:
            prefix = "eigen"
        else:
            prefix = "sr"

        for i in range(self.optSize):

            filePath = path + prefix + "policy" + str(i) + self.postfix + ".csv"
            fp = open(filePath, "rb")
            curOptQvals, stateMap = pickle.load(fp)
            fp.close()

            self.Qopt.append(deepcopy(curOptQvals))

        if self.optSize != 0:
            self.stateMap = deepcopy(stateMap)
        else:
            filePath = path + prefix + "policy0.csv"
            fp = open(filePath, "rb")
            curOptQvals, stateMap = pickle.load(fp)
            fp.close()
            self.stateMap = deepcopy(stateMap)

    def train(self, iters=int(10e6), evalNum=50, rat=20.0,
              policyPath="tmp", uniform=True, adaptive=False, adaPath=None):
        """
        Train using SMDP-Q-learning
        The following update are used
            * Execute action a
                - Update action a using normal Q-learning update
                - For all options, check if opt(o, s) = a
                - If true:
                    Q(s, o)  = a*Q(s,o) + (1-a)*(r + Q(s', o)) if s' non-terminal
                    Q(s, o)  = a*Q(s,o) + (1-a)*(r + max_a Q(s', a)) if s' terminal
            * Execute option o:
                * Q(s, o)  = a*Q(s,o) + (1-a)*(r + Q(s', o)) if s' non-terminal
                * Q(s, o)  = a*Q(s,o) + (1-a)*(r + max_a Q(s', a)) if s' terminal
                * Q(s, a) for all (s,a) pairs encountered while executing option
                * If opt(o', s) = a = opt(o, s)
                    Q(s, o')  = a*Q(s,o') + (1-a)*(r + Q(s', o') if s' non-terminal
                    Q(s, o')  = a*Q(s,o') + (1-a)*(r + max_a Q(s', a)) if s' terminal
        """

        clusterLabels = None
        if adaptive:
            with open(adaPath, "rb") as f:
                clusterLabels = pickle.load(f)
            clusterLabels = np.array(clusterLabels)
            clusterLabels = clusterLabels / np.mean(clusterLabels)

        EPS_START = 1
        EPS_END = 0
        alpha = 0.95
        state = self.stateMap[tuple(self.env.reset())]

        if evalNum != 0:
            loginterval = int(iters/evalNum)
        else:
            loginterval = iters + 1

        i = 1
        j = 1
        prevOption = 0
        while i < iters:
            eps = max(0, EPS_START - (EPS_START - EPS_END) * float((4/5) * i/iters))

            if j - loginterval >= 0:
                j -= loginterval
                tst, var = self.test(policyPath)
                print("Return at " + str(i) + " : " + str(tst) + " +- " + str(var))

            prevState = int(state)
            if np.random.rand() < eps:
                #  Uniformly random
                if self.optSize != 0:
                    if uniform:
                        act = np.random.randint(self.optSize + self.actSize)
                    else:
                        if adaptive:
                            newRat = float(rat) * clusterLabels[prevOption]
                            if np.random.rand() >= (1.0/newRat):
                                act = np.random.randint(self.actSize)
                            else:
                                act = np.random.randint(self.optSize) + self.actSize
                        else:
                            if np.random.rand() >= (1.0/rat):
                                act = np.random.randint(self.actSize)
                            else:
                                act = np.random.randint(self.optSize) + self.actSize
                else:
                    act = np.random.randint(self.actSize)
            else:
                #  Greedy
                act = self.getGreedyQ(prevState)

            if act < self.actSize:
                state, rew = self.env.step(act)
                i += 1
                j += 1
                done = self.env.isDone()
                state = self.stateMap[tuple(state)]

                actNext = self.getGreedyQ(state)
                self.Q[prevState, act] = alpha * self.Q[prevState, act] + \
                    (1 - alpha) * (rew + self.gamma * self.Q[state, actNext])

                for op in range(self.actSize, self.actSize + self.optSize):
                    if act in self.getGreedyQoptList(prevState, op):
                        if self.Qopt[op-self.actSize][prevState, act] > 0:
                            if self.Qopt[op-self.actSize][state, act] <= 0:
                                actNextOpt = self.getGreedyQ(state)
                                self.Q[prevState, op] = alpha * self.Q[prevState, op] + \
                                    (1 - alpha) * (rew + self.gamma * self.Q[state, actNextOpt])
                            else:
                                #  actNextOpt = self.getGreedyQopt(state, op)
                                self.Q[prevState, op] = alpha * self.Q[prevState, op] + \
                                    (1 - alpha) * (rew + self.gamma * self.Q[state, op])

                if done:
                    self.env.reset()

            else:
                #  Execute the option
                prevState = int(state)
                optAct = self.getGreedyQopt(prevState, act)
                prevOption = act - self.actSize

                opLen = 0
                while self.Qopt[act-self.actSize][prevState, optAct] > 0:
                    opLen += 1
                    state, rew = self.env.step(optAct)
                    i += 1
                    j += 1
                    done = self.env.isDone()
                    state = self.stateMap[tuple(state)]

                    if state == prevState:
                        break

                    for op in range(self.actSize, self.actSize + self.optSize):
                        if optAct in self.getGreedyQoptList(prevState, op):
                            if self.Qopt[op-self.actSize][prevState, optAct] > 0:
                                if self.Qopt[op-self.actSize][state, optAct] <= 0:
                                    actNextOpt = self.getGreedyQ(state)
                                    self.Q[prevState, op] = alpha * self.Q[prevState, op] + \
                                        (1 - alpha) * (rew + self.gamma * self.Q[state, actNextOpt])
                                else:
                                    #  actNextOpt = self.getGreedyQopt(state, op)
                                    self.Q[prevState, op] = alpha * self.Q[prevState, op] + \
                                        (1 - alpha) * (rew + self.gamma * self.Q[state, op])

                    actNextOpt = self.getGreedyQ(state)
                    self.Q[prevState, optAct] = alpha * self.Q[prevState, optAct] + \
                        (1 - alpha) * (rew + self.gamma * self.Q[state, actNextOpt])

                    if done:
                        self.env.reset()
                        break

                    prevState = int(state)
                    optAct = self.getGreedyQopt(prevState, act)
        m, v = self.test(policyPath)
        print("Final performance : " + str(m) + " +- " + str(v))
        return self.Q

    def randomDensity(self, iters=int(10e6), evalNum=50, rat=20.0,
                      policyPath="tmp", uniform=True):
        state = self.stateMap[tuple(self.env.reset())]

        self.randomDensity = np.zeros((self.stateSize))

        loginterval = 100
        i = 1
        j = 1
        while i < iters:

            if j - loginterval >= 0:
                j -= loginterval
                self.env.reset()
                #  tst, var = self.test(policyPath)
                #  print("Return at " + str(i) + " : " + str(tst) + " +- " + str(var))
            prevState = int(state)

            if self.optSize != 0:
                if uniform:
                    act = np.random.randint(self.optSize + self.actSize)
                else:
                    if np.random.rand() >= (1.0/rat):
                        act = np.random.randint(self.actSize)
                    else:
                        act = np.random.randint(self.optSize) + self.actSize
            else:
                act = np.random.randint(self.actSize)

            if act < self.actSize:
                state, rew = self.env.step(act)
                i += 1
                j += 1
                state = self.stateMap[tuple(state)]

                self.randomDensity[state] += 1

            else:
                #  Execute the option
                prevState = int(state)
                optAct = self.getGreedyQopt(prevState, act)

                opLen = 0
                while self.Qopt[act-self.actSize][prevState, optAct] > 0:
                    opLen += 1
                    state, rew = self.env.step(optAct)
                    i += 1
                    j += 1
                    state = self.stateMap[tuple(state)]

                    self.randomDensity[state] += 1

                    prevState = int(state)
                    optAct = self.getGreedyQopt(prevState, act)

        self.randomDensity /= evalNum
        return self.randomDensity

    def test(self, policyPath):
        self.testNum += 1
        envTest = deepcopy(self.env)
        allRet = []
        for i in range(100):
            state = self.stateMap[tuple(envTest.reset())]

            ret = 0.0
            steps = 0
            MAXSTEPS = 50
            while steps <= MAXSTEPS:
                act = self.getGreedyQ(state)
                if act < self.actSize:
                    state, rew = envTest.step(act)
                    if self.plot and i == 0:
                        envTest.plotState(policyPath + "/tmp" + str(self.testNum) + "/")
                    steps += 1
                    state = self.stateMap[tuple(state)]

                    ret = ret + self.gamma**(steps) * rew
                    done = envTest.isDone()
                    if done or steps > MAXSTEPS:
                        break

                else:
                    optAct = self.getGreedyQopt(state, act)
                    doneFlag = False

                    #  Execute option until you reach terminal state
                    while self.Qopt[act-self.actSize][state, optAct] > 0:

                        state, rew = envTest.step(optAct)
                        if self.plot and i == 0:
                            envTest.plotState(policyPath + "/tmp" + str(self.testNum) + "/")
                        steps += 1
                        state = self.stateMap[tuple(state)]
                        ret = self.gamma**(steps) * rew + ret
                        done = envTest.isDone()
                        if done or steps > MAXSTEPS:
                            doneFlag = True
                            break

                        optAct = self.getGreedyQopt(state, act)

                    if doneFlag:
                        break
            allRet.append(ret)

        mean = np.mean(allRet)
        var = np.var(allRet)
        var = var ** 0.5
        return mean, var

    def getGreedyQ(self, state):
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def getGreedyQtest(self, state):
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def getGreedyQ2(self, state):
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        return maxvals[0]

    def getGreedyQopt(self, state, op):
        op = op - self.actSize
        qvals = self.Qopt[op][state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def getGreedyQoptList(self, state, op):
        op = op - self.actSize
        qvals = self.Qopt[op][state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        return list(maxvals)

    def savePolicy(self, path):
        fp = open(path, "wb")
        pickle.dump((self.Q, self.stateMap), fp)
        fp.close()

    def loadPolicy(self, path):
        fp = open(path, "rb")
        self.Q, self.stateMap = pickle.load(fp)
        fp.close()


class SimpleQLearnerOverOptions(object):
    def __init__(self, env, actSize=5, optSize=10, gamma=0.9999, optionPath=None, eigen=False):
        self.env = env
        self.gamma = gamma
        self.testNum = 0
        self.plot = False
        self.eigen = eigen
        self.actionSize = actSize
        self.optSize = optSize
        self.stateSize = (self.env.height * self.env.width)
        self.Q = np.zeros((self.stateSize, self.actionSize))

        self.loadOptions(optionPath)

    def loadOptions(self, path):
        """
        Load the Q-values(option policy) of options into self.Qopt
        """
        self.Qopt = []
        if self.eigen:
            prefix = "eigen"
        else:
            prefix = "sr"

        for i in range(self.optSize):
            filePath = path + prefix + "policy" + str(i) + ".csv"
            fp = open(filePath, "rb")
            curOptQvals, stateMap = pickle.load(fp)
            fp.close()

            self.Qopt.append(deepcopy(curOptQvals))

        if self.optSize != 0:
            self.stateMap = deepcopy(stateMap)
        else:
            filePath = path + prefix + "policy0.csv"
            fp = open(filePath, "rb")
            curOptQvals, stateMap = pickle.load(fp)
            fp.close()
            self.stateMap = deepcopy(stateMap)

    def train(self, num_episodes=int(500), policyPath="tmp"):

        EPS_START = 1
        EPS_END = 0
        alpha = 0.9
        allRet = []
        behavior_opt = np.arange(self.actionSize + len(self.Qopt))

        for i in tqdm(range(num_episodes)):

            steps = 0
            ret = 0
            state = self.stateMap[tuple(self.env.reset())]
            prevState = int(state)
            while True:

                option_sampled = np.random.choice(behavior_opt)

                # sampled an option, run it till it terminates
                if option_sampled >= self.actionSize:

                    #print("following option policy now...")
                    while True:

                        act = self.getGreedyQOption(prevState, option_sampled - self.actionSize)
                        if act == 0: #prevState == maxState:
                            #print("option successfully completed", act, maxState, option_sampled)
                            break
                        else:
                            state, rew = self.env.step(act)
                            steps += 1
                            #self.env.render()

                            state = self.stateMap[tuple(state)]
                            actNext = self.getGreedyQ(state)

                            self.Q[prevState, act] = (1 - alpha) * self.Q[prevState, act] + \
                                alpha * (rew + self.gamma * self.Q[state, actNext])

                            prevState = int(state)
                            if self.env.isDone() or steps >= 100:
                                break

                # sampled a primitive action
                else:
                    #print("following primitive actions")
                    act = option_sampled
                    state, rew = self.env.step(act)
                    steps += 1

                    state = self.stateMap[tuple(state)]
                    actNext = self.getGreedyQ(state)

                    self.Q[prevState, act] = (1 - alpha) * self.Q[prevState, act] + \
                        alpha * (rew + self.gamma * self.Q[state, actNext])

                    prevState = int(state)

                if self.env.isDone() or steps >= 100:
                    break

        mu, var = self.test(policyPath)
        print("mean reward", mu)
        return self.Q

    def test(self, policyPath):
        self.testNum += 1
        envTest = deepcopy(self.env)
        envTest.setRender()

        allRet = []
        for i in range(3):
            state = self.stateMap[tuple(envTest.reset())]

            ret = 0.0
            steps = 0
            MAXSTEPS = 50
            while steps <= MAXSTEPS:
                act = self.getGreedyQ(state)
                if act < self.actionSize:
                    state, rew = envTest.step(act)
                    envTest.render()
                    if self.plot and i == 0:
                        envTest.plotState(policyPath + "/tmp" + str(self.testNum) + "/")
                    steps += 1
                    state = self.stateMap[tuple(state)]

                    ret = ret + self.gamma**(steps) * rew
                    done = envTest.isDone()
                    if done or steps > MAXSTEPS:
                        break

                else:
                    optAct = self.getGreedyQopt(state, act)
                    doneFlag = False

                    #  Execute option until you reach terminal state
                    while self.Qopt[act-self.actionSize][state, optAct] > 0:

                        state, rew = envTest.step(optAct)
                        envTest.render()
                        if self.plot and i == 0:
                            envTest.plotState(policyPath + "/tmp" + str(self.testNum) + "/")
                        steps += 1
                        state = self.stateMap[tuple(state)]
                        ret = self.gamma**(steps) * rew + ret
                        done = envTest.isDone()
                        if done or steps > MAXSTEPS:
                            doneFlag = True
                            break

                        optAct = self.getGreedyQopt(state, act)

                    if doneFlag:
                        break
            allRet.append(ret)

        mean = np.mean(allRet)
        var = np.var(allRet)
        var = var ** 0.5
        return mean, var

    def getGreedyQ(self, state):
        qvals = self.Q[state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def getGreedyQOption(self, state, id):
        qvals = self.Qopt[id][state, :]
        maxvals = np.argwhere(qvals == np.amax(qvals)).flatten().tolist()
        ind = np.random.randint(len(maxvals))
        return maxvals[ind]

    def savePolicy(self, path):
        fp = open(path, "wb")
        pickle.dump((self.Q, self.stateMap), fp)
        fp.close()

    def loadPolicy(self, path):
        fp = open(path, "rb")
        self.Q, self.stateMap = pickle.load(fp)
        fp.close()
