import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import numpy as np
import os

from Gym.tools.utils import MemoryDataset
from .base import IModels

Trajectory = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class QLearningBase(IModels):
    def __init__(
        self,
        device,
        #
        netEval,
        netTarget,
        optimizer,
        #
        n_actions,
        #
        learning_rate,
        gamma,
        tau,
        #
        updateTargetFreq,
        #
        epsilonStart,
        epsilonEnd,
        epsilonDecayFreq,
        #
        mSize,
        batchSize,
        startTrainSize,
        #
        transforms,
    ):
        """
        gamma: reward 的衰減係數
        tau: update target 的係數，target = eval + tau * (target - eval)
        epsilonStart: epsilon 的起始值，隨機的機率，越大越隨機
        epsilonEnd: epsilon 的終值
        epsilonDecayFreq: epsilon 衰減頻率
        updateTargetFreq: 更新 target 的頻率
        startTrainSize: 當 memory size 大於此值時，便開始訓練
        """
        super().__init__(device, mSize, transforms)

        self.device = device

        self.netEval = netEval.to(self.device)
        self.netTarget = netTarget.to(self.device)

        print("device", self.device)
        print(self.netEval)

        self.n_actions = n_actions

        # self.memory = MemoryDataset(mSize, transforms)
        self.batchSize = batchSize
        self.startTrainSize = startTrainSize

        self.lr = learning_rate
        # reward 衰減係數
        self.gamma = gamma
        # target = eval + tau * (target - eval)
        self.tau = tau

        self.updateTargetFreq = updateTargetFreq

        self.epsilonStart = epsilonStart
        self.epsilon = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecayFreq = epsilonDecayFreq

        # optimizer 是訓練的工具
        self.optimizer = optimizer

        # 從 1 開始，以免 updateTarget
        self.trainStep = 1

    def decay_parameters(self):
        self.epsilon = max(
            self.epsilonEnd, self.epsilonStart - self.trainStep / self.epsilonDecayFreq
        )

    def choose_action(self, state):
        randomChoice = np.random.choice([0, 1], p=((1 - self.epsilon), self.epsilon))

        # epslion greedy
        if randomChoice == 1 and self.netEval.training:
            action = np.random.choice(range(self.n_actions), 1)
        else:
            # state = torch.unsqueeze(self.transforms(state), dim=0).to(self.device)
            state = torch.FloatTensor(state)
            state = torch.unsqueeze(state, dim=0).to(self.device)
            qValues = self.netEval(state)
            action_max_value, action = torch.max(qValues, 1)

        return action.item()

    def store_trajectory(self, s, a, r, done, s_):
        self.memory.add(s, a, r, done, s_)

    def train_step(self):
        if len(self.memory) < self.startTrainSize:
            return

        self.train()
        self.decay_parameters()

        self.trainStep += 1
        self.update_target()

    def train_episode(self):
        pass

    def train(self):
        batch = Trajectory(*zip(*self.memory.sample(self.batchSize)))

        # 轉成 np.array 加快轉換速度
        s = np.array(batch.state)
        a = np.array(batch.action)
        r = np.array(batch.reward)
        done = np.array(batch.done)
        s_ = np.array(batch.next_state)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.tensor(a).long().to(self.device)
        a = torch.unsqueeze(a, dim=1)  # 在 dim=1 增加維度 ex: (50,) => (50,1)
        r = torch.FloatTensor(r).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)

        # caculate Target
        futureQValue = self.netTarget(s_)
        futureQValue, _ = futureQValue.max(dim=1)

        # done 是關鍵之一，不導入計算會導致 qNext 預估錯誤
        # 這也是讓 qValue 收斂的要素，不然 target 會一直往上累加，進而估不準
        target = r + self.gamma * futureQValue * (1 - done)
        # detach from graph, don't backpropagate
        target = target.detach()

        # 在 dim=1，以 a 為 index 取值
        predict = self.netEval(s).gather(dim=1, index=a).squeeze(1)

        lossFun = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = lossFun(predict, target)
        loss.backward()
        self.optimizer.step()

        # print(list(self.netEval.parameters()))
        # print("=============================================")

        return loss

    # 逐步更新 target net
    def update_target(self):
        if (self.trainStep % self.updateTargetFreq) != 0:
            return

        print(f"Update target network... tau={self.tau}")
        for paramEval, paramTarget in zip(
            self.netEval.parameters(), self.netTarget.parameters()
        ):
            paramTarget.data = paramEval.data + self.tau * (
                paramTarget.data - paramEval.data
            )

    # def get_default_params_path(self, filePath, envName):
    #    _dirPath = os.path.dirname(os.path.realpath(filePath))
    #    paramsPath = os.path.join(
    #        _dirPath, f"params_{envName}_{type(self).__name__}_{self.device.type}.pkl"
    #    )

    #    return paramsPath

    def save_models(self, paramsPath):
        print(f"Save parameters to {paramsPath}")
        torch.save(self.netEval.state_dict(), paramsPath)

    def load_models(self, paramsPath):
        if not os.path.exists(paramsPath):
            return False

        print(f"Load parameters from {paramsPath}")
        self.netEval.load_state_dict(torch.load(paramsPath, map_location=self.device))
        self.netEval.load_state_dict(torch.load(paramsPath, map_location=self.device))

        self.epsilonStart = self.epsilonEnd
        self.epsilon = self.epsilonEnd

        return True

    def eval(self):
        print("Evaluation Mode")
        self.netEval.eval()

    def print_info(self):
        print(f"totalTrainStep:{self.trainStep:7d} epsilon:{self.epsilon:.2f}")
